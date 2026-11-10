"""TTS vocoder manager: undelays the LM's codebook rows and decodes them with the registry's matched codec."""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Sequence

import numpy as np
import torch
from zonos3.generation import SAMPLE_RATE
from zonos3.tokenizer.matched_codec import capture_stream_graphs, served_decoders
from zonos3.tokenizer.matched_codec import codec_names as matched_codec_names
from zonos3.tokenizer.matched_codec import decode as decode_matched
from zonos3.tokenizer.matched_codec import entry as matched_codec_entry
from zonos3.tokenizer.matched_codec import stream as stream_matched
from zonos3.tts import plan
from zonos3.tts.delay import shear_up, undelay_window
from zonos3.tts.stream import PrefixDecoder, StreamSettings, prefix_chunks

from zonos3_sgl.message import TTSDetokenizeMsg


def decode_audio_codes(codes: torch.Tensor, audio_codec: str, decoder: str | None = None) -> torch.Tensor:
    """Decode through the codec's decoder ``decoder`` (None = its default)."""
    if audio_codec in matched_codec_names():
        return decode_matched(codes, audio_codec, decoder)
    raise ValueError(f"Unsupported audio codec: {audio_codec!r}")


class TTSVocoderManager:
    """Converts audio codebook tokens to PCM audio with the registry's matched codec.

    The model outputs tokens with a delay pattern where codebook j is delayed by j frames; shear_up removes this
    delay once n_codebooks - 1 later rows have arrived. The matched codecs are long-context neural decoders, so a
    request is decoded whole at EOS, unless it asks for the progressive prefix decode (stream_decode="prefix"): that
    decodes incrementally on a stateful codec stream (zonos3.tts.stream.PrefixDecoder) and flushes it at EOS. Each
    request decodes through its own decoder (``decoder`` of its messages: one of ``decoders``, None = the default).
    """

    def __init__(
        self,
        n_codebooks: int,
        audio_pad_id: int,
        audio_codec: str,
        sample_rate: int = SAMPLE_RATE,
        overlap_frames: int = 4,
        prefix_chunk_frames: int | None = None,
        prefix_hold_back_frames: int | None = None,
        prefix_context_frames: int | None = None,
        prefix_first_chunk_frames: int | None = None,
        decoders: Sequence[str] | None = None,
    ):
        """Initialize the vocoder manager.

        Args:
            n_codebooks: Number of audio codebooks
            audio_pad_id: Audio padding token ID (used in shear_up)
            audio_codec: Matched codec name (a ``codec`` of the codec registry)
            sample_rate: Output sample rate
            overlap_frames: Already-counted rows (the silence prefill) re-decoded as left context at EOS
            decoders: The decoder ids served (matched_codec.served_decoders; None = every decoder of the codec), the
                default first
        """
        self.audio_codec = str(audio_codec).strip().lower().replace("-", "_")
        if self.audio_codec not in matched_codec_names():
            raise ValueError(f"Unsupported audio codec: {self.audio_codec!r}")
        specification = matched_codec_entry(self.audio_codec)
        if n_codebooks != specification["codebooks"] or audio_pad_id != 4097:
            raise ValueError("Matched codec codebook/padding mismatch")
        self.decoders = served_decoders(self.audio_codec, decoders)
        self.n_codebooks = n_codebooks
        self.audio_pad_id = audio_pad_id
        self.eoa_id = audio_pad_id - 1
        self.sample_rate = sample_rate
        self.overlap_frames = overlap_frames
        self.hop_length = int(specification["hop"])
        # Opt-in progressive decode (request field stream_decode="prefix"): the frames are decoded incrementally on a
        # stateful codec stream, chunks ramping from first_chunk (1, 2, 4, 8 ...) up to chunk behind hold_back frames,
        # and at EOS the stream continues to the end and is flushed. Explicit settings win over ZONOS3_PREFIX_STREAM_*
        # (zonos3.tts.stream.StreamSettings).
        self.prefix_stream_supported = True
        explicit = {
            "chunk": prefix_chunk_frames,
            "first_chunk": prefix_first_chunk_frames,
            "hold_back": prefix_hold_back_frames,
            "context": prefix_context_frames,
        }
        self.prefix = dataclasses.replace(
            StreamSettings.from_env(), **{k: int(v) for k, v in explicit.items() if v is not None}
        )
        self._prefix: Dict[int, PrefixDecoder] = {}

        # Per-request state
        self._frame_buffers: Dict[int, List[List[int]]] = {}
        self._decoded_counts: Dict[int, int] = {}
        self._eos_frames: Dict[int, int] = {}
        # Decode context / silence prefill (2026-10-17): per request, rows that precede the generated rows in the
        # same stream (prefilled silence, delayed layout; kept in the row buffer for the undelay, never emitted) and
        # undelayed frames decoded as extra left context (the reference prompt's tail; prepended at decode time
        # whenever a decode starts at the stream start, their audio dropped).
        self._pending_context: Dict[int, dict] = {}
        self._prefill_len: Dict[int, int] = {}
        self._context_codes: Dict[int, np.ndarray | None] = {}
        self._decoder: Dict[int, str] = {}  # the request's decoder id
        self._silence_frame: torch.Tensor | None = None
        # Prefix streams that decoded P pure silence frames, and the samples that came out, per (decoder, P): a
        # request whose prefill rows are the silence prefill (no decode context) continues a fork, see _prefix_stream.
        self._primed: Dict[tuple[str, int], tuple[object, int]] = {}

    def request_decoder(self, decoder: str | None) -> str:
        """A message's decoder id (None = the default); ValueError for one this server does not serve."""
        if decoder is None:
            return self.decoders[0]
        if decoder not in self.decoders:
            raise ValueError(f"Unknown decoder {decoder!r}; this server serves: {', '.join(self.decoders)}")
        return decoder

    def warm_prefix_stream(self) -> None:
        """Before the first prefix request (CUDA only), for every served decoder: prime its codec stream on the default
        silence prefill and CUDA-graph the chunk shapes that follow it (the ramp 1, 2, 4, 8 ... until every KV cache is
        full, then the steady chunk). Other shapes decode eagerly; a replay is bit-identical to the eager decode."""
        if not torch.cuda.is_available():
            return
        prefill = self.silence_prefill_frames_for(None)
        for decoder in self.decoders:
            frames = stream_matched(self.audio_codec, decoder).full_frames + self.prefix.chunk
            capture_stream_graphs(self.audio_codec, prefix_chunks(prefill, self.prefix, frames), decoder)
            if prefill:
                self._primed_stream(decoder, prefill)

    def _primed_stream(self, decoder: str, frames: int) -> tuple[object, int]:
        if (decoder, frames) not in self._primed:
            stream = stream_matched(self.audio_codec, decoder)
            silence = self.silence_frame_codes().to(torch.int64).numpy()
            self._primed[decoder, frames] = (stream, len(stream.decode(np.repeat(silence[None], frames, 0))))
        return self._primed[decoder, frames]

    def _prefix_stream(self, decoder: str, prefill: List[List[int]], context) -> tuple[object, int]:
        """A prefix request's codec stream and the samples it already produced. With the silence prefill rows and no
        decode context the stream is a fork of one that decoded the prefill frames as pure silence, so the first chunk
        is generated frames only. Those frames' last codebooks are the LM's (row f + j > prefill): decoding them as
        silence moves the PCM ~1e-3 RMS, the size of a chunk-boundary change; the codes are unchanged."""
        if prefill and context is None and self.silence_prefill_supported:
            silence = self.silence_frame_codes().tolist()
            pad = self.audio_pad_id
            if all(
                row[j] == (silence[j] if j <= r else pad)
                for r, row in enumerate(prefill)
                for j in range(self.n_codebooks)
            ):
                stream, samples = self._primed_stream(decoder, len(prefill))
                return stream.fork(), samples
        return stream_matched(self.audio_codec, decoder), 0

    # ------------------------------------------------------------------ decode context / silence prefill (2026-10-17)
    def silence_prefill_frames_for(self, requested_frames: int | None) -> int:
        """How many silence frames a request gets (zonos3.tts.plan.silence_prefill_frames on this codec): 0 = none,
        None = the server default; an explicit silence_prefill_frames outside 0..registry max is a ValueError."""
        return plan.silence_prefill_frames(requested_frames, available=self.silence_prefill_supported)

    @property
    def silence_prefill_supported(self) -> bool:
        """Codec-matched silence frames exist for the matched dual-family codecs (they have the encoder)."""
        return matched_codec_entry(self.audio_codec).get("family") == "dual"

    def silence_frame_codes(self) -> torch.Tensor:
        """One codec-matched silence frame: the registry entry's ``silence_frame`` (the steady-state codes of 2 s of
        digital silence through the codec encoder), else encoded here (a frame well past the encoder's own start edge)."""
        if self._silence_frame is None:
            frame = matched_codec_entry(self.audio_codec).get("silence_frame")
            if frame is not None and len(frame) == self.n_codebooks:
                self._silence_frame = torch.tensor([int(v) for v in frame], dtype=torch.int32)
                print(
                    f"[vocoder] codec-matched silence frame {self._silence_frame.tolist()} from the codec registry",
                    flush=True,
                )
                return self._silence_frame
            import io

            import soundfile as sf

            buf = io.BytesIO()
            sf.write(buf, np.zeros(2 * 44100, dtype=np.float32), 44100, format="WAV", subtype="FLOAT")
            codes = self.encode_reference_audio_bytes(buf.getvalue(), max_seconds=5.0)
            if codes.ndim != 2 or codes.shape[0] < 4:
                raise ValueError(f"silence encode returned {tuple(codes.shape)}")
            self._silence_frame = codes[-2].to(torch.int32).clone()
            print(
                f"[vocoder] codec-matched silence frame {self._silence_frame.tolist()} (last frames {codes[-4:].tolist()})",
                flush=True,
            )
        return self._silence_frame

    def silence_prompt_rows(self, frames: int, prompt_config) -> torch.Tensor:
        from zonos3.tts.prompt import silence_rows_from_codes

        return silence_rows_from_codes(self.silence_frame_codes(), int(frames), prompt_config)

    def register_request_context(self, uid: int, *, prefill_rows=None, context_codes=None) -> None:
        """Called by the tokenizer before the request's first generated row reaches decode_frames."""
        rows = (
            [[int(v) for v in row[: self.n_codebooks]] for row in prefill_rows.tolist()]
            if prefill_rows is not None
            else []
        )
        ctx = None
        if context_codes is not None and len(context_codes):
            arr = (
                context_codes.detach().cpu().numpy() if hasattr(context_codes, "detach") else np.asarray(context_codes)
            )
            ctx = np.asarray(arr, dtype=np.int64).reshape(-1, self.n_codebooks)
        if rows or ctx is not None:
            self._pending_context[uid] = {"rows": rows, "context": ctx}
        else:
            self._pending_context.pop(uid, None)

    def _decode_with_context(self, uid: int, codes: torch.Tensor, at_stream_start: bool) -> np.ndarray:
        """decode_audio_codes with the request's context frames prepended when the slice begins at the stream
        start; the context's samples are dropped so sample 0 stays frame 0 of the slice."""
        context = self._context_codes.get(uid)
        decoder = self._decoder[uid]
        if at_stream_start and context is not None and len(context):
            ctx = torch.as_tensor(context, dtype=codes.dtype, device=codes.device).unsqueeze(0)
            audio = decode_audio_codes(torch.cat((ctx, codes), dim=1), self.audio_codec, decoder)
            return audio[0].numpy().astype(np.float32)[len(context) * self.hop_length :]
        audio = decode_audio_codes(codes, self.audio_codec, decoder)
        return audio[0].numpy().astype(np.float32)

    def decode_frames(self, msgs: List[TTSDetokenizeMsg]) -> List[bytes]:
        """Decode audio frames to PCM chunks.

        Accumulates the delayed rows of each request. The shear pattern requires n_codebooks-1 extra rows to
        complete an output frame: with N accumulated rows, output frames 0 to N-n_codebooks are complete. At end of
        stream every pre-EOS frame is decoded in one call (prefix-mode requests also emit progressive chunks);
        tail rows generated for delay alignment are kept as context but sliced away before the codec decode.

        Args:
            msgs: List of TTS detokenize messages

        Returns:
            List of PCM audio bytes (float32, 44.1kHz), one per message
        """
        results = []
        for msg in msgs:
            uid = msg.uid

            # Initialize state if needed
            if uid not in self._frame_buffers:
                if (
                    getattr(msg, "prefill_rows", None) is not None
                    or getattr(msg, "decode_context_codes", None) is not None
                ):
                    # the first row of a request carries its decode context from the tokenizer (another process)
                    self.register_request_context(
                        uid,
                        prefill_rows=getattr(msg, "prefill_rows", None),
                        context_codes=getattr(msg, "decode_context_codes", None),
                    )
                pending = self._pending_context.pop(uid, None) or {}
                prefill = [list(row) for row in (pending.get("rows") or [])]
                self._prefill_len[uid] = len(prefill)
                self._context_codes[uid] = pending.get("context")
                self._decoder[uid] = self.request_decoder(getattr(msg, "decoder", None))
                self._frame_buffers[uid] = prefill  # prefilled silence rows open the stream (never emitted)
                self._decoded_counts[uid] = len(prefill)
                if self.prefix_stream_supported and getattr(msg, "stream_decode", None) == "prefix":
                    stream, primed_samples = self._prefix_stream(
                        self._decoder[uid], prefill, self._context_codes.get(uid)
                    )
                    self._prefix[uid] = PrefixDecoder(
                        stream,
                        n_codebooks=self.n_codebooks,
                        audio_pad_id=self.audio_pad_id,
                        hop=self.hop_length,
                        prefill=len(prefill),
                        context_codes=self._context_codes.get(uid),
                        settings=self.prefix,
                        primed_samples=primed_samples,
                        codebooks=msg.stream_codebooks,
                        first_only=msg.stream_codebooks_scope == "first",
                    )
            # Accumulate frame
            self._frame_buffers[uid].append(msg.audio_codes)

            # Do not rely solely on scheduler metadata for the codec boundary.
            # Tokenizer and scheduler run asynchronously, and under a busy
            # batched stream the detokenizer may observe the EOA-bearing row
            # before its aligned eos_frame hint. Derive the same boundary from
            # the delayed row locally and retain the earliest observation.
            eos_candidates = []
            if msg.eos_frame is not None:
                eos_candidates.append(
                    max(0, int(msg.eos_frame)) + self._prefill_len.get(uid, 0)
                )  # scheduler counts generated rows
            eos_columns = [
                index for index, token in enumerate(msg.audio_codes[: self.n_codebooks]) if int(token) == self.eoa_id
            ]
            if eos_columns:
                row = len(self._frame_buffers[uid]) - 1
                eos_candidates.append(max(0, row - max(eos_columns)))
            if eos_candidates:
                observed = min(eos_candidates)
                previous = self._eos_frames.get(uid)
                self._eos_frames[uid] = observed if previous is None else min(previous, observed)

            # Calculate how many complete frames we can decode
            total = len(self._frame_buffers[uid])
            # With N frames, can decode output frames 0 to N-(n_codebooks-1)-1
            complete_frames = total - (self.n_codebooks - 1)
            if uid in self._eos_frames:
                complete_frames = min(complete_frames, self._eos_frames[uid])
            # End of stream: decode everything before EOS in one call.
            should_decode = msg.finished and complete_frames > self._decoded_counts[uid]

            prefix = self._prefix.get(uid)
            if prefix is not None:
                # Progressive decode: the next ramped chunk on the request's codec stream; at EOS the rest + flush.
                buffer = self._frame_buffers[uid]
                if msg.finished:
                    audio = prefix.finish(buffer, complete_frames)
                else:
                    audio = prefix.step(buffer, complete_frames, self._eos_frames.get(uid))
                results.append(audio.tobytes())
            elif should_decode:
                audio = self._decode_final(uid)
                results.append(audio)
            else:
                results.append(b"")

            # Cleanup finished requests
            if msg.finished:
                self._frame_buffers.pop(uid, None)
                self._decoded_counts.pop(uid, None)
                self._eos_frames.pop(uid, None)
                self._prefix.pop(uid, None)
                self._prefill_len.pop(uid, None)
                self._context_codes.pop(uid, None)
                self._decoder.pop(uid, None)

        return results

    def encode_reference_audio_bytes(
        self,
        audio_bytes: bytes,
        *,
        max_seconds: float = 30.0,
    ) -> torch.Tensor:
        """Reference audio (any container ffmpeg/soundfile reads) -> int32 codes [frames, n_codebooks] of the dual codec."""
        from zonos3.tts.audio import _decode_audio_bytes

        if matched_codec_entry(self.audio_codec).get("family") != "dual":
            raise ValueError(f"Raw audio-prefix conditioning has no encoder for matched codec {self.audio_codec!r}")
        from zonos3.tokenizer.dual_reference import encode_dual_reference

        waveform, sample_rate = _decode_audio_bytes(audio_bytes)
        return encode_dual_reference(waveform, sample_rate, max_seconds=max_seconds)

    def _decode_final(self, uid: int) -> bytes:
        """Decode every complete pre-EOS frame not yet counted, in one codec call (end of stream).

        A generated row contains codebook c for output frame (row - c), so the last n_codebooks - 1 output frames
        are incomplete unless later delayed rows arrive. A max-token finish has no such flush, so decoding all rows
        would feed audio padding into the codec; EOS finishes provide the delayed flush rows and are further bounded
        by the aligned eos_frame. The last ``overlap_frames`` already-counted rows (the silence prefill) are
        re-decoded as left context and emitted with the rest.
        """
        buffer = self._frame_buffers[uid]
        decoded = self._decoded_counts[uid]
        total = len(buffer)
        target_frames = total - (self.n_codebooks - 1)
        if uid in self._eos_frames:
            target_frames = min(target_frames, self._eos_frames[uid])
        if target_frames <= decoded:
            return b""
        decode_start = decoded - min(self.overlap_frames, decoded)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes = undelay_window(
            buffer, decode_start, target_frames, self.n_codebooks, self.audio_pad_id, device=device
        ).unsqueeze(0)
        audio = self._decode_with_context(uid, codes, decode_start == 0)
        self._decoded_counts[uid] = target_frames
        return audio.tobytes()

    def decode_all(
        self,
        codes: torch.Tensor,
        apply_shear_up: bool = True,
        decoder: str | None = None,
    ) -> torch.Tensor:
        """Decode a complete sequence of audio codes.

        Args:
            codes: Tensor of shape (batch, seq_len, n_codebooks)
            apply_shear_up: Whether to apply shear_up before decoding
            decoder: A served decoder id (None = the default)

        Returns:
            Audio tensor of shape (batch, num_samples)
        """
        if apply_shear_up:
            codes = shear_up(codes, self.audio_pad_id)

        codes = codes.to("cuda")
        return decode_audio_codes(codes, self.audio_codec, self.request_decoder(decoder))

    def reset(self, uid: int | None = None):
        """Reset frame buffers and decode state.

        Args:
            uid: If provided, reset only this request's state. Otherwise reset all.
        """
        if uid is not None:
            self._frame_buffers.pop(uid, None)
            self._decoded_counts.pop(uid, None)
            self._eos_frames.pop(uid, None)
            self._prefix.pop(uid, None)
            self._decoder.pop(uid, None)
        else:
            self._frame_buffers.clear()
            self._decoded_counts.clear()
            self._eos_frames.clear()
            self._prefix.clear()
            self._decoder.clear()
