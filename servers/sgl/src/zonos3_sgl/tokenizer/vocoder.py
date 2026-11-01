"""TTS vocoder manager: undelays the LM's codebook rows and decodes them with the registry's matched codec."""

from __future__ import annotations
from zonos3.tokenizer.matched_codec import codec_names as matched_codec_names, entry as matched_codec_entry, decode as decode_matched

import os
from typing import Dict, List

import numpy as np
import torch
from zonos3_sgl.message import TTSDetokenizeMsg

from zonos3.tts.delay import shear_up


def decode_audio_codes(codes: torch.Tensor, audio_codec: str) -> torch.Tensor:
    if audio_codec in matched_codec_names():
        return decode_matched(codes,audio_codec)
    raise ValueError(f"Unsupported audio codec: {audio_codec!r}")


class TTSVocoderManager:
    """Converts audio codebook tokens to PCM audio with the registry's matched codec.

    The model outputs tokens with a delay pattern where codebook j is delayed by j frames; shear_up removes this
    delay once n_codebooks - 1 later rows have arrived. The matched codecs are long-context neural decoders, so a
    request is decoded whole at EOS (chunked overlap-add is not numerically equivalent to the reference decode),
    unless it asks for the progressive prefix decode (stream_decode="prefix").
    """

    def __init__(
        self,
        n_codebooks: int,
        audio_pad_id: int,
        audio_codec: str,
        sample_rate: int = 44100,
        overlap_frames: int = 4,
        prefix_chunk_frames: int | None = None,
        prefix_hold_back_frames: int | None = None,
        prefix_context_frames: int | None = None,
    ):
        """Initialize the vocoder manager.

        Args:
            n_codebooks: Number of audio codebooks
            audio_pad_id: Audio padding token ID (used in shear_up)
            audio_codec: Matched codec name (a ``codec`` of the codec registry)
            sample_rate: Output sample rate
            overlap_frames: Already-counted rows (the silence prefill) re-decoded as left context at EOS
        """
        self.audio_codec = str(audio_codec).strip().lower().replace("-", "_")
        if self.audio_codec not in matched_codec_names():
            raise ValueError(f"Unsupported audio codec: {self.audio_codec!r}")
        specification=matched_codec_entry(self.audio_codec)
        if n_codebooks!=specification['codebooks'] or audio_pad_id!=4097:
            raise ValueError('Matched codec codebook/padding mismatch')
        self.n_codebooks = n_codebooks
        self.audio_pad_id = audio_pad_id
        self.eoa_id = audio_pad_id - 1
        self.sample_rate = sample_rate
        self.overlap_frames = overlap_frames
        self.hop_length = int(specification['hop'])
        # Opt-in progressive decode for the matched long-context codecs (request
        # field stream_decode="prefix"): the causal decoder re-decodes the growing
        # prefix holding back the last frames, and the exact whole decode at EOS
        # supplies the tail. Off unless a request asks for it: the default output
        # is unchanged. Defaults = the arena's measured settings (1 s chunks, 3
        # frames held back, 450 frames = 30 s of left context, -48 dB vs whole).
        self.prefix_stream_supported = True
        env = os.environ.get
        self.prefix_chunk_frames = int(prefix_chunk_frames if prefix_chunk_frames is not None else env("ZONOS3_PREFIX_STREAM_CHUNK_FRAMES", 15))
        self.prefix_hold_back_frames = int(prefix_hold_back_frames if prefix_hold_back_frames is not None else env("ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES", 3))
        self.prefix_context_frames = int(prefix_context_frames if prefix_context_frames is not None else env("ZONOS3_PREFIX_STREAM_CONTEXT_FRAMES", 450))
        if self.prefix_chunk_frames < 1 or self.prefix_hold_back_frames < 0 or self.prefix_context_frames < self.prefix_chunk_frames:
            raise ValueError("prefix stream settings: chunk_frames >= 1, hold_back_frames >= 0, context_frames >= chunk_frames")
        self._prefix_mode: Dict[int, bool] = {}
        self._prefix_emitted: Dict[int, int] = {}
        self._prefix_live: Dict[int, List[np.ndarray]] = {}
        self._prefix_receipts: List[dict] = []
        self.last_prefix_receipt: dict | None = None

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
        self._silence_frame: torch.Tensor | None = None

    # ------------------------------------------------------------------ decode context / silence prefill (2026-10-17)
    @property
    def silence_prefill_default_frames(self) -> int:
        """ZONOS3_SILENCE_PREFILL_FRAMES (default 3 = 0.2 s at 15 fps): the prefill length a request gets when it does not
        name one."""
        try:
            return max(0, int(os.environ.get("ZONOS3_SILENCE_PREFILL_FRAMES", "3")))
        except ValueError:
            return 3

    @property
    def silence_prefill_default_on(self) -> bool:
        """ZONOS3_SILENCE_PREFILL_DEFAULT (default 1): prefill when a request leaves silence_prefill_frames unset."""
        return os.environ.get("ZONOS3_SILENCE_PREFILL_DEFAULT", "1").strip().lower() not in ("0", "false", "no", "off") \
            and self.silence_prefill_default_frames > 0

    def silence_prefill_frames_for(self, requested_frames: int | None) -> int:
        """How many silence frames a request gets: 0 = none. None = the server default; requested_frames
        (silence_prefill_frames) sets the length, 0 switches it off."""
        if not self.silence_prefill_supported:
            return 0
        if requested_frames is None:
            return self.silence_prefill_default_frames if self.silence_prefill_default_on else 0
        return max(0, int(requested_frames))

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
                print(f"[vocoder] codec-matched silence frame {self._silence_frame.tolist()} from the codec registry", flush=True)
                return self._silence_frame
            import io
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, np.zeros(2 * 44100, dtype=np.float32), 44100, format="WAV", subtype="FLOAT")
            codes = self.encode_reference_audio_bytes(buf.getvalue(), max_seconds=5.0)
            if codes.ndim != 2 or codes.shape[0] < 4:
                raise ValueError(f"silence encode returned {tuple(codes.shape)}")
            self._silence_frame = codes[-2].to(torch.int32).clone()
            print(f"[vocoder] codec-matched silence frame {self._silence_frame.tolist()} (last frames {codes[-4:].tolist()})", flush=True)
        return self._silence_frame

    def silence_prompt_rows(self, frames: int, prompt_config) -> torch.Tensor:
        from zonos3.tts.prompt import silence_rows_from_codes
        return silence_rows_from_codes(self.silence_frame_codes(), int(frames), prompt_config)

    def register_request_context(self, uid: int, *, prefill_rows=None, context_codes=None) -> None:
        """Called by the tokenizer before the request's first generated row reaches decode_frames."""
        rows = [[int(v) for v in row[: self.n_codebooks]] for row in prefill_rows.tolist()] if prefill_rows is not None else []
        ctx = None
        if context_codes is not None and len(context_codes):
            arr = context_codes.detach().cpu().numpy() if hasattr(context_codes, "detach") else np.asarray(context_codes)
            ctx = np.asarray(arr, dtype=np.int64).reshape(-1, self.n_codebooks)
        if rows or ctx is not None:
            self._pending_context[uid] = dict(rows=rows, context=ctx)
        else:
            self._pending_context.pop(uid, None)

    def _decode_with_context(self, uid: int, codes: torch.Tensor, at_stream_start: bool) -> np.ndarray:
        """decode_audio_codes with the request's context frames prepended when the slice begins at the stream
        start; the context's samples are dropped so sample 0 stays frame 0 of the slice."""
        context = self._context_codes.get(uid)
        if at_stream_start and context is not None and len(context):
            ctx = torch.as_tensor(context, dtype=codes.dtype, device=codes.device).unsqueeze(0)
            audio = decode_audio_codes(torch.cat((ctx, codes), dim=1), self.audio_codec)
            return audio[0].numpy().astype(np.float32)[len(context) * self.hop_length:]
        audio = decode_audio_codes(codes, self.audio_codec)
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
                if getattr(msg, "prefill_rows", None) is not None or getattr(msg, "decode_context_codes", None) is not None:
                    # the first row of a request carries its decode context from the tokenizer (another process)
                    self.register_request_context(uid, prefill_rows=getattr(msg, "prefill_rows", None),
                                                  context_codes=getattr(msg, "decode_context_codes", None))
                pending = self._pending_context.pop(uid, None) or {}
                prefill = [list(row) for row in (pending.get("rows") or [])]
                self._prefill_len[uid] = len(prefill)
                self._context_codes[uid] = pending.get("context")
                self._frame_buffers[uid] = prefill             # prefilled silence rows open the stream (never emitted)
                self._decoded_counts[uid] = len(prefill)
                self._prefix_mode[uid] = bool(
                    self.prefix_stream_supported and getattr(msg, "stream_decode", None) == "prefix"
                )
                if self._prefix_mode[uid]:
                    self._prefix_emitted[uid] = len(prefill)
                    self._prefix_live[uid] = []
            # Accumulate frame
            self._frame_buffers[uid].append(msg.audio_codes)

            # Do not rely solely on scheduler metadata for the codec boundary.
            # Tokenizer and scheduler run asynchronously, and under a busy
            # batched stream the detokenizer may observe the EOA-bearing row
            # before its aligned eos_frame hint. Derive the same boundary from
            # the delayed row locally and retain the earliest observation.
            eos_candidates = []
            if msg.eos_frame is not None:
                eos_candidates.append(max(0, int(msg.eos_frame)) + self._prefill_len.get(uid, 0))   # scheduler counts generated rows
            eos_columns = [
                index
                for index, token in enumerate(msg.audio_codes[: self.n_codebooks])
                if int(token) == self.eoa_id
            ]
            if eos_columns:
                row = len(self._frame_buffers[uid]) - 1
                eos_candidates.append(max(0, row - max(eos_columns)))
            if eos_candidates:
                observed = min(eos_candidates)
                previous = self._eos_frames.get(uid)
                self._eos_frames[uid] = (
                    observed if previous is None else min(previous, observed)
                )

            # Calculate how many complete frames we can decode
            total = len(self._frame_buffers[uid])
            # With N frames, can decode output frames 0 to N-(n_codebooks-1)-1
            complete_frames = total - (self.n_codebooks - 1)
            if uid in self._eos_frames:
                complete_frames = min(complete_frames, self._eos_frames[uid])
            # End of stream: decode everything before EOS in one call.
            should_decode = msg.finished and complete_frames > self._decoded_counts[uid]

            if self._prefix_mode.get(uid):
                # Progressive decode: new prefix chunks as frames arrive, the exact
                # whole-decode tail (plus a receipt) when the request finishes.
                audio = self._decode_prefix_final(uid) if msg.finished else self._decode_prefix_chunk(uid, complete_frames)
                results.append(audio)
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
                self._prefix_mode.pop(uid, None)
                self._prefix_emitted.pop(uid, None)
                self._prefix_live.pop(uid, None)
                self._prefill_len.pop(uid, None)
                self._context_codes.pop(uid, None)

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
            raise ValueError(
                f"Raw audio-prefix conditioning has no encoder for matched codec {self.audio_codec!r}"
            )
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
        raw_end = min(target_frames + self.n_codebooks - 1, total)
        codes = shear_up(
            torch.tensor(buffer[decode_start:raw_end], dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu"),
            self.audio_pad_id,
        )[: target_frames - decode_start].unsqueeze(0)
        audio = self._decode_with_context(uid, codes, decode_start == 0)
        self._decoded_counts[uid] = target_frames
        return audio.tobytes()

    # ------------------------------------------------------------------ progressive (prefix) decode
    def pop_prefix_receipts(self) -> List[dict]:
        """Receipts (one per finished prefix-streamed request) accumulated since the last call."""
        receipts, self._prefix_receipts = self._prefix_receipts, []
        return receipts

    def _decode_prefix_rows(self, buffer: List[List[int]], start: int, frames_end: int, uid: int | None = None) -> np.ndarray:
        """Decode output frames [start, frames_end) from the delayed rows; the rows after
        frames_end (up to n_codebooks - 1 of them) complete the shear of the last frames."""
        raw_end = min(frames_end + self.n_codebooks - 1, len(buffer))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes = torch.tensor(buffer[start:raw_end], dtype=torch.int64, device=device)
        codes = shear_up(codes, self.audio_pad_id)[: frames_end - start].unsqueeze(0)
        if uid is None:
            audio = decode_audio_codes(codes, self.audio_codec)
            return audio[0].numpy().astype(np.float32)
        return self._decode_with_context(uid, codes, start == 0)

    def _decode_prefix_chunk(self, uid: int, complete_frames: int) -> bytes:
        """Emit the samples of the next chunk of frames from a prefix re-decode (causal decoder).

        decodable = complete frames minus the held-back tail: the held-back frames are decoded as right context
        but not emitted (the last decoded frame of the iSTFT head sits ~-22 dB from the whole decode; one frame of
        right context already reaches the bf16 floor near -48 dB). Once the history is longer than the context
        window only the last context_frames of left context are re-decoded, so the cost per chunk stays bounded."""
        decodable = complete_frames - self.prefix_hold_back_frames
        emitted = self._prefix_emitted[uid]
        if decodable - emitted < self.prefix_chunk_frames:
            return b""
        windowed = decodable > self.prefix_context_frames + self.prefix_chunk_frames
        start = max(0, emitted - self.prefix_context_frames) if windowed else 0
        audio = self._decode_prefix_rows(self._frame_buffers[uid], start, complete_frames, uid)
        chunk = audio[(emitted - start) * self.hop_length : (decodable - start) * self.hop_length].copy()
        self._prefix_emitted[uid] = decodable
        self._prefix_live[uid].append(chunk)
        return chunk.tobytes()

    def _decode_prefix_final(self, uid: int) -> bytes:
        """At EOS: the exact whole decode (the same call as the non-streaming path) supplies everything after
        the frames already streamed; the live chunks are compared with it for the receipt."""
        buffer = self._frame_buffers[uid]
        target_frames = len(buffer) - (self.n_codebooks - 1)
        if uid in self._eos_frames:
            target_frames = min(target_frames, self._eos_frames[uid])
        live_chunks = self._prefix_live.get(uid) or []
        emitted = min(self._prefix_emitted[uid], max(target_frames, 0))
        context = self._context_codes.get(uid)
        receipt = dict(uid=uid, frames=max(int(target_frames), 0), emitted_frames=int(emitted), chunks=len(live_chunks),
                       live_vs_whole_max_abs=0.0, live_vs_whole_rms_db=None,
                       prefill_frames=int(self._prefill_len.get(uid, 0)), context_frames=int(len(context)) if context is not None else 0)
        if target_frames <= 0:
            self.last_prefix_receipt = receipt
            self._prefix_receipts.append(receipt)
            return b""
        whole = self._decode_prefix_rows(buffer, 0, target_frames, uid)
        live = np.concatenate(live_chunks) if live_chunks else np.zeros(0, dtype=np.float32)
        skip = self._prefill_len.get(uid, 0) * self.hop_length           # prefilled silence: decoded, never emitted
        n = min(live.size, emitted * self.hop_length - skip, whole.size - skip)
        if n > 0:
            diff = live[:n] - whole[skip:skip + n]
            rms = float(np.sqrt(np.mean(np.square(whole[skip:skip + n], dtype=np.float64))))
            receipt["live_vs_whole_max_abs"] = float(np.abs(diff).max())
            receipt["live_vs_whole_rms_db"] = (
                round(float(20 * np.log10(float(np.sqrt(np.mean(np.square(diff, dtype=np.float64)))) / rms + 1e-12)), 1) if rms > 0 else None
            )
        self.last_prefix_receipt = receipt
        self._prefix_receipts.append(receipt)
        return whole[emitted * self.hop_length :].tobytes()

    def decode_all(
        self,
        codes: torch.Tensor,
        apply_shear_up: bool = True,
    ) -> torch.Tensor:
        """Decode a complete sequence of audio codes.

        Args:
            codes: Tensor of shape (batch, seq_len, n_codebooks)
            apply_shear_up: Whether to apply shear_up before decoding

        Returns:
            Audio tensor of shape (batch, num_samples)
        """
        if apply_shear_up:
            codes = shear_up(codes, self.audio_pad_id)

        codes = codes.to("cuda")
        return decode_audio_codes(codes, self.audio_codec)

    def reset(self, uid: int | None = None):
        """Reset frame buffers and decode state.

        Args:
            uid: If provided, reset only this request's state. Otherwise reset all.
        """
        if uid is not None:
            self._frame_buffers.pop(uid, None)
            self._decoded_counts.pop(uid, None)
            self._eos_frames.pop(uid, None)
            self._prefix_mode.pop(uid, None)
            self._prefix_emitted.pop(uid, None)
            self._prefix_live.pop(uid, None)
        else:
            self._frame_buffers.clear()
            self._decoded_counts.clear()
            self._eos_frames.clear()
            self._prefix_mode.clear()
            self._prefix_emitted.clear()
            self._prefix_live.clear()

# DECODE_CONTEXT_RUNTIME_20261017

# DECODE_CONTEXT_RUNTIME2_20261017

# SILENCE_DEFAULT_RUNTIME_20261017
