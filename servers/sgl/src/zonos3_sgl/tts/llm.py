"""Offline TTSLLM class for batch TTS generation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Dict, List

import torch
from pathlib import Path
from zonos3_sgl.distributed import DistributedInfo
from zonos3_sgl.message import (
    BaseTTSBackendMsg,
    BatchTTSTokenizerMsg,
    TTSDetokenizeMsg,
    TTSUserMsg,
)
from zonos3_sgl.message.tts import TTSSamplingParams
from zonos3_sgl.scheduler import SchedulerConfig
from zonos3_sgl.scheduler.scheduler import TTSScheduler
from zonos3_sgl.tokenizer.vocoder import TTSVocoderManager
from zonos3.condition_buckets import check_ids
from zonos3.generation import SAMPLE_RATE
from zonos3.tts import plan
from zonos3.tts.conditioning import default_quality_buckets
from zonos3.tts.delay import shear_up

from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig


class RequestAllFinished(Exception):
    """Raised when all requests are finished."""

    pass


@dataclass
class TTSRequestStatus:
    """Status tracking for a TTS request."""

    uid: int
    input_ids: torch.Tensor  # 2D (seq_len, frame_width)
    output_frames: List[List[int]]  # List of audio code frames
    eos_frame: int | None = None
    finished: bool = False
    laci: dict | None = None  # LACI record of a LACI request (set with the final row)


@dataclass
class PendingRequest:
    """A tokenized prompt queued for offline generation.

    Mirrors the per-request fields the server forwards on a ``TTSUserMsg`` so the
    offline path conditions identically (text already tokenized into ``input_ids``).
    """

    input_ids: torch.Tensor  # 2D (seq_len, frame_width)
    sampling_params: TTSSamplingParams
    speaker_embedding: torch.Tensor | None = None
    # Raw-reference checkpoints: rows of the delayed reference prefix already prepended to input_ids.
    speaker_reference_prefix_length: int = 0
    # Condition rows (cond3 finetunes): bucket ids per table + the position of the first row in input_ids.
    condition_values: list[int] | None = None
    condition_token_start: int = -1
    accent: str | None = None
    accent_strength: float = 1.0
    # Alignment probe (zonos3_sgl.tts.align_probe) and the trailing prompt rows that count as repetition history.
    align_probe: object | None = None
    history_rows: int = 0
    # Audio prefill rows appended to input_ids (they open the audio stream), the LACI head-spec key and the LACI
    # duration target (zonos3.tts.laci.DurationTarget).
    prefill_rows: torch.Tensor | None = None
    laci: str | None = None
    laci_duration: object | None = None


class TTSLLM(TTSScheduler):
    """TTS-specific LLM interface for offline audio generation.

    This class provides a simplified interface for TTS generation with:
    - Text or pre-tokenized prompt input
    - Multi-codebook sampling with repetition penalty and top-k/top-p/min-p
    - EOS detection with frame alignment
    - Batch generation support
    - Optional audio decoding with the matched codec

    Example usage:
        tts = TTSLLM(model_path="/path/to/model")

        # Generate from text
        results = tts.generate(["Hello world"], TTSSamplingParams())

        # Results contain audio tokens and optionally decoded audio
        for r in results:
            print(r["audio_tokens"])  # List of frames
            if r["audio"]:
                # r["audio"] is PCM bytes at 44.1kHz
                pass
    """

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        n_codebooks: int | None = None,
        codebook_size: int | None = None,
        text_vocab: int | None = None,
        eoa_id: int | None = None,
        audio_pad_id: int | None = None,
        decode_audio: bool = True,
        **kwargs,
    ):
        """Initialize TTSLLM.

        Args:
            model_path: Path to the model checkpoint
            dtype: Data type for model (default: bfloat16)
            n_codebooks: Number of audio codebooks (optional, auto-detected from model)
            codebook_size: Size of each codebook vocabulary (optional, auto-detected from model)
            text_vocab: Text vocabulary size (optional, auto-detected from model)
            eoa_id: End-of-audio token ID (optional, auto-detected from model)
            audio_pad_id: Audio padding token ID (optional, auto-detected from model)
            decode_audio: Whether to decode audio with the matched codec (default: True)
            **kwargs: Additional arguments passed to scheduler
        """
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)

        # Override TTS settings only if explicitly provided (otherwise use model config values)
        if n_codebooks is not None:
            self.n_codebooks = n_codebooks
        if codebook_size is not None:
            self.codebook_size = codebook_size
        if eoa_id is not None:
            self.eoa_id = eoa_id
        if audio_pad_id is not None:
            self.audio_pad_id = audio_pad_id
        if text_vocab is not None:
            self.text_vocab = text_vocab
        self.tts_sampler.audio_pad_id = self.audio_pad_id
        self.tts_sampler.end_token_ids = (
            (self.eoa_id,) if self.span_end_id is None else (self.eoa_id, self.span_end_id)
        )

        self.decode_audio = decode_audio

        # Text normalization, gated exactly like the server tokenizer worker so the
        # offline path produces the same tokens (e.g. "123" -> "one hundred ...").
        from zonos3.tokenizer.textnorm import TTSTextNormalizer, normalization_enabled

        self._text_normalizer = TTSTextNormalizer() if normalization_enabled() else None
        if self._text_normalizer is not None:
            # Compile the English grammars off the hot path, matching the server.
            threading.Thread(
                target=self._text_normalizer.warmup, args=(["en"],), daemon=True
            ).start()

        # Speaker encoder is loaded lazily on first embed_speaker* call.
        self._speaker_embedder = None

        # Request tracking
        self.pending_requests: List[PendingRequest] = []
        self.status_map: Dict[int, TTSRequestStatus] = {}
        self.counter = 0

        # Optional vocoder for audio decoding
        if decode_audio:
            self._vocoder = TTSVocoderManager(
                n_codebooks=self.n_codebooks,
                audio_pad_id=self.audio_pad_id,
                audio_codec=self.engine.model_config.audio_codec,
            )
        else:
            self._vocoder = None
        self._prompt_builder = TTSPromptBuilder(
            TTSPromptConfig(
                n_codebooks=self.n_codebooks,
                audio_pad_id=self.audio_pad_id,
                text_vocab=self.text_vocab,
                text_tokenizer_spec=config.text_tokenizer_spec,
                codebook_sizes=tuple(self.codebook_sizes) if getattr(self, "codebook_sizes", None) else None,
                speaking_rate_num_buckets=self.speaking_rate_num_buckets,
                quality_bucket_counts=tuple(self.quality_bucket_counts),
                speaker_background_num_buckets=self.speaker_background_num_buckets,
                accurate_mode_num_buckets=self.accurate_mode_num_buckets,
            )
        )

    @staticmethod
    def _broadcast_per_prompt(value, n: int) -> List:
        """Expand a scalar to one value per prompt, or pass a per-prompt list through.

        A plain ``list`` is treated as per-prompt; everything else (including a
        single ``torch.Tensor`` speaker embedding) is broadcast to every prompt.
        """
        if isinstance(value, list):
            if len(value) != n:
                raise ValueError(
                    f"Expected {n} per-prompt values, got {len(value)}."
                )
            return value
        return [value] * n

    def embed_speaker(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Compute a speaker embedding from a waveform, matching the server.

        Mirrors the server's _compute_speaker_embedding_from_waveform: runs the
        release speaker encoder and returns the float32 CPU embedding whose size
        matches the model's speaker_embedding_dim. The result can be passed
        directly as the ``speaker_embedding`` argument to generate().
        """
        if not self.speaker_enabled or self.speaker_embedding_dim <= 0:
            raise ValueError("Current model does not support speaker conditioning.")

        if self._speaker_embedder is None:
            import os

            from zonos3.models.speaker_cloning import create_speaker_embedder

            # Load the encoder on CPU by default (same env var the server uses) so
            # it does not permanently occupy GPU memory next to the TTS model.
            device = os.getenv("ZONOS3_SPEAKER_EMBEDDER_DEVICE", "cpu")
            self._speaker_embedder = create_speaker_embedder(
                self.speaker_embedding_codec,
                self.speaker_embedding_dim,
                device=device,
            )

        with torch.inference_mode():
            output = self._speaker_embedder(wav, sample_rate)

        if isinstance(output, tuple):
            candidates = [t.squeeze(0).to(dtype=torch.float32, device="cpu") for t in output]
        else:
            candidates = [output.squeeze(0).to(dtype=torch.float32, device="cpu")]

        for candidate in candidates:
            if candidate.numel() == self.speaker_embedding_dim:
                return candidate.contiguous()

        produced = ", ".join(str(c.numel()) for c in candidates)
        raise ValueError(
            f"Reference embedding dimension mismatch. Model expects "
            f"{self.speaker_embedding_dim}, but speaker encoder produced {produced}."
        )

    def embed_speaker_file(self, path: str) -> torch.Tensor:
        """Load an audio file and compute its speaker embedding (see embed_speaker).

        Decodes via the same ffmpeg path the server uses
        (``_decode_audio_bytes``) so the resulting embedding matches the server's
        for the same file; falls back to torchaudio if the server module (and its
        ffmpeg dependency) is unavailable.
        """
        with open(path, "rb") as f:
            audio_bytes = f.read()
        try:
            from zonos3.tts.audio import _decode_audio_bytes

            wav, sample_rate = _decode_audio_bytes(audio_bytes)
        except Exception:
            import torchaudio

            wav, sample_rate = torchaudio.load(path)
        return self.embed_speaker(wav, sample_rate)

    def _tokenize_one(
        self,
        prompt: str | List[List[int]],
        language: str = "en_us",
        text_normalization: bool = True,
    ) -> torch.Tensor:
        """Convert a prompt to 2D token tensor.

        Args:
            prompt: Text string or pre-tokenized unpacked tokens
            language: Server language code used for text normalization.
            text_normalization: Apply language-aware text normalization (matching
                the server) before tokenizing string prompts.

        Returns:
            2D tensor of shape (seq_len, frame_width)
        """
        if isinstance(prompt, str):
            text = prompt
            if text_normalization and self._text_normalizer is not None:
                text = self._text_normalizer.normalize(text, language)
            # legacy quality-conditioned checkpoints get the server's default quality tokens
            return self._prompt_builder.build(
                text, quality_buckets=default_quality_buckets(self.quality_features, self.quality_bucket_counts)
            )
        else:
            prompt_ids = prompt

        return torch.tensor(prompt_ids, dtype=torch.int32, device="cpu")

    def _resolve_reference_codes(self, codes, audio, n: int):
        """Per-prompt reference codes: given codes ([frames, n_codebooks]), or encoded from audio paths/waveforms."""
        if codes is not None and audio is not None:
            raise ValueError("Provide speaker_reference_codes or speaker_reference_audio, not both")
        if codes is None and audio is None:
            return [None] * n
        max_seconds = float(getattr(self.engine.model_config, "speaker_reference_max_seconds", 30.0))
        if audio is not None:
            items = audio if isinstance(audio, list) else [audio] * n
            if len(items) != n:
                raise ValueError("speaker_reference_audio must have one entry per prompt")
            resolved = []
            for item in items:
                if item is None:
                    resolved.append(None)
                    continue
                if isinstance(item, (str, Path)):
                    import soundfile as sf

                    wav, sr = sf.read(str(item), dtype="float32", always_2d=True)
                    waveform, sample_rate = torch.from_numpy(wav.T.copy()), int(sr)
                else:
                    waveform, sample_rate = item
                resolved.append(self.encode_reference(waveform, sample_rate, max_seconds=max_seconds))
            return resolved
        if torch.is_tensor(codes) or (isinstance(codes, list) and codes and isinstance(codes[0], (int, float))):
            items = [codes] * n
        elif isinstance(codes, list) and codes and isinstance(codes[0], list) and codes[0] and isinstance(codes[0][0], (int, float)):
            items = [codes] * n                      # one [frames, codebooks] list applied to every prompt
        else:
            items = list(codes)
        if len(items) != n:
            raise ValueError("speaker_reference_codes must have one entry per prompt")
        limit = plan.reference_frame_limit(max_seconds, self._reference_hop(), SAMPLE_RATE)
        out = []
        for item in items:
            if item is None:
                out.append(None)
                continue
            tensor = torch.as_tensor(item, dtype=torch.int32)
            if tensor.ndim != 2 or tensor.shape[1] != self.n_codebooks or tensor.shape[0] == 0:
                raise ValueError(f"reference codes must be [frames, {self.n_codebooks}]")
            out.append(plan.truncate_reference(tensor, limit))
        return out

    def _reference_hop(self) -> int:
        from zonos3.tokenizer.matched_codec import entry

        return int(entry(self.engine.model_config.audio_codec)["hop"])

    def encode_reference(self, waveform: torch.Tensor, sample_rate: int, *, max_seconds: float | None = None) -> torch.Tensor:
        """Reference waveform -> prefix codes with the checkpoint's codec (same path as the server tokenizer)."""
        from zonos3_sgl.tokenizer.vocoder import TTSVocoderManager

        if max_seconds is None:
            max_seconds = float(getattr(self.engine.model_config, "speaker_reference_max_seconds", 30.0))
        manager = getattr(self, "_reference_manager", None)
        if manager is None:
            manager = TTSVocoderManager(
                n_codebooks=self.n_codebooks,
                audio_pad_id=self.audio_pad_id,
                audio_codec=self.engine.model_config.audio_codec,
            )
            self._reference_manager = manager
        import io

        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, waveform.detach().cpu().float().T.numpy(), int(sample_rate), format="WAV", subtype="FLOAT")
        return manager.encode_reference_audio_bytes(buffer.getvalue(), max_seconds=max_seconds)

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseTTSBackendMsg]:
        """Receive messages from the pending queue."""
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()

        results: List[BaseTTSBackendMsg] = []
        added, sum_input_len = 0, 0

        for i, req in enumerate(self.pending_requests):
            if sum_input_len >= self.prefill_budget:
                break

            input_len = len(req.input_ids)
            sum_input_len += input_len
            uid = self.counter + added
            added += 1

            results.append(
                TTSUserMsg(
                    uid=uid,
                    input_ids=req.input_ids,
                    sampling_params=req.sampling_params,
                    # Speaker conditioning, identical to the server's TTSUserMsg: the shared scheduler injects the
                    # speaker slot (and a legacy checkpoint's background / accurate-mode markers) when an embedding
                    # is present, so the offline path only passes the embedding through.
                    speaker_embedding=req.speaker_embedding,
                    speaker_reference_prefix_length=int(req.speaker_reference_prefix_length),
                    condition_values=req.condition_values,
                    condition_token_start=int(req.condition_token_start),
                    accent=req.accent,
                    accent_strength=req.accent_strength,
                    align_probe=req.align_probe,
                    history_rows=int(req.history_rows),
                    prefill_rows=req.prefill_rows,
                    laci=req.laci,
                    laci_target_frames=None if req.laci_duration is None else float(req.laci_duration.frames),
                    laci_adjust_rate=True if req.laci_duration is None else bool(req.laci_duration.adjust_rate),
                )
            )

            self.status_map[uid] = TTSRequestStatus(
                uid=uid,
                input_ids=req.input_ids,
                output_frames=[],
            )

        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: BatchTTSTokenizerMsg) -> None:
        """Process results from the scheduler."""
        for msg in reply.data:
            assert isinstance(msg, TTSDetokenizeMsg)
            status = self.status_map[msg.uid]

            if not msg.finished:
                status.output_frames.append(msg.audio_codes)
            else:
                status.finished = True
                status.eos_frame = msg.eos_frame
                status.laci = msg.laci

    def _attach_extras(self, req: PendingRequest, rows, history, probe, where: str) -> None:
        """Audio prefill rows, repetition history and alignment probe of one queued request."""
        n_rows = 0
        if rows is not None:
            rows = torch.as_tensor(rows, dtype=req.input_ids.dtype)
            if rows.ndim != 2 or rows.shape[1] != req.input_ids.shape[1]:
                raise ValueError(
                    f"audio_prefill_rows{where} must be [T, {req.input_ids.shape[1]}], got {tuple(rows.shape)}"
                )
            req.input_ids = torch.cat((req.input_ids, rows), dim=0)
            req.prefill_rows = rows.to(torch.int32)
            n_rows = int(rows.shape[0])
        history = int(history or 0)
        if not 0 <= history <= n_rows:
            raise ValueError(f"audio_prefill_history{where}={history} must be in [0, {n_rows}] (its prefill rows)")
        if probe is not None:
            from zonos3_sgl.attention.fa import FlashAttentionBackend

            # The probe reads the request's own KV cache and replaces head outputs: no CFG twin, the FA cache layout.
            if float(getattr(req.sampling_params, "speaker_cfg_scale", 1.0)) != 1.0:
                raise ValueError("align_probes need speaker_cfg_scale == 1.0")
            if not isinstance(self.engine.attn_backend, FlashAttentionBackend):
                raise ValueError("align_probes need attention_backend='fa'")
        req.align_probe, req.history_rows = probe, history

    def edit(
        self, audio_codes, target_text, spans, sampling_params=None, *,
        language="en_us", text_normalization=True, decode_audio=None,
        speaker_embedding=None, condition_buckets=None, windows=None, min_frames=None,
    ) -> dict:
        """Generate masked regions independently and preserve unmasked codec frames.

        Each masked span is generated until the span-end token, so the checkpoint
        needs ``text_extra_tokens`` marker rows and a ``span_end_id``.
        ``speaker_embedding`` (speaker-embedding checkpoints) fills the speaker row and
        ``condition_buckets`` the condition rows in front of the text, as in training;
        edit prompts never carry a raw reference prefix (the unmasked audio is the voice).
        ``windows`` (``edit_align.plan_edit_windows``) restrict each region's prompt to its
        window's codes and text; by default the whole clip is the context. ``min_frames`` (one
        per span) masks the span-end token until that fill has as many frames.
        """
        from zonos3.tts.edit import (
            edit_marker_rows,
            edit_prompt_tokens,
            extract_fill,
            splice_edit,
            validate_edit_spans,
        )
        from zonos3.tts.edit_align import whole_clip_window

        model_config = self.engine.model_config
        span_end_id = model_config.span_end_id
        if span_end_id is None:
            raise ValueError(
                "Speech editing needs a checkpoint with a span-end audio token (span_end_id); "
                "this checkpoint only has the end-of-utterance token"
            )
        max_spans = edit_marker_rows(model_config.text_extra_tokens, self._prompt_builder.config.text_tokenizer_spec)
        if max_spans <= 0:
            raise ValueError("Speech editing needs edit marker rows; this checkpoint's extra text rows are phoneme tokens")
        codes = torch.as_tensor(audio_codes, dtype=torch.int64, device="cpu")
        validate_edit_spans(spans, len(codes), max_spans)
        config = self._prompt_builder.config
        limits = torch.tensor(config.codebook_sizes, dtype=torch.int64)
        if codes.ndim != 2 or codes.shape[1] != self.n_codebooks or bool(((codes < 0) | (codes >= limits)).any()):
            raise ValueError("Invalid input codec frames")
        sp = replace(sampling_params if sampling_params is not None else TTSSamplingParams(), end_token="span")
        fills, stopped = [None] * len(spans), [None] * len(spans)
        for window in windows or [whole_clip_window(spans, len(codes), target_text)]:
            text_rows = self._tokenize_one(
                window.text, language=language, text_normalization=text_normalization,
            )
            window_codes, window_spans, window_fills = codes[window.start: window.end], window.spans(spans), []
            for region in window.regions:
                prompt = edit_prompt_tokens(
                    text_rows, window_codes, window_spans, window_fills, config, span_end_id=span_end_id
                )
                result = self.generate(
                    [prompt.tolist()], decode_audio=False,
                    sampling_params=replace(sp, min_frames=min_frames[region] if min_frames else 0),
                    speaker_embedding=speaker_embedding, condition_buckets=condition_buckets,
                )[0]
                fill, ended = extract_fill(
                    result["audio_tokens"], result["eos_frame"],
                    audio_pad_id=self.audio_pad_id, codebook_sizes=config.codebook_sizes,
                )
                window_fills.append(fill)
                fills[region], stopped[region] = fill, ended
        spliced = splice_edit(codes, spans, fills)
        output = {"codes": spliced, "fills": fills, "stopped": stopped}
        should_decode = self.decode_audio if decode_audio is None else decode_audio
        if should_decode:
            if self._vocoder is None:
                raise ValueError("Audio decoding requires a vocoder")
            output["audio"] = self._vocoder.decode_all(
                spliced[None].cuda(), apply_shear_up=False,
            )[0].cpu().numpy().astype("float32")
        return output

    def generate(
        self,
        prompts: List[str] | List[List[List[int]]],
        sampling_params: TTSSamplingParams | List[TTSSamplingParams],
        decode_audio: bool | None = None,
        max_tokens: int | None = None,
        language: str = "en_us",
        text_normalization: bool = True,
        speaker_embedding: torch.Tensor | List[torch.Tensor | None] | None = None,
        speaker_reference_codes=None,
        speaker_reference_audio=None,
        condition_buckets=None,
        audio_prefill_rows=None,
        accent: str | List[str | None] | None = None,
        accent_strength: float | List[float] = 1.0,
        align_probes=None,
        audio_prefill_history=None,
        laci: bool | List[bool] = False,
        laci_duration=None,
    ) -> List[Dict]:
        """Generate audio tokens for a batch of prompts.

        Args:
            prompts: List of text strings or pre-tokenized prompts
            sampling_params: Sampling parameters (single or per-prompt)
            decode_audio: Override instance setting for audio decoding
            max_tokens: Per-request decode cap; clamped to the model limit and
                applied to every sampling_params (server parity).
            language: Server language code (e.g. "en_us") for text normalization.
            text_normalization: Apply language-aware text normalization, matching
                the server default.
            speaker_embedding: Speaker embedding tensor for voice cloning (single,
                applied to all prompts, or one per prompt; None disables cloning).
                Use embed_speaker()/embed_speaker_file() to compute one from audio.
            accent: Accent adapter name (single or per-prompt; None = base model),
                one of the adapters the engine was started with (accent_adapters).
            accent_strength: Adapter strength alpha (single or per-prompt): the
                request runs base + alpha * adapter.
            audio_prefill_rows: Sheared audio rows appended after each prompt (one tensor for all, or one per prompt);
                they open the audio stream (e.g. codec silence) and are not part of the returned frames.
            align_probes: One ``zonos3_sgl.tts.align_probe.AlignProbe`` (or None) per prompt.
            audio_prefill_history: Per prompt, how many trailing prefill rows count as repetition-penalty history.
            laci: LACI (zonos3.tts.laci), one flag for all prompts or one per prompt: the scheduler guards those
                generations against skips / hallucinations / truncation with the alignment-head spec of ``language``
                (``laci_heads`` of the TTSLLM config); speaker / controls CFG must be 1.
            laci_duration: per LACI prompt, a ``zonos3.tts.laci.DurationTarget`` (or None) its take's length is held
                to: a miss regenerates the take in the speaking-rate bucket the measured duration asks for
                (``adjust_rate``; needs ``cache_type="naive"``) or is only recorded.

        Returns:
            List of dicts with:
                - "audio_tokens": List of generated frames [[cb0, ..., cb8], ...]
                - "eos_frame": Frame index where EOS was detected (or None)
                - "audio": PCM audio bytes if decode_audio=True, else None
                - "sample_rate": Audio sample rate (zonos3.generation.SAMPLE_RATE)
                - "laci": the LACI record (attempts, end, failures, ...) of a LACI prompt, else None
        """
        # Validate the language code up front, as the server does.
        if text_normalization:
            from zonos3.languages import normalize

            language = normalize(language)

        # Reset state
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0

        # Normalize per-prompt arguments
        if isinstance(sampling_params, TTSSamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for sp in sampling_params:
            self._request_stop_id(sp)
        # Clamp max_tokens to the model limit (zonos3.tts.plan.max_tokens_budget) and apply to every request.
        if max_tokens is not None:
            clamped = plan.max_tokens_budget(max_tokens, self.engine.max_seq_len)
            sampling_params = [replace(sp, max_tokens=clamped) for sp in sampling_params]
        speaker_embeddings = self._broadcast_per_prompt(speaker_embedding, len(prompts))
        reference_codes = self._resolve_reference_codes(
            speaker_reference_codes, speaker_reference_audio, len(prompts)
        )
        # condition_buckets: one list of ids for every prompt, or one list per prompt (None entries = all unknown)
        if condition_buckets is None:
            condition_lists = [None] * len(prompts)
        elif len(condition_buckets) and isinstance(condition_buckets[0], (list, tuple)):
            if len(condition_buckets) != len(prompts):
                raise ValueError("condition_buckets must have one entry per prompt")
            condition_lists = [None if c is None else list(c) for c in condition_buckets]
        else:
            condition_lists = [list(condition_buckets)] * len(prompts)
        accents = self._broadcast_per_prompt(accent, len(prompts))
        accent_strengths = self._broadcast_per_prompt(accent_strength, len(prompts))
        for name in accents:
            if name is not None:
                if self.engine.accent_adapters is None:
                    raise ValueError("accent requested, but no accent adapters are loaded")
                self.engine.accent_adapters.slot(name)

        # Tokenize and queue all requests
        for prompt, sp, spk_emb, ref_codes, cond_ids, acc, strength in zip(
            prompts,
            sampling_params,
            speaker_embeddings,
            reference_codes,
            condition_lists,
            accents,
            accent_strengths,
        ):
            input_ids = self._tokenize_one(
                prompt,
                language=language,
                text_normalization=text_normalization,
            )
            # condition rows (cond3 finetunes) lead the text; the reference prefix goes in front of them and the
            # scheduler's speaker slot (offline path) shifts them by one more row
            condition_values, condition_token_start = None, -1
            if self.condition_count:
                ids = cond_ids if cond_ids is not None else list(self.condition_unknowns)
                condition_values, condition_token_start = check_ids(ids, self.condition_unknowns), 0
                input_ids = torch.cat(
                    (self._prompt_builder.condition_rows(self.condition_count, dtype=input_ids.dtype), input_ids), dim=0
                )
            elif cond_ids is not None:
                raise ValueError("This checkpoint has no condition rows (loudness / fmax / speaking rate)")
            prefix_length = 0
            if ref_codes is not None:
                if spk_emb is not None:
                    raise ValueError("speaker_embedding and a raw reference are mutually exclusive")
                prefix = self._prompt_builder.speaker_reference_prefix(ref_codes)
                prefix_length = int(prefix.shape[0])
                input_ids = torch.cat((prefix.to(torch.int32), input_ids), dim=0)
                if condition_values is not None:
                    condition_token_start += prefix_length
            elif float(getattr(sp, "speaker_cfg_scale", 1.0)) != 1.0 and spk_emb is None:
                raise ValueError("speaker_cfg_scale != 1 needs a speaker prompt (a raw reference or a speaker embedding)")
            self.pending_requests.append(
                PendingRequest(
                    input_ids=input_ids,
                    sampling_params=sp,
                    speaker_embedding=spk_emb,
                    speaker_reference_prefix_length=prefix_length,
                    condition_values=condition_values,
                    condition_token_start=condition_token_start,
                    accent=acc,
                    accent_strength=float(strength),
                )
            )

        # Audio prefill (2026-10-17): sheared audio rows (e.g. codec silence, prompt.silence_rows_from_codes) appended
        # after the text of each prompt; the LM continues the delay pattern from them. One tensor for all prompts or
        # one entry per prompt (None = no prefill). Callers prepend the same rows to audio_tokens before shear_up.
        # Alignment probes / repetition history (LACI): one entry per prompt.
        n = len(prompts)
        if audio_prefill_rows is None:
            rows_list = [None] * n
        elif isinstance(audio_prefill_rows, (list, tuple)):
            rows_list = list(audio_prefill_rows)
        else:
            rows_list = [audio_prefill_rows] * n
        probes = list(align_probes) if align_probes is not None else [None] * n
        history = list(audio_prefill_history) if audio_prefill_history is not None else [0] * n
        laci_on = self._broadcast_per_prompt(laci, n)
        durations = list(laci_duration) if isinstance(laci_duration, (list, tuple)) else [laci_duration] * n
        for name, values in (("audio_prefill_rows", rows_list), ("align_probes", probes),
                             ("audio_prefill_history", history), ("laci_duration", durations)):
            if len(values) != n:
                raise ValueError(f"{name} must have one entry per prompt")
        for idx, (req, rows, probe, hist, on, duration) in enumerate(
            zip(self.pending_requests, rows_list, probes, history, laci_on, durations)
        ):
            self._attach_extras(req, rows, hist, probe, f"[{idx}]")
            if on:
                if probe is not None:
                    raise ValueError(f"prompt {idx}: align_probes and laci are exclusive")
                self.check_laci(language, req.sampling_params)
                req.laci, req.laci_duration = language, duration
            elif duration is not None:
                raise ValueError(f"prompt {idx}: laci_duration needs laci")

        # Run generation
        try:
            self.run_forever()
        except RequestAllFinished:
            pass

        # Determine audio decoding
        should_decode = decode_audio if decode_audio is not None else self.decode_audio

        # Collect results in order
        results = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            audio_tokens = status.output_frames

            # Decode audio if requested
            audio_bytes = None
            if should_decode and audio_tokens and self._vocoder:
                # Convert to tensor, align delayed codebooks, then drop EOS and
                # post-EOS frames before the codec decode.
                codes = torch.tensor(audio_tokens, dtype=torch.int64, device="cuda")
                codes = shear_up(codes, self.audio_pad_id)
                if status.eos_frame is not None:
                    codes = codes[: max(0, status.eos_frame)]
                if codes.numel() == 0:
                    results.append(
                        {
                            "audio_tokens": audio_tokens,
                            "eos_frame": status.eos_frame,
                            "audio": b"",
                            "sample_rate": SAMPLE_RATE,
                            "laci": status.laci,
                        }
                    )
                    continue
                codes = codes.unsqueeze(0)  # Add batch dim

                # Decode with vocoder
                audio = self._vocoder.decode_all(codes, apply_shear_up=False)
                audio_bytes = audio[0].numpy().astype("float32").tobytes()

            results.append(
                {
                    "audio_tokens": audio_tokens,
                    "eos_frame": status.eos_frame,
                    "audio": audio_bytes,
                    "sample_rate": SAMPLE_RATE,
                    "laci": status.laci,
                }
            )

        return results

    def generate_one(
        self,
        prompt: str | List[List[int]],
        sampling_params: TTSSamplingParams | None = None,
        decode_audio: bool | None = None,
        max_tokens: int | None = None,
        language: str = "en_us",
        text_normalization: bool = True,
        speaker_embedding: torch.Tensor | None = None,
        accent: str | None = None,
        accent_strength: float = 1.0,
    ) -> Dict:
        """Generate audio for a single prompt.

        Convenience method that wraps generate() for single inputs. See generate()
        for the full argument semantics (max_tokens, speaker_embedding, etc.).

        Returns:
            Dict with audio_tokens, eos_frame, audio, sample_rate
        """
        if sampling_params is None:
            sampling_params = TTSSamplingParams()

        results = self.generate(
            [prompt],
            sampling_params,
            decode_audio=decode_audio,
            max_tokens=max_tokens,
            language=language,
            text_normalization=text_normalization,
            speaker_embedding=speaker_embedding,
            accent=accent,
            accent_strength=accent_strength,
        )
        return results[0]

    def save_audio(
        self,
        audio_bytes: bytes,
        path: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """Save PCM audio bytes to a WAV file.

        Args:
            audio_bytes: PCM audio data (float32)
            path: Output file path
            sample_rate: Audio sample rate
        """
        import wave

        import numpy as np

        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        # Convert to int16 for WAV
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)  # 16-bit
            f.setframerate(sample_rate)
            f.writeframes(audio_int16.tobytes())

# CONDITION_ROWS_RUNTIME_20261016
# EMBEDDING_CFG_20261016

# LLM_AUDIO_PREFILL_20261017
