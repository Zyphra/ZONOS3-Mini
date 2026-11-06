"""Speech editing in every server (``/v2/edit/transcribe``, ``/v2/edit/align``, ``/v2/edit``): ``EditRuntime``.

The runtime holds everything of an edit that is not the model: request checks, the ASR transcript and FuseAlign word
timings (``zonos3.tts.edit_align``: the server's ``AsrWorker`` / ``FuseAlignWorker``, shared with plugins), the word
diff, the context windows of a long clip, the fill lengths, the splice and ``verify``. What needs the served model goes
through the server's ``EditEngine``: decoding an upload, the speaker row and reference codes of the clip, the
engine-level request checks, one region's span-end-terminated generation and the decode of the spliced codes. The
CUDA and MLX servers implement the engine in-process; the ggml server runs this runtime in
zonos3/workers/pyhost.py and answers the engine's calls over its pipe.

Speech recognition is optional: alignment and edits of a clip whose transcript the caller sends need FuseAlign only.
Without it (``edit_align.asr_available``), /v2/edit/transcribe, ``verify`` and an edit that needs the words of a clip
sent without ``original_text`` fail with ``edit_align.AsrUnavailable`` (503) before any encode or generation.

Errors: ``ValueError`` is the request's fault (HTTP 400), ``zonos3.tts.edit_align.WorkerError`` an unavailable ASR /
FuseAlign worker (503), ``RuntimeError`` a failed generation or decode (500); ``zonos3.server.errors.status_of`` maps
them.
"""

from __future__ import annotations

import asyncio
import base64
import os
import random
import tempfile
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from zonos3 import languages
from zonos3.generation import EDIT_MAX_SECONDS, SAMPLE_RATE
from zonos3.tts import edit_align, textmatch
from zonos3.tts.audio import pcm16_wav_bytes
from zonos3.tts.edit import extract_fill, splice_edit, validate_edit_spans
from zonos3.tts.edit_align import (
    GROUP_OPEN,
    canonical_markup,
    diff_spans,
    edit_words,
    expected_fill_seconds,
    fill_groups,
    plan_edit_windows,
    seconds_to_spans,
    target_tokens,
    whole_clip_window,
    wildcard_verify,
)
from zonos3.tts.markup import strip_nonverbal_markup


@dataclass(frozen=True)
class EditModel:
    """What an edit needs to know of the served checkpoint."""

    frame_rate: float  # codec frames per second
    n_codebooks: int
    audio_pad_id: int
    codebook_sizes: tuple[int, ...]
    max_spans: int  # edit marker rows; 0 when the checkpoint cannot edit (no span_end_id or no marker rows)
    speaker_row: bool  # the checkpoint has a speaker-embedding row


@dataclass
class FillJob:
    """One region's generation: the edit prompt of ``window_codes`` with ``spans`` masked and ``fills`` already
    generated, ended by span-end (masked until ``min_frames`` rows), as a /v2/generate request would sample it."""

    text: str  # the window's target text (normalized by the engine when ``text_normalization``)
    language: str  # normalized server language code
    text_normalization: bool
    sampling: dict  # temperature, topk, top_p, min_p, repetition_window, repetition_penalty, repetition_codebooks
    seed: int
    max_tokens: int  # the engine clamps it to its own limit
    min_frames: int
    condition_buckets: Any
    condition_values: Any
    speaker_embedding: Any  # the engine's embedding of the clip (``EditEngine.embed``), or None: no speaker row
    controls: dict[str, float] | None  # the region's steering doses (checked by ``EditEngine.check``)
    controls_text: str  # the text the doses' character spans refer to (the whole target text)
    window_codes: np.ndarray  # [frames, n_codebooks] int64
    spans: list[list[int]]  # window-relative [start_frame, length_frames]
    fills: list[np.ndarray] = field(default_factory=list)  # [n, n_codebooks] int32 each, earlier regions of the window


class EditEngine(Protocol):
    """The served model's part of an edit (see the module docstring). Every method raises ValueError for input at
    fault and RuntimeError for a failure of the engine."""

    model: EditModel

    async def decode_audio(self, audio: bytes) -> tuple[np.ndarray, int]:
        """An upload (any sound file) -> (mono float32 PCM, its sample rate)."""

    async def embed(self, audio: bytes) -> Any:
        """The speaker-row embedding of an upload (opaque to the runtime; handed back in ``FillJob``)."""

    async def encode(self, audio: bytes, max_seconds: float) -> np.ndarray:
        """Ordinary codec codes ``[frames, n_codebooks]`` (int64) of an upload."""

    async def check(self, target_text: str, doses: list[dict[str, float]] | None, condition_buckets,
                    condition_values) -> set[str]:
        """Engine-level checks before any work: the target's pronunciation groups, every region's steering doses
        against the server's bundle (ValueError without one) and the condition buckets / values. Returns the dosed
        ids that move the speaker embedding (raw-space directions)."""

    async def fill(self, job: FillJob) -> tuple[np.ndarray, int]:
        """The generated delay-stacked rows ``[n, n_codebooks]`` without the finishing row, and the aligned EOS frame
        (-1 = none), of one region."""

    async def decode(self, codes: np.ndarray) -> np.ndarray:
        """Float32 44.1 kHz audio (with the server's output gain) of ordinary codes ``[frames, n_codebooks]``."""


def edit_capabilities(model: EditModel, available: bool, transcribe: bool) -> dict:
    """The speech-editing keys of /v2/capabilities. ``available``: the server can run the FuseAlign worker and encode
    reference audio (speech editing); ``transcribe``: it also has speech recognition (the ASR worker installed), for
    /v2/edit/transcribe, transcripts inside /v2/edit and ``verify``."""
    editing = model.max_spans > 0
    speech_editing = editing and available
    return {"speech_editing": speech_editing, "edit_transcribe": speech_editing and transcribe,
            "edit_max_spans": model.max_spans if editing else 0,
            "edit_frame_rate": model.frame_rate if editing else None, "edit_max_seconds": EDIT_MAX_SECONDS,
            "edit_align_max_seconds": EDIT_MAX_SECONDS}


def public_words(words) -> list[dict]:
    return [{"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence, "tokens": w.tokens}
            for w in words]


def region_doses(req, model: EditModel) -> list[dict[str, float]] | None:
    """Steering doses of each edited region: ``controls`` plus ``region_controls[k]`` summed per id (zero doses
    dropped), or None without controls."""
    if req.region_controls is not None:
        if req.spans is None:
            raise ValueError("region_controls needs explicit spans (one entry per span).")
        if len(req.region_controls) != len(req.spans):
            raise ValueError(f"region_controls has {len(req.region_controls)} entries for {len(req.spans)} spans; "
                             "send one per span (null for none).")
    regions = len(req.spans) if req.spans is not None else 1
    doses = []
    for k in range(regions):
        merged = {did: float(v) for did, v in (req.controls or {}).items()}
        extra = req.region_controls[k] if req.region_controls is not None else None
        for did, v in (extra or {}).items():
            merged[did] = merged.get(did, 0.0) + float(v)
        doses.append({did: v for did, v in merged.items() if v != 0.0})
    return doses if any(doses) else None


# The fixes of AsrUnavailable (what a caller of a server without speech recognition sends instead)
SEND_ORIGINAL_TEXT = "send the clip's transcript as original_text"
SEND_TO_ALIGN = "send the transcript to /v2/edit/align instead"
VERIFY_FALSE = "set verify to false"


class EditRuntime:
    def __init__(self, engine: EditEngine, asr: edit_align.AsrWorker | None, aligner: edit_align.FuseAlignWorker):
        self.engine = engine
        # the server's ASR and FuseAlign workers (shared with plugins; the server closes them); ``asr`` may be None or
        # not installed: no speech recognition (edit_align.asr_available)
        self.asr = asr
        self.aligner = aligner

    @property
    def model(self) -> EditModel:
        return self.engine.model

    @property
    def transcribes(self) -> bool:
        """Whether the server has speech recognition (/v2/capabilities ``edit_transcribe``)."""
        return edit_align.asr_available(self.asr)

    def require_editing(self) -> None:
        if self.model.max_spans <= 0:
            raise ValueError("The loaded checkpoint does not support speech editing (needs span_end_id and marker "
                             "rows).")

    # ---- workers ----------------------------------------------------------------------------------------------

    def language(self, language: str) -> tuple[str, str]:
        """``(Whisper-style code, Qwen3-ASR name)`` of a request's ``language`` (edit_align.edit_language over
        ``EDIT_LANGUAGES``; ValueError for one editing does not serve). No worker is consulted."""
        return edit_align.edit_language(languages.normalize(language))

    async def require_recognition(self, language: str, fix: str) -> None:
        """Check that a request's speech recognition can run: ``AsrUnavailable(fix)`` without the ASR worker,
        ValueError for a ``language`` (a request's) the worker does not transcribe."""
        if not self.transcribes:
            raise edit_align.AsrUnavailable(fix)
        edit_align.edit_language(languages.normalize(language), await asyncio.to_thread(self.asr.languages))

    async def _transcribe(self, path: str, language: tuple[str, str]) -> edit_align.Transcript:
        """Qwen3-ASR transcript of a clip in the edit's language (chunk-wise past 30 s). The routes check
        ``require_recognition`` first; the check here only backs up an edit whose clip turned out windowed after its
        encode."""
        if not self.transcribes:
            raise edit_align.AsrUnavailable(SEND_ORIGINAL_TEXT)
        return await asyncio.to_thread(edit_align.transcribe_audio, self.asr, self.aligner, path, language[1])

    async def _aligned_words(self, path: str, text: str, language: tuple[str, str], utterances=None):
        """The words of ``text`` in the clip with their edit tokens (``utterances``: the long-form anchors of a
        transcript of the same clip; without speech recognition a long clip aligns without them)."""
        asr = self.asr if self.transcribes else None
        return await asyncio.to_thread(edit_align.edit_aligned_words, asr, self.aligner, path, text, language,
                                       utterances)

    async def _write_audio(self, audio: bytes, directory: str, max_seconds: float) -> tuple[str, float]:
        """Decode an upload, check its length and store it as WAV for the ASR and FuseAlign workers."""
        import soundfile as sf

        pcm, sample_rate = await self.engine.decode_audio(audio)
        duration = len(pcm) / sample_rate
        if duration > max_seconds:
            raise ValueError(f"Audio is {duration:.1f} s long; at most {max_seconds:.0f} s are supported here.")
        path = os.path.join(directory, "original.wav")
        sf.write(path, pcm, sample_rate, subtype="FLOAT")
        return path, duration

    # ---- routes -----------------------------------------------------------------------------------------------

    async def transcribe(self, req) -> dict:
        """/v2/edit/transcribe: the Qwen3-ASR transcript (chunk-wise past 30 s) and its words' FuseAlign timings."""
        from zonos3.reference.synth import decode_base64_blob

        self.require_editing()
        language = self.language(req.language)
        await self.require_recognition(req.language, SEND_TO_ALIGN)
        with tempfile.TemporaryDirectory(prefix="zonos3-edit-") as tmp:
            path, duration = await self._write_audio(decode_base64_blob(req.audio_base64, "audio_base64"), tmp,
                                                     EDIT_MAX_SECONDS)
            transcript = await self._transcribe(path, language)
            words = await self._aligned_words(path, transcript.text, language, transcript.utterances)
        return {"text": transcript.text, "duration": duration, "words": public_words(words)}

    async def align(self, req) -> dict:
        """/v2/edit/align: FuseAlign word timings, presences (``confidence``) and edit tokens of ``text``."""
        from zonos3.reference.synth import decode_base64_blob

        self.require_editing()
        text = req.text.strip()
        if not text:
            raise ValueError("text is empty.")
        language = self.language(req.language)
        with tempfile.TemporaryDirectory(prefix="zonos3-edit-") as tmp:
            path, duration = await self._write_audio(decode_base64_blob(req.audio_base64, "audio_base64"), tmp,
                                                     EDIT_MAX_SECONDS)
            words = await self._aligned_words(path, text, language)
        return {"text": text, "duration": duration, "words": public_words(words)}

    async def edit(self, req) -> dict:
        """/v2/edit: regenerate the edited regions of recorded speech; unedited codec frames are kept exactly.

        Each region is one span-end-terminated generation over [speaker row][condition rows][text][unmasked
        context][earlier fills][marker]. A clip longer than ``max_context_seconds`` is cut into word-snapped windows
        around the regions (``plan_edit_windows``): each window's prompts hold its codes and its stretch of the target
        text, and every fill splices into the full clip, which is decoded once. Region k's steering (``controls`` +
        ``region_controls[k]``) moves its speaker row and adds its hidden delta on the frames it generates."""
        from zonos3.reference.synth import decode_base64_blob

        m = self.model
        self.require_editing()
        if req.tag_cfg_scale != 1.0:
            raise ValueError("tag_cfg_scale is not supported by /v2/edit (speech-edit requests run without tag CFG); "
                             "use 1.0.")
        frame_rate, n_cb = m.frame_rate, m.n_codebooks
        # Nonverbal markup is text for the model only: markers (canonical, no inner whitespace) ride in the prompt
        # text and never reach the ASR or FuseAlign (word diff, alignment, verify; target_tokens).
        target_text = canonical_markup(req.target_text.strip())
        plain_target_text = strip_nonverbal_markup(target_text)
        edit_lang = self.language(req.language)  # every ASR and FuseAlign call of the edit
        if not target_text:
            raise ValueError("target_text is empty.")
        if req.original_words is not None and not (req.original_text or "").strip():
            raise ValueError("original_words needs the original_text they spell.")
        # Without original_text (or original_words) the clip's words, when read (no explicit spans, or a clip to
        # window), come from a transcript: refused here without speech recognition, before any model work.
        untranscribed = req.original_words is None and not req.original_text
        if req.verify:
            await self.require_recognition(req.language, VERIFY_FALSE)
        if untranscribed and req.spans is None:
            await self.require_recognition(req.language, SEND_ORIGINAL_TEXT)
        doses = region_doses(req, m)
        language = languages.normalize(req.language)
        use_row = m.speaker_row if req.speaker_embedding is None else bool(req.speaker_embedding)
        if use_row and not m.speaker_row:
            raise ValueError("speaker_embedding needs a checkpoint with a speaker-embedding row.")
        raw = await self.engine.check(target_text, doses, req.condition_buckets, req.condition_values)
        if raw and not use_row:
            raise ValueError(f"Controls {', '.join(sorted(raw))} move the speaker embedding and need the speaker row "
                             "(speaker_embedding true or unset, on a speaker-embedding checkpoint).")
        audio = decode_base64_blob(req.audio_base64, "audio_base64")
        with tempfile.TemporaryDirectory(prefix="zonos3-edit-") as tmp:
            original_path, duration = await self._write_audio(audio, tmp, EDIT_MAX_SECONDS)
            if untranscribed and req.spans is not None and duration > req.max_context_seconds:
                await self.require_recognition(req.language, SEND_ORIGINAL_TEXT)
            speaker_embedding = await self.engine.embed(audio) if use_row else None
            codes = np.asarray(await self.engine.encode(audio, EDIT_MAX_SECONDS), dtype=np.int64)
            # Longer clips are windowed and every fill gets a minimum length; both read word timings of the original
            # (the caller's original_words, else a FuseAlign alignment) and the target words' edit tokens. Without
            # original_text, a short clip with explicit spans is not transcribed: its regions' own lengths are
            # expected.
            windowed = len(codes) / frame_rate > req.max_context_seconds
            target_words = target_text.split()
            # [target tokens, tokens of each target word]; a pronunciation group is one target token.
            normalized_target = target_tokens(target_text)
            original_text, words = req.original_text, None
            if original_text:
                original_text = strip_nonverbal_markup(original_text)  # what the workers read (and the response)
            if req.original_words is not None:
                given = [w.model_dump() for w in req.original_words]
                for w in given:
                    if w["tokens"] is None:
                        w["tokens"] = textmatch.tokens(w["word"])
                words = edit_words(given, [w["tokens"] for w in given], textmatch.tokens(original_text),
                                   duration=duration)
            elif req.spans is None or windowed or (req.min_fill_ratio > 0 and original_text):
                utterances = None
                if not original_text:
                    transcript = await self._transcribe(original_path, edit_lang)
                    original_text, utterances = transcript.text, transcript.utterances
                words = await self._aligned_words(original_path, original_text, edit_lang, utterances)
            # verify with pronunciation groups: the reference's tokens, each group a wildcard word
            verify_reference = normalized_target[0] if req.verify and GROUP_OPEN in target_text else None
            if req.spans is not None:
                spans = seconds_to_spans(req.spans, len(codes), frame_rate)
            else:
                spans = diff_spans(words, normalized_target[0], total_frames=len(codes), frame_rate=frame_rate,
                                   margin_seconds=req.margin_seconds, max_spans=m.max_spans).spans
            validate_edit_spans(spans, len(codes), m.max_spans)
            windows = [whole_clip_window(spans, len(codes), target_text)]
            if windowed:
                windows = plan_edit_windows(spans, words, target_words, normalized_target[1:],
                                            total_frames=len(codes), frame_rate=frame_rate,
                                            context_seconds=req.context_seconds, target_text=target_text)
            if words is not None:
                expected = expected_fill_seconds(spans, words, target_words, normalized_target[1:],
                                                 frame_rate=frame_rate)
            else:
                expected = [length / frame_rate for _, length in spans]
            # The span-end token is masked until a fill reaches min_fill_ratio of its expected length.
            min_frames = [min(int(req.min_fill_ratio * seconds * frame_rate), int(req.max_span_seconds * frame_rate))
                          for seconds in expected]
            max_tokens = int(req.max_span_seconds * frame_rate) + n_cb + 1
            # Without explicit spans the request's controls apply to every diffed region alike.
            region_controls = [None if doses is None else doses[k if req.spans is not None else 0]
                               for k in range(len(spans))]
            base_seed = req.seed if req.seed is not None else random.randrange(2**31 - 8)
            sampling = {"temperature": req.temperature, "topk": req.topk, "top_p": req.top_p, "min_p": req.min_p,
                        "repetition_window": req.repetition_window, "repetition_penalty": req.repetition_penalty,
                        "repetition_codebooks": req.repetition_codebooks}

            async def take(k: int) -> dict:
                fills, stopped = [None] * len(spans), [None] * len(spans)
                for window in windows:
                    window_codes = codes[window.start: window.end]
                    window_spans = [[int(start), int(length)] for start, length in window.spans(spans)]
                    window_fills: list[np.ndarray] = []
                    for region in window.regions:
                        rows, eos_frame = await self.engine.fill(FillJob(
                            text=window.text, language=language, text_normalization=req.text_normalization,
                            sampling=sampling, seed=base_seed + k, max_tokens=max_tokens,
                            min_frames=min_frames[region], condition_buckets=req.condition_buckets,
                            condition_values=req.condition_values, speaker_embedding=speaker_embedding,
                            controls=region_controls[region], controls_text=target_text, window_codes=window_codes,
                            spans=window_spans, fills=list(window_fills)))
                        fill, ended = extract_fill(rows, None if eos_frame is None or eos_frame < 0 else eos_frame,
                                                   audio_pad_id=m.audio_pad_id, codebook_sizes=list(m.codebook_sizes))
                        window_fills.append(np.asarray(fill, dtype=np.int32))
                        fills[region], stopped[region] = window_fills[-1], ended
                spliced = np.asarray(splice_edit(codes, spans, fills), dtype=np.int32)
                edited = np.asarray(await self.engine.decode(spliced), dtype=np.float32)
                verify, take_words = None, None
                if req.verify:
                    import soundfile as sf

                    edited_path = os.path.join(tmp, f"take{k}.wav")
                    sf.write(edited_path, edited, SAMPLE_RATE, subtype="FLOAT")
                    # The take's transcript (Qwen3-ASR) and its anchors for the alignment of the take's words.
                    heard = await self._transcribe(edited_path, edit_lang)
                    if verify_reference is None:
                        scored = textmatch.verify(plain_target_text, heard.text)
                        verify = {"asr_text": heard.text, "wer": scored["word_errors"] / max(1, scored["ref_words"]),
                                  "cer": scored["cer"]}
                        take_words = public_words(await self._aligned_words(edited_path, plain_target_text,
                                                                            edit_lang, heard.utterances))
                    else:
                        # Groups are wildcard words: any one recognized word matches; the take's words come from
                        # aligning the target with each group replaced by the word heard.
                        scored = wildcard_verify(verify_reference, textmatch.tokens(heard.text))
                        verify = {"asr_text": heard.text, "wer": scored["word_errors"] / max(1, scored["ref_words"]),
                                  "cer": scored["cer"], "groups": scored["groups"]}
                        heard_text = fill_groups(plain_target_text, [g["heard"] for g in scored["groups"]])
                        if heard_text:
                            take_words = public_words(await self._aligned_words(edited_path, heard_text, edit_lang,
                                                                                heard.utterances))
                edited_regions, shift = [], 0
                for (start, length), fill in zip(spans, fills, strict=True):
                    edited_regions.append([(start + shift) / frame_rate, (start + shift + len(fill)) / frame_rate])
                    shift += len(fill) - length
                return {"seed": base_seed + k,
                        "audio_base64": base64.b64encode(pcm16_wav_bytes(edited, SAMPLE_RATE)).decode("ascii"),
                        "duration": len(edited) / SAMPLE_RATE, "edited_regions": edited_regions,
                        "fill_seconds": [len(fill) / frame_rate for fill in fills], "stopped": stopped,
                        "verify": verify, "words": take_words}

            takes = await asyncio.gather(*(take(k) for k in range(req.num_takes)))
        return {
            "sample_rate": SAMPLE_RATE, "original_text": original_text, "target_text": target_text,
            "words": None if words is None else public_words(words),
            "spans": [[start / frame_rate, (start + length) / frame_rate] for start, length in spans],
            # The audio the model saw per window (full-clip seconds); null when the clip was edited whole.
            "contexts": [[w.start / frame_rate, w.end / frame_rate] for w in windows] if windowed else None,
            # Each region's expected fill length (seconds); its fills last at least min_fill_ratio of it.
            "expected_seconds": expected, "takes": list(takes),
        }
