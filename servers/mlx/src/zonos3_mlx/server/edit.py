"""Speech editing: /v2/edit/transcribe, /v2/edit/align and /v2/edit (a port of the sgl routes, servers/ggml edit.cpp).

Qwen3-ASR transcribes clips (the server's ``zonos3.tts.edit_align.AsrWorker`` in .venv-asr) and every word timing
comes from the server's FuseAlign worker (``zonos3.tts.edit_align.FuseAlignWorker``), both shared with dubbing; words
are normalized to edit tokens by ``zonos3.tts.textmatch``. The original clip is encoded with the MLX reference
encoder, every region is one span-end-terminated job on the scheduler over the training edit layout
(``plan_request(edit=...)``), and the spliced codes are decoded once.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import random
import tempfile
from dataclasses import replace

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, HTTPException
from zonos3.reference.synth import (
    Sampling,
    decode_base64_blob,
    normalize_language,
    validate_phoneme_groups,
)
from zonos3.server.schemas import EditAlignRequest, EditRequest, EditTranscribeRequest
from zonos3.tts import edit_align, textmatch
from zonos3.tts.directions import RAW_SPACE
from zonos3.tts.edit import extract_fill, splice_edit, validate_edit_spans
from zonos3.tts.edit_align import (
    GROUP_OPEN,
    WorkerError,
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
from zonos3.tts.steering import check_controls, resolve_controls

from zonos3_mlx.audio_io import decode_pcm
from zonos3_mlx.engine import EditLayout, ResolvedSpeaker, plan_request
from zonos3_mlx.server import capabilities as caps
from zonos3_mlx.server import speakers
from zonos3_mlx.server.generate import JobRun
from zonos3_mlx.server.state import ServerState

EDIT_DECODE_MAX_SECONDS = 600  # uploads are decoded this far, so the length check sees them
SAMPLE_RATE = 44100


def require_editing(state: ServerState) -> None:
    if not caps.model_supports_editing(state):
        raise HTTPException(status_code=400, detail="The loaded checkpoint does not support speech editing (needs "
                                                    "span_end_id and marker rows).")


def write_edit_audio(audio: bytes, directory: str, max_seconds: float) -> tuple[str, float, np.ndarray]:
    """Decode an upload (44.1 kHz mono), check its length and store it as WAV for the ASR and FuseAlign workers."""
    pcm = decode_pcm(audio, max_seconds=EDIT_DECODE_MAX_SECONDS)
    duration = len(pcm) / SAMPLE_RATE
    if duration > max_seconds:
        raise ValueError(f"Audio is {duration:.1f} s long; at most {max_seconds:.0f} s are supported here.")
    path = os.path.join(directory, "original.wav")
    sf.write(path, pcm, SAMPLE_RATE, subtype="FLOAT")
    return path, duration, pcm


def public_words(words) -> list[dict]:
    return [{"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence, "tokens": w.tokens}
            for w in words]


# ------------------------------------------------------------------------------------------------- workers


async def worker(fn, *args):
    """A blocking ASR / FuseAlign call of zonos3.tts.edit_align in a thread: HTTP 503 when a worker is unavailable
    (``WorkerError``), 400 when the input is at fault (``ValueError``); the messages are the workers' clients'."""
    try:
        return await asyncio.to_thread(fn, *args)
    except WorkerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def edit_language(state: ServerState, language: str) -> tuple[str, str]:
    """``(Whisper-style code, Qwen3-ASR name)`` of a request's ``language`` (edit_align.edit_language over the ASR
    worker's languages; HTTP 400 for one editing does not serve)."""
    try:
        normalized = normalize_language(language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    supported = await worker(state.asr.languages)
    try:
        return edit_align.edit_language(normalized, supported)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def transcribe(state: ServerState, audio_path: str, language: tuple[str, str]):
    """Qwen3-ASR transcript of a clip in the edit's language (edit_align.transcribe_audio: chunk-wise past 30 s)."""
    return await worker(edit_align.transcribe_audio, state.asr, state.aligner, audio_path, language[1])


async def edit_aligned_words(state: ServerState, audio_path: str, text: str, language: tuple[str, str],
                             utterances=None):
    """The words of ``text`` in the clip with their edit tokens (edit_align.edit_aligned_words; ``utterances``: the
    long-form anchors of a transcript of the same clip)."""
    return await worker(edit_align.edit_aligned_words, state.asr, state.aligner, audio_path, text, language, utterances)


# ----------------------------------------------------------------------------------------------------- edit


def uses_speaker_row(state: ServerState, req: EditRequest) -> bool:
    return state.config.speaker_embedding_dim > 0 if req.speaker_embedding is None else bool(req.speaker_embedding)


def region_doses(state: ServerState, req: EditRequest) -> list[dict[str, float]] | None:
    """Steering doses of each edited region: ``controls`` plus ``region_controls[k]`` summed per id (zero doses
    dropped), or None without controls (sgl _edit_region_doses)."""
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
    if not any(doses):
        return None
    bundle = state.engine.bundle
    if bundle is None:
        raise ValueError("This server has no steering bundle (place steer.gguf next to the backbone or pass --steer); "
                         "`controls` are unavailable.")
    for region in doses:
        check_controls(bundle, region)
    raw = sorted({did for region in doses for did in region if bundle.directions[did].space == RAW_SPACE})
    if raw and not uses_speaker_row(state, req):
        raise ValueError(f"Controls {', '.join(raw)} move the speaker embedding and need the speaker row "
                         "(speaker_embedding true or unset, on a speaker-embedding checkpoint).")
    return doses


async def generate_fill(state: ServerState, request: dict, layout: EditLayout, embedding, delta, min_frames: int):
    """One region's fill: (codec frames [n, K], whether it ended on span-end)."""
    c = state.config
    speaker = ResolvedSpeaker("embedding", embedding, None)
    try:
        plan = await asyncio.to_thread(plan_request, state.engine, request, speaker, edit=layout, hidden_delta=delta)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    sampling = replace(Sampling.from_request(plan.request), min_frames=int(min_frames))
    run = JobRun(state, plan, None, sampling, stop_id=c.span_end_id).start()
    try:
        await run.wait()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        run.abort()
    # the finishing row is not part of the generated span (it only completes the delayed codebooks' flush)
    rows = (torch.stack([torch.as_tensor(r) for r in run.rows[:-1]])[:, : c.n_codebooks] if len(run.rows) > 1
            else torch.zeros((0, c.n_codebooks), dtype=torch.int32))
    return extract_fill(rows, run.eos_frame, audio_pad_id=c.audio_pad_id, codebook_sizes=c.codebook_sizes)


async def handle_edit(state: ServerState, req: EditRequest) -> dict:
    c = state.config
    require_editing(state)
    if req.tag_cfg_scale != 1.0:
        raise HTTPException(status_code=400, detail="tag_cfg_scale is not supported by /v2/edit (speech-edit requests "
                                                    "run without tag CFG); use 1.0.")
    max_spans = caps.edit_max_spans(state)
    n_cb = c.n_codebooks
    frame_rate = caps.frame_rate(state)
    # Nonverbal markup is text for the model only: markers (canonical, no inner whitespace) ride in the prompt text and
    # never reach the ASR or FuseAlign (word diff, alignment, verify; target_tokens).
    target_text = canonical_markup(req.target_text.strip())
    plain_target_text = strip_nonverbal_markup(target_text)
    edit_lang = await edit_language(state, req.language)  # every ASR and FuseAlign call of the edit
    with tempfile.TemporaryDirectory(prefix="zonos3-edit-") as tmp:
        try:
            if not target_text:
                raise ValueError("target_text is empty.")
            validate_phoneme_groups(c.text_tokenizer_spec or None, target_text)
            if req.original_words is not None and not (req.original_text or "").strip():
                raise ValueError("original_words needs the original_text they spell.")
            doses = region_doses(state, req)
            language = normalize_language(req.language)
            use_row = uses_speaker_row(state, req)
            if use_row and c.speaker_embedding_dim <= 0:
                raise ValueError("speaker_embedding needs a checkpoint with a speaker-embedding row.")
            audio = decode_base64_blob(req.audio_base64, "audio_base64")
            original_path, duration, _ = write_edit_audio(audio, tmp, caps.EDIT_MAX_SECONDS)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            speaker_embedding = await speakers.embed_audio(state, audio) if use_row else None
            codes = (await speakers.encode_reference(state, audio, max_seconds=caps.EDIT_MAX_SECONDS)).to(torch.int64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Longer clips are windowed and every fill gets a minimum length; both read word timings of the original (the
        # caller's original_words, else a FuseAlign alignment) and the target words' edit tokens. Without original_text,
        # a short clip with explicit spans is not transcribed: its regions' own lengths are expected.
        windowed = len(codes) / frame_rate > req.max_context_seconds
        target_words = target_text.split()
        # [target tokens, tokens of each target word]; a pronunciation group is one target token (target_tokens).
        normalized_target = target_tokens(target_text)
        original_text, words = req.original_text, None
        if original_text:
            original_text = strip_nonverbal_markup(original_text)  # what the workers read (and the response reports)
        if req.original_words is not None:
            given = [w.model_dump() for w in req.original_words]
            for w in given:
                if w["tokens"] is None:
                    w["tokens"] = textmatch.tokens(w["word"])
            try:
                words = edit_words(given, [w["tokens"] for w in given], textmatch.tokens(original_text),
                                   duration=duration)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        elif req.spans is None or windowed or (req.min_fill_ratio > 0 and original_text):
            utterances = None
            if not original_text:
                transcript = await transcribe(state, original_path, edit_lang)
                original_text, utterances = transcript.text, transcript.utterances
            words = await edit_aligned_words(state, original_path, original_text, edit_lang, utterances)
        # verify with pronunciation groups: the reference's tokens, each group a wildcard word
        verify_reference = normalized_target[0] if req.verify and GROUP_OPEN in target_text else None
        try:
            if req.spans is not None:
                spans = seconds_to_spans(req.spans, len(codes), frame_rate)
            else:
                spans = diff_spans(words, normalized_target[0], total_frames=len(codes), frame_rate=frame_rate,
                                   margin_seconds=req.margin_seconds, max_spans=max_spans).spans
            validate_edit_spans(spans, len(codes), max_spans)
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
            # Per region: (speaker-row embedding, hidden steering delta). Without explicit spans the request's
            # controls apply to every diffed region alike.
            steering = [(speaker_embedding, None)] * len(spans)
            if doses is not None:
                steering = []
                for k in range(len(spans)):
                    resolved = resolve_controls(state.engine.bundle, doses[k if req.spans is not None else 0], None,
                                                text=target_text, speaker_embedding=speaker_embedding,
                                                sources=state.engine.steering_sources(), hidden_size=c.dim)
                    steering.append((resolved.speaker_embedding, resolved.hidden_delta))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        base_seed = req.seed if req.seed is not None else random.randrange(2**31 - 8)
        gain = np.float32(state.engine.output_gain(None))

        async def take(k: int) -> dict:
            fills, stopped = [None] * len(spans), [None] * len(spans)
            for window in windows:
                window_codes = codes[window.start: window.end]
                window_spans = [[int(start), int(length)] for start, length in window.spans(spans)]
                window_fills: list[torch.Tensor] = []
                for region in window.regions:
                    embedding, delta = steering[region]
                    request = {
                        "text": window.text, "language": language, "text_normalization": req.text_normalization,
                        "temperature": req.temperature, "topk": req.topk, "top_p": req.top_p, "min_p": req.min_p,
                        "repetition_window": req.repetition_window, "repetition_penalty": req.repetition_penalty,
                        "repetition_codebooks": req.repetition_codebooks, "seed": base_seed + k, "max_tokens": max_tokens,
                        "condition_buckets": req.condition_buckets, "condition_values": req.condition_values, "stream": False}
                    layout = EditLayout(window_codes, window_spans, list(window_fills), min_frames[region])
                    fill, ended = await generate_fill(state, request, layout, embedding, delta, min_frames[region])
                    window_fills.append(fill.to(torch.int32))
                    fills[region], stopped[region] = window_fills[-1], ended
            spliced = splice_edit(codes, spans, fills).numpy()
            edited = (await state.run(state.engine.decoder.decode, spliced)) * gain
            verify, take_words = None, None
            if req.verify:
                edited_path = os.path.join(tmp, f"take{k}.wav")
                sf.write(edited_path, edited, SAMPLE_RATE, subtype="FLOAT")
                # The take's transcript (Qwen3-ASR) and its anchors for the alignment of the take's words.
                heard = await transcribe(state, edited_path, edit_lang)
                if verify_reference is None:
                    scored = textmatch.verify(plain_target_text, heard.text)
                    verify = {"asr_text": heard.text, "wer": scored["word_errors"] / max(1, scored["ref_words"]),
                              "cer": scored["cer"]}
                    take_words = public_words(await edit_aligned_words(state, edited_path, plain_target_text,
                                                                       edit_lang, heard.utterances))
                else:
                    # Groups are wildcard words: any one recognized word matches; the take's words come from aligning
                    # the target with each group replaced by the word heard.
                    scored = wildcard_verify(verify_reference, textmatch.tokens(heard.text))
                    verify = {"asr_text": heard.text, "wer": scored["word_errors"] / max(1, scored["ref_words"]),
                              "cer": scored["cer"], "groups": scored["groups"]}
                    heard_text = fill_groups(plain_target_text, [g["heard"] for g in scored["groups"]])
                    if heard_text:
                        take_words = public_words(await edit_aligned_words(state, edited_path, heard_text,
                                                                           edit_lang, heard.utterances))
            wav = io.BytesIO()
            sf.write(wav, edited, SAMPLE_RATE, subtype="PCM_16", format="WAV")
            edited_regions, shift = [], 0
            for (start, length), fill in zip(spans, fills, strict=True):
                edited_regions.append([(start + shift) / frame_rate, (start + shift + len(fill)) / frame_rate])
                shift += len(fill) - length
            return {"seed": base_seed + k, "audio_base64": base64.b64encode(wav.getvalue()).decode("ascii"),
                    "duration": len(edited) / SAMPLE_RATE, "edited_regions": edited_regions,
                    "fill_seconds": [len(fill) / frame_rate for fill in fills], "stopped": stopped, "verify": verify,
                    "words": take_words}

        takes = await asyncio.gather(*(take(k) for k in range(req.num_takes)))
    return {
        "sample_rate": SAMPLE_RATE, "original_text": original_text, "target_text": target_text,
        "words": None if words is None else public_words(words),
        "spans": [[start / frame_rate, (start + length) / frame_rate] for start, length in spans],
        "contexts": [[w.start / frame_rate, w.end / frame_rate] for w in windows] if windowed else None,
        "expected_seconds": expected, "takes": list(takes),
    }


def router(state: ServerState) -> APIRouter:
    r = APIRouter()

    @r.post("/v2/edit/transcribe")
    async def transcribe_route(req: EditTranscribeRequest):
        """Qwen3-ASR transcript of the audio to edit (chunk-wise past 30 s) and its words' FuseAlign timings."""
        require_editing(state)
        with tempfile.TemporaryDirectory(prefix="zonos3-edit-") as tmp:
            try:
                path, duration, _ = write_edit_audio(decode_base64_blob(req.audio_base64, "audio_base64"), tmp,
                                                     caps.EDIT_ALIGN_MAX_SECONDS)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            language = await edit_language(state, req.language)
            transcript = await transcribe(state, path, language)
            words = await edit_aligned_words(state, path, transcript.text, language, transcript.utterances)
        return {"text": transcript.text, "duration": duration, "words": public_words(words)}

    @r.post("/v2/edit/align")
    async def align(req: EditAlignRequest):
        """FuseAlign forced alignment of the audio to ``text``: word timings, presences (``confidence``) and edit
        tokens."""
        require_editing(state)
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is empty.")
        with tempfile.TemporaryDirectory(prefix="zonos3-edit-") as tmp:
            try:
                path, duration, _ = write_edit_audio(decode_base64_blob(req.audio_base64, "audio_base64"), tmp,
                                                     caps.EDIT_ALIGN_MAX_SECONDS)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            words = await edit_aligned_words(state, path, text, await edit_language(state, req.language))
        return {"text": text, "duration": duration, "words": public_words(words)}

    @r.post("/v2/edit")
    async def edit(req: EditRequest):
        """Regenerate the edited regions of recorded speech; unedited codec frames are kept exactly."""
        return await handle_edit(state, req)

    return r
