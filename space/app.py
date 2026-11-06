"""ZONOS3 Mini on a ZeroGPU Space: the Voice Studio (studio/dist) over the v2 HTTP API, served by the plain-PyTorch
reference (zonos3.reference) with generation on ZeroGPU.

``gr.Server`` is the FastAPI app; its ``launch()`` runs through ``gr.Blocks.launch``, where ``spaces`` starts ZeroGPU.
The routes are the part of api/openapi.yaml the studio and the lite page use when the server has no speech editing,
dubbing, project store or pronunciation lexicon (``/v2/capabilities`` says so and the studio adapts): GET
``/v2/capabilities``, ``/v2/controls``, ``/v2/speakers``, ``/v2/speakers/{id}/preview``,
``/v2/speakers/{id}/embedding``, POST ``/v2/speakers`` and ``/v2/generate`` (prefix-streamed float32 PCM, a 4-frame first
chunk). Speaker embeddings run on this process's CPU; only ``/v2/generate`` takes a GPU, in a held GPU session
(``gpu_session`` / ``GpuSessions``), on the visitor's ZeroGPU quota through the X-IP-Token the studio forwards from
huggingface.co (studio/src/lib/zerogpu.ts).
"""

import spaces  # noqa: I001  (ZeroGPU: must be imported before torch)

import subprocess
import sys

# qwen-asr pins transformers==4.57.6, which the Space's pins (4.57.3, ZONOS3's lock) exclude; it runs on 4.57.3, so it
# is installed without its dependencies (its runtime ones are in requirements.txt) before anything imports it.
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--quiet", "qwen-asr==0.0.6"], check=True)

import asyncio
import base64
import contextvars
import hashlib
import io
import multiprocessing
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from fastapi import Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fusealign_hub.aligner import CHECKPOINT_FILE as FUSEALIGN_CHECKPOINT
from fusealign_hub.aligner import HF_REPO as FUSEALIGN_REPO
from gradio.context import LocalContext
from qwen_asr.inference.utils import SUPPORTED_LANGUAGES
from zonos3 import languages
from zonos3.generation import (
    DECODE_CONTEXT_FRAMES,
    SAMPLE_RATE,
    TAG_CFG_MAX_SCALE,
    TAG_CFG_MIN_SCALE,
)
from zonos3.models.accent_format import load_adapter
from zonos3.reference.accents import AccentState
from zonos3.reference.lm import Zonos3LM
from zonos3.reference.synth import (
    EditLayout,
    ReferenceAssets,
    ReferenceContext,
    StreamSettings,
    _speaker_embedding,
    decode_base64_blob,
    edit_region,
    load_embedding_vector,
    output_gain,
    synthesize,
)
from zonos3.server.edit import call as edit_errors
from zonos3.server.schemas import (
    EditAlignRequest,
    EditRequest,
    EditTranscribeRequest,
    GenerateRequest,
    SpeakerCreateRequest,
)
from zonos3.tokenizer import matched_codec
from zonos3.tokenizer.dual_reference import encode_dual_reference
from zonos3.tokenizer.matched_codec import decode_model
from zonos3.tts import plan as P
from zonos3.tts.audio import _decode_audio_bytes, _transcode_audio_bytes_to_wav
from zonos3.tts.directions import RAW_SPACE
from zonos3.tts.edit import edit_marker_rows
from zonos3.tts.edit_runtime import EditModel, EditRuntime, FillJob, edit_capabilities
from zonos3.tts.markup import NONVERBAL_MARKUP_TAGS, trained_on_tags
from zonos3.tts.steering import (
    check_controls,
    controls_document,
    disabled_controls,
    resolve_controls,
)
from zonos3.utils.release import RELEASE_LM, resolve_dir_or_local, resolve_file
from zonos3.workers import asr as asr_worker
from zonos3.workers import fusealign as fusealign_worker

ROOT = Path(__file__).resolve().parent
STUDIO = ROOT / "studio"
# One device object everywhere: synthesize() moves the LM when lm.device != device, and ReferenceAssets caches the
# decoder by str(device). On ZeroGPU this process only emulates CUDA; the GPU exists inside @spaces.GPU calls.
DEVICE = torch.device("cuda:0")
CPU = torch.device("cpu")

# The edit checkpoint's vocabulary: its span-end audio row and edit-marker text rows, which plain takes never read or
# sample (so they run exactly as on the release LM). Edit regions add the edit LoRA per request (EDIT_LORA). Grown on
# the CPU, then moved: ZeroGPU's emulated CUDA refuses mixing its tensors with CPU ones in this process.
_EDIT = f"{RELEASE_LM}/edit_lora"
_GROWN = torch.load(resolve_file(f"{_EDIT}/grown_rows.pt"), map_location="cpu", weights_only=True)
LM = Zonos3LM.load(RELEASE_LM, device="cpu", dtype=torch.float32)
LM.grow(_GROWN["rows"], span_end_id=_GROWN["span_end_id"], text_extra_tokens=_GROWN["text_extra_tokens"])
LM = LM.to(DEVICE)
CONFIG = LM.config
ASSETS = ReferenceAssets.load(steer_bundle=str(ROOT / "steering" / "studio_bundle_mect60k.json"))
CONTEXT = ReferenceContext(LM, ASSETS, DEVICE)  # the request rules' view of the checkpoint (no GPU work)
EDIT_LORA = AccentState.for_model(LM, load_adapter(resolve_file(f"{_EDIT}/z3m_omni3b_edit_lora.pt"), tables=True), 1.0)
ASSETS.decoder(CONFIG.audio_codec, DEVICE)
ASSETS.speaker_encoder(CONFIG.speaker_embedding_codec, CONFIG.speaker_embedding_dim, CPU)
# A clip in reference mode is encoded by the dual codec inside the GPU call; fetch its teacher folder now.
_DUAL = next(e for e in matched_codec.registry().values() if e.get("family") == "dual")
resolve_dir_or_local(_DUAL["teacher"])
# All nine NeMo grammars, synchronously: GPU workers are forked from this process and normalize with them.
ASSETS.normalizer().warmup()


# --------------------------------------------------------------------------------------------- capabilities


EDIT_MODEL = EditModel(
    frame_rate=SAMPLE_RATE / int(matched_codec.entry(CONFIG.audio_codec)["hop"]), n_codebooks=CONFIG.n_codebooks,
    audio_pad_id=CONFIG.audio_pad_id, codebook_sizes=CONFIG.codebook_sizes,
    max_spans=edit_marker_rows(CONFIG.text_extra_tokens, CONTEXT.text_tokenizer_spec), speaker_row=True)


def capabilities() -> dict:
    """GET /v2/capabilities (the MLX server's document, for the reference and this Space's routes)."""
    c, modes = CONFIG, list(CONTEXT.speaker_modes)
    codec = matched_codec.entry(c.audio_codec)
    hop = int(codec["hop"])
    names = list(c.condition_names)
    counts = dict(zip(names, (int(n) for n in c.condition_bucket_counts)))
    silence = P.silence_prefill_default()
    spec = CONTEXT.text_tokenizer_spec
    phonemes = None if not spec else spec.get("phonemes")
    return {
        "text_tokenizer_spec": spec or None,
        "text_normalization_enabled": True,
        "text_norm_languages": list(languages.CODES),
        "accents": CONTEXT.accent_names,
        "speaker_enabled": True,
        "speaker_conditioning_type": "raw_reference_prefix",
        "speaker_embedding_dim": c.speaker_embedding_dim,
        "speaker_embedding_codec": c.speaker_embedding_codec,
        "n_codebooks": c.n_codebooks,
        "codebook_sizes": list(c.codebook_sizes),
        "audio_codec": c.audio_codec,
        "max_tokens": CONTEXT.max_frames,
        "stream_decode_modes": ["prefix", "whole"],
        "stream_decode_default": "prefix",
        "silence_prefill": {
            "available": CONTEXT.silence_available, "default_frames": silence.frames, "default_on": silence.on,
            "default_seconds": round(silence.frames * hop / SAMPLE_RATE, 3), "frame_seconds": hop / SAMPLE_RATE,
            "codec": c.audio_codec,
            "note": "codec-matched silence frames open the audio stream before the model starts (their audio is "
                    "not emitted); a request without silence_prefill_frames gets the server default, "
                    "silence_prefill_frames=0 turns it off and silence_prefill_frames=N sets the length"},
        "tag_cfg": {"available": True, "trained": trained_on_tags(resolve_dir_or_local(RELEASE_LM)),
                    "tags": list(NONVERBAL_MARKUP_TAGS), "min_scale": TAG_CFG_MIN_SCALE,
                    "max_scale": TAG_CFG_MAX_SCALE, "default_scale": 1.0},
        "decode_context": {"available": True, "default_frames": DECODE_CONTEXT_FRAMES, "max_frames": 450,
                           "note": "decode the generated frames behind the last decode_context_frames of the "
                                   "raw-reference prompt (their audio dropped), so the first generated frame is not "
                                   "the decoder's position 0; no-op without a reference prompt"},
        "speaker_modes": modes,
        "speaker_mode_default": modes[0],
        "decoder": {"kind": "istft_mr", "label": f"Multi-resolution iSTFT head ({Path(codec['checkpoint']).name}); "
                                                 "causal",
                    "checkpoint": Path(codec["checkpoint"]).name, "expect_step": None,
                    "headroom_db": float(matched_codec.decoder_headroom_db(c.audio_codec)), "causal": True,
                    "codec": c.audio_codec},
        "condition_tokens": None if c.condition_count == 0 else {
            "enabled": True, "names": names, "flags": {}, "count": c.condition_count, "bucket_counts": counts,
            "unknown_ids": dict(counts), "edges": {n: [float(v) for v in c.condition_edges.get(n, ())] for n in names},
            "units": {n: c.condition_units[n] for n in names if n in c.condition_units},
            "layout": "[reference prefix | speaker row]" + "".join(f"[{n}]" for n in names) + " BOS text EOS"},
        "speaker_audio_upload": True,
        "speaker_cfg_supported": True,
        "speaker_cfg_modes": modes,
        "speaker_cfg_unconditional": "remove_reference_prefix_or_speaker_row_condition_rows_unknown_keep_text_and_"
                                     "generated_history",
        "speaker_cfg_formula": "uncond + scale * (cond - uncond)",
        "laci_languages": sorted(CONTEXT.laci_languages),
        "dub": False,
        "speaker_embedding_upload": True,
        "speaker_embedding_cache": True,
        "speaker_embedding_blend": True,
        "default_voices_enabled": True,
        "projects": {"enabled": False},
        **edit_capabilities(EDIT_MODEL, True),
        "phoneme_tokens": None if phonemes is None else {
            "open_marker": phonemes["open_marker"], "close_marker": phonemes["close_marker"],
            "symbols": list(phonemes["symbols"])},
        "phonemize_enabled": False,
        "phonemize_languages": [],
        "phonemize_guess_languages": [],
    }


CAPABILITIES = capabilities()
CONTROLS = controls_document(ASSETS.steer_bundle, speaker_embedding_dim=CONFIG.speaker_embedding_dim)


# --------------------------------------------------------------------------------------------------- voices


@dataclass
class Voice:
    """A default voice (default_voices/) or a session voice (POST /v2/speakers, per X-TTS-Session-ID)."""

    speaker_id: str
    label: str
    source_type: Literal["audio", "embedding"]
    created_at: float
    is_default: bool
    original_name: str | None = None
    audio: bytes | None = None  # an audio voice's clip
    codes: list | None = None  # a default voice's shipped <stem>.codes.npy (reference prefix codes)
    embedding: torch.Tensor | None = None
    preview_wav: bytes | None = None


def default_voices() -> dict[str, Voice]:
    """The clips of default_voices/ (ids ``default_<sha1[:16]>`` of the file name, as the servers name them)."""
    voices = {}
    for path in sorted((ROOT / "default_voices").glob("*.mp3")):
        speaker_id = "default_" + hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:16]
        codes = path.with_name(path.stem + ".codes.npy")
        voices[speaker_id] = Voice(speaker_id, path.stem.replace("_", " ").replace("-", " "), "audio",
                                   path.stat().st_mtime, True, path.name, path.read_bytes(),
                                   np.load(codes).tolist() if codes.is_file() else None)
    return voices


DEFAULT_VOICES = default_voices()
SESSIONS: dict[str, dict[str, Voice]] = {}


def serialize(voice: Voice) -> dict:
    return {
        "id": voice.speaker_id, "label": voice.label, "source_type": voice.source_type,
        "scope": "default" if voice.is_default else "session", "is_default": voice.is_default,
        "original_name": voice.original_name, "created_at": voice.created_at,
        "has_preview": voice.source_type == "audio",
        "speaker_modes": ["embedding"] if voice.source_type == "embedding" else list(CONTEXT.speaker_modes),
        "embedding_dim": CONFIG.speaker_embedding_dim, "transcript": None,
        "disabled_controls": disabled_controls(ASSETS.steer_bundle, voice.speaker_id),
    }


def session_of(header: str | None) -> str | None:
    return (header.strip() or None) if header is not None else None


def find_voice(session_id: str | None, speaker_id: str) -> Voice:
    voice = DEFAULT_VOICES.get(speaker_id)
    if voice is not None:
        return voice
    if session_id is None:
        raise ValueError("Cached speaker operations require the X-TTS-Session-ID header.")
    voice = SESSIONS.get(session_id, {}).get(speaker_id)
    if voice is None:
        raise ValueError(f"Unknown cached speaker id '{speaker_id}'.")
    return voice


def embed_audio(audio: bytes) -> torch.Tensor:
    """The MECT speaker embedding of a clip, on this process's CPU (no GPU quota spent on it)."""
    req = P.Request(speaker_audio_base64=base64.b64encode(audio).decode("ascii"), speaker_embedding_base64=None)
    return _speaker_embedding(LM, req, ASSETS, CPU)


async def voice_embedding(voice: Voice) -> torch.Tensor:
    if voice.embedding is None:
        voice.embedding = await asyncio.to_thread(embed_audio, voice.audio)
    return voice.embedding.detach().to(torch.float32).clone()


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    v0, v1 = v0.reshape(-1).float(), v1.reshape(-1).float()
    if not 0.0 <= t <= 1.0:
        raise ValueError("speaker_blend_t must be between 0.0 and 1.0.")
    a, b = v0 / (torch.linalg.vector_norm(v0) + 1e-8), v1 / (torch.linalg.vector_norm(v1) + 1e-8)
    omega = torch.acos(torch.clamp(torch.dot(a, b), -1.0, 1.0))
    if float(omega) < 1e-6:
        return ((1.0 - t) * v0 + t * v1).contiguous()
    s = torch.sin(omega)
    return ((torch.sin((1.0 - t) * omega) / s) * v0 + (torch.sin(t * omega) / s) * v1).contiguous()


async def resolve_embedding(session_id: str | None, req: P.Request) -> torch.Tensor | None:
    """speaker_mode=embedding: an upload, an embedding file, one voice id or a blend of two (the servers' rules)."""
    direct = sum(bool(v) for v in (req.speaker_audio_base64, req.speaker_embedding_base64))
    if direct > 1:
        raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")
    cached = bool(req.speaker_embedding_id)
    blended = bool(req.speaker_blend_embedding_id_a or req.speaker_blend_embedding_id_b)
    selected = sum((direct == 1, cached, blended))
    if selected == 0:
        return None
    if selected > 1:
        raise ValueError("Provide speaker conditioning via upload, one cached speaker id, or two cached ids for "
                         "blending.")
    if blended:
        if not req.speaker_blend_embedding_id_a or not req.speaker_blend_embedding_id_b:
            raise ValueError("Provide both cached speaker ids to blend between two speakers.")
        a = await voice_embedding(find_voice(session_id, req.speaker_blend_embedding_id_a))
        b = await voice_embedding(find_voice(session_id, req.speaker_blend_embedding_id_b))
        return slerp(a, b, 0.5 if req.speaker_blend_t is None else float(req.speaker_blend_t))
    if cached:
        return await voice_embedding(find_voice(session_id, req.speaker_embedding_id))
    if req.speaker_embedding_base64:
        return load_embedding_vector(decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64"),
                                     expected_dim=CONFIG.speaker_embedding_dim, file_name=req.speaker_embedding_name)
    raw = decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64")
    return await asyncio.to_thread(embed_audio, raw)


def npy_base64(vector: torch.Tensor) -> str:
    buf = io.BytesIO()
    np.save(buf, vector.reshape(-1).numpy().astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def prepare(body: dict, session_id: str | None) -> dict:
    """The request the reference synthesizes: speaker ids, blends and embedding-mode clips resolved here (embeddings
    as .npy, a voice's reference codes or clip), so the GPU call only generates."""
    req = P.parse_request(body)
    mode = P.request_head(CONTEXT, req).mode
    ids = {"speaker_embedding_id": None, "speaker_blend_embedding_id_a": None, "speaker_blend_embedding_id_b": None,
           "speaker_blend_t": None}
    if mode == "embedding":
        vector = await resolve_embedding(session_id, req)
        if vector is not None:
            body.update(ids, speaker_audio_base64=None, speaker_audio_name=None,
                        speaker_embedding_base64=npy_base64(vector), speaker_embedding_name=None)
        return body
    if req.speaker_embedding_base64 or req.speaker_blend_embedding_id_a or req.speaker_blend_embedding_id_b:
        raise ValueError("This checkpoint conditions on raw reference audio and does not accept speaker embedding "
                         "files or embedding blends.")
    if req.speaker_embedding_id:
        if req.speaker_audio_base64 or req.speaker_reference_codes is not None:
            raise ValueError("Provide either reference audio or codes, or a cached speaker id, not both.")
        voice = find_voice(session_id, req.speaker_embedding_id)
        if voice.source_type != "audio":
            raise ValueError(f"Speaker '{voice.label}' is an embedding; speaker_mode=reference needs reference audio.")
        body.update(ids)
        if voice.codes is not None:
            body["speaker_reference_codes"] = voice.codes
        else:
            body.update(speaker_audio_base64=base64.b64encode(voice.audio).decode("ascii"),
                        speaker_audio_name=voice.original_name)
    return body


# ------------------------------------------------------------------------------------------------------ GPU


def gpu_seconds(body) -> int:
    """A job's worst-case GPU time. A take: 10 s, +60 s for encoding an uploaded clip in reference mode (a fresh GPU
    worker loads the dual codec first), and 0.15 s per character (twice that with a guidance twin). An edit call: 90 s
    (a fresh worker loads Qwen3-ASR or FuseAlign or the dual codec on its first one)."""
    if isinstance(body, tuple):
        return 90
    clip = 60 if body["speaker_mode"] == "reference" and body["speaker_audio_base64"] else 0
    guided = any(float(body[k]) != 1.0 for k in ("speaker_cfg_scale", "tag_cfg_scale", "controls_cfg_scale"))
    return min(240, int(10 + clip + 0.15 * len(body["text"]) * (2 if guided else 1)))


# ------------------------------------------------------------------------------------------------- speech editing

# Speech editing (/v2/edit/transcribe, /v2/edit/align, /v2/edit): zonos3.tts.edit_runtime.EditRuntime in this process,
# its EditEngine (SpaceEditEngine) and its ASR / aligner calls (InProcessAsr, InProcessAligner) sent to the GPU session
# as edit jobs. The LM is the release LM grown to the edit checkpoint's vocabulary (its span-end and marker rows; plain
# takes never read or sample them, so they are unchanged), and every edit region runs with the edit LoRA applied per
# request (zonos3.reference.accents, strength 1): release + LoRA, as trained. Qwen3-ASR and FuseAlign load in the GPU
# worker on first use, beside the LM.
# Fetched now (no GPU time on downloads); loaded by a GPU worker's first edit job.
resolve_dir_or_local("hf://Qwen/Qwen3-ASR-1.7B")
resolve_file(f"hf://{FUSEALIGN_REPO}/{FUSEALIGN_CHECKPOINT}")
_WORKERS: dict = {}  # the GPU worker's Qwen3-ASR model and FuseAlign aligner (each GPU worker loads its own)


def _asr():
    if "asr" not in _WORKERS:
        from qwen_asr import Qwen3ASRModel

        _WORKERS["asr"] = Qwen3ASRModel.from_pretrained(resolve_dir_or_local("hf://Qwen/Qwen3-ASR-1.7B"),
                                                        dtype=torch.bfloat16, device_map="cuda:0",
                                                        max_inference_batch_size=32, max_new_tokens=1024)
    return _WORKERS["asr"]


def _aligner():
    if "aligner" not in _WORKERS:
        from fusealign_hub import load_aligner

        aligner = load_aligner(resolve_file(f"hf://{FUSEALIGN_REPO}/{FUSEALIGN_CHECKPOINT}"), device="cuda:0")
        aligner.whisper_words = fusealign_worker.no_whisper  # anchors always come from Qwen3-ASR
        _WORKERS["aligner"] = aligner
    return _WORKERS["aligner"]


def edit_job(op: str, args: dict):
    """One edit call in the GPU worker: the ops of zonos3/workers/asr.py and fusealign.py, run in-process, and the
    EditEngine's GPU work (encode, fill, decode)."""
    from fusealign_hub.longform import guide_words
    from fusealign_hub.vad import speech_regions

    if op == "transcribe":
        return asr_worker.transcribe(_asr(), SUPPORTED_LANGUAGES, args["items"])
    if op == "segments":
        return fusealign_worker.segments(args["items"], speech_regions)
    if op == "speech":
        return fusealign_worker.speech(args["items"], speech_regions)
    if op == "align":
        return fusealign_worker.align(_aligner(), args["items"], guide_words)
    if op == "encode":
        wav, sample_rate = _decode_audio_bytes(args["audio"])
        return encode_dual_reference(wav, sample_rate, max_seconds=args["max_seconds"], device="cuda:0").numpy()
    if op == "decode":
        codes = torch.as_tensor(args["codes"], dtype=torch.int64)
        model = ASSETS.decoder(CONFIG.audio_codec, DEVICE)
        audio = decode_model(model, codes.unsqueeze(0), matched_codec.entry(CONFIG.audio_codec))[0].numpy()
        return (audio * np.float32(output_gain(CONFIG.audio_codec, None))).astype(np.float32)
    if op == "fill":
        layout = EditLayout(torch.as_tensor(args["codes"], dtype=torch.int64), args["spans"],
                            [torch.as_tensor(f, dtype=torch.int32) for f in args["fills"]])
        embedding = args["embedding"]
        rows, eos_frame = edit_region(LM, args["body"], assets=ASSETS, layout=layout, min_frames=args["min_frames"],
                                      speaker_embedding=None if embedding is None else embedding.to(DEVICE),
                                      hidden_delta=args["hidden_delta"], accents=EDIT_LORA, device=DEVICE)
        n_cb = CONFIG.n_codebooks
        # the finishing row is not part of the generated span (it only completes the delayed codebooks' flush)
        out = (torch.stack(rows[:-1])[:, :n_cb].numpy() if len(rows) > 1 else np.zeros((0, n_cb), dtype=np.int32))
        return out, -1 if eos_frame is None else int(eos_frame)
    raise ValueError(f"unknown edit job {op!r}")


# Takes run in a GPU session: one ZeroGPU call that holds its GPU and serves takes over a pipe until HOLD_SECONDS pass
# without one or its SESSION_SECONDS run out. A ZeroGPU call per take paid for a fresh GPU worker every time
# (measured: each call got a new process, ~1.2 s to start, plus ~5 s "Waiting for a GPU" when the node was busy, and
# pings between takes did not keep the GPU). A session close to its end is renewed right after a take, so the next one
# starts while the visitor listens. ZeroGPU bills the held time, idle included, to the visitor whose take opened the
# session (its X-IP-Token).
SESSION_SECONDS = 240
HOLD_SECONDS = 60
RENEW_SECONDS = 45  # renew a session with less than this left after a take
MARGIN_SECONDS = 5  # a session stops taking work this long before its ZeroGPU deadline
_FORK = multiprocessing.get_context("fork")  # spaces forks its GPU workers, which inherit both pipes
TAKE_RX, TAKE_TX = _FORK.Pipe(duplex=False)  # (take id, request body) or None (close), main process -> session
RESULT_RX, RESULT_TX = _FORK.Pipe(duplex=False)  # (take id, kind, payload), session -> main process
# Prefix streaming as the servers do it (15-frame chunks behind a 3-frame hold-back), but the first chunk once 4 frames
# are decodable: first audio 16 rows into the take instead of 27.
STREAM = StreamSettings(first_chunk=4)


@spaces.GPU(duration=lambda seconds: seconds)
def gpu_session(seconds: int) -> int:
    """Serve jobs until none comes for HOLD_SECONDS, None arrives or ``seconds`` (less the margin) are up; returns the
    number served. A job is a take (a GenerateRequest body) or an edit call (``("edit", op, args)``: ``edit_job``). A
    take's messages: "start" (seed, whether painted spans follow the reading) once planned, "audio" (a float32 PCM chunk)
    as it is decoded, then "end"; an edit call's: "result" (its value). Either may end "invalid" (a request the
    reference refused) or "error" (the session then ends: the next job gets a fresh worker)."""
    deadline, served = time.monotonic() + seconds - MARGIN_SECONDS, 0
    while (left := deadline - time.monotonic()) > 0 and TAKE_RX.poll(min(HOLD_SECONDS, left)):
        message = TAKE_RX.recv()
        if message is None:
            break
        job_id, body = message
        served += 1
        try:
            if isinstance(body, tuple):
                RESULT_TX.send((job_id, "result", edit_job(*body[1:])))
                continue
            synthesize(LM, body, assets=ASSETS, device=DEVICE, stream=STREAM,
                       on_plan=lambda plan: RESULT_TX.send((job_id, "start",
                                                            (plan.seed, plan.follow_tokens is not None))),
                       on_audio=lambda pcm: RESULT_TX.send((job_id, "audio", pcm)))
        except ValueError as exc:
            RESULT_TX.send((job_id, "invalid", str(exc)))
        except Exception as exc:
            RESULT_TX.send((job_id, "error", f"{type(exc).__name__}: {exc}"))
            break
        else:
            RESULT_TX.send((job_id, "end", None))
    return served


@dataclass
class Take:
    body: dict
    request: Request
    # (kind, payload) as the session sends them, then "end" / "invalid" / "error"; "skipped" when the client left
    # before the take ran, "failed" (payload: the exception) when no session could run it
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    last: tuple | None = None  # the message that ended its audio


class GpuSessions:
    """The main-process side: one take at a time, on the open session or a new one. A take whose client left while it
    waited is dropped (the studio aborts the previous render 250 ms after every change in the voice design)."""

    def __init__(self) -> None:
        self.takes: asyncio.Queue[Take] = asyncio.Queue()
        self.session: asyncio.Task | None = None
        self.opened = self.last = 0.0  # time.monotonic() when it opened and when its last take ended
        self.seconds = 0  # the open session's ZeroGPU duration
        self.ids = 0
        self.runner: asyncio.Task | None = None

    def submit(self, body: dict, request: Request) -> Take:
        if self.runner is None or self.runner.done():
            self.runner = asyncio.create_task(self.serve(), context=contextvars.Context())
        take = Take(body, request)
        self.takes.put_nowait(take)
        return take

    async def serve(self) -> None:
        while True:
            take = await self.takes.get()
            try:
                if await take.request.is_disconnected():
                    take.out.put_nowait(("skipped", None))
                    continue
                await self.take(take)
            except Exception as exc:
                take.out.put_nowait(("failed", exc))
                continue
            if self.open() and self.left() < RENEW_SECONDS:
                await self.close()
                self.start(take.request, SESSION_SECONDS)

    def open(self) -> bool:
        return self.session is not None and not self.session.done()

    def left(self) -> float:
        """Seconds the open session can still take work for."""
        return self.opened + self.seconds - MARGIN_SECONDS - time.monotonic()

    async def close(self) -> None:
        await asyncio.to_thread(TAKE_TX.send, None)
        await asyncio.wait([self.session])

    async def take(self, take: Take) -> None:
        need = gpu_seconds(take.body)
        if self.open() and (need > self.left() or time.monotonic() - self.last > HOLD_SECONDS - MARGIN_SECONDS):
            await self.close()  # it would run out mid-take, or is about to idle out
        self.ids += 1
        sender = None  # a large take (a clip) blocks in the pipe until a session reads it: send alongside
        for _ in range(2):  # a session that idled out as the take was sent left it in the pipe for the next one
            if not self.open():
                self.start(take.request, max(SESSION_SECONDS, need + MARGIN_SECONDS + 1))
            if sender is None:
                sender = asyncio.create_task(asyncio.to_thread(TAKE_TX.send, (self.ids, take.body)))
            done = await self.pump(self.ids, take.out)
            if done is not None:
                await sender
                self.last = time.monotonic()
                if not done:
                    take.out.put_nowait(("error", "the GPU session ended mid-take"))
                return
            if self.session.exception() is not None:
                await self.drain(sender)
                raise self.session.exception()
        await self.drain(sender)
        raise RuntimeError("the GPU session ended without running the take")

    def start(self, request: Request, seconds: int) -> None:
        """A session on behalf of ``request``'s visitor (ZeroGPU reads the X-IP-Token from Gradio's request context)."""
        context = contextvars.Context()
        context.run(LocalContext.request.set, gr.Request(request=request))
        self.session = asyncio.create_task(asyncio.to_thread(gpu_session, seconds), context=context)
        self.session.add_done_callback(lambda s: print(f"[gpu] session closed: {s.exception() or s.result()} take(s)",
                                                       flush=True))
        self.opened = self.last = time.monotonic()
        self.seconds = seconds
        print(f"[gpu] session opened ({seconds} s)", flush=True)

    async def pump(self, take_id: int, out: asyncio.Queue) -> bool | None:
        """Forward the take's messages to ``out``: True once its last one passed, False when the session ended
        mid-take, None when it ended before the take began."""
        began = False
        while True:
            if await asyncio.to_thread(RESULT_RX.poll, 0.25):
                rid, kind, payload = await asyncio.to_thread(RESULT_RX.recv)
                if rid != take_id:
                    continue
                began = True
                out.put_nowait((kind, payload))
                if kind in ("end", "result", "invalid", "error"):
                    return True
            elif self.session.done() and not RESULT_RX.poll(0):
                return began or None

    @staticmethod
    async def drain(sender: asyncio.Task) -> None:
        """Read back what a failed session left unread (unblocking the sender), so no later session runs it."""
        while not sender.done() or TAKE_RX.poll(0):
            if await asyncio.to_thread(TAKE_RX.poll, 0.05):
                await asyncio.to_thread(TAKE_RX.recv)


GPU = GpuSessions()


async def edit_call(request: Request, op: str, **args):
    """One edit job on the GPU session (``edit_job``), on behalf of ``request``'s visitor: its value, ValueError for
    a request the reference refused, RuntimeError for a failed job."""
    job = GPU.submit(("edit", op, args), request)
    kind, payload = await job.out.get()
    if kind == "result":
        return payload
    if kind == "skipped":
        raise RuntimeError("the client left before the edit ran")
    if kind == "failed":
        raise payload
    raise (ValueError if kind == "invalid" else RuntimeError)(payload)


# The HTTP request and the event loop of the edit route being served (set by the edit_request_context middleware):
# the runtime's ASR / aligner calls run in worker threads (asyncio.to_thread copies this context into them).
EDIT_REQUEST: contextvars.ContextVar[tuple[Request, asyncio.AbstractEventLoop]] = contextvars.ContextVar("edit_request")


def edit_call_sync(op: str, **args):
    """``edit_call`` from a worker thread of the edit runtime."""
    request, loop = EDIT_REQUEST.get()
    return asyncio.run_coroutine_threadsafe(edit_call(request, op, **args), loop).result()


class InProcessAsr:
    """``zonos3.tts.edit_align.AsrWorker`` on the GPU session (Qwen3-ASR loaded in the GPU worker). Audio items are the
    runtime's temp WAV paths: the forked GPU worker shares this file system."""

    def languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    def transcribe(self, items: list[dict]) -> list[dict]:
        return edit_call_sync("transcribe", items=items)


class InProcessAligner:
    """``zonos3.tts.edit_align.FuseAlignWorker`` on the GPU session (FuseAlign loaded in the GPU worker)."""

    def segments(self, audios: list[str]) -> list[list[list[float]]]:
        return [r["chunks"] for r in edit_call_sync("segments", items=[{"audio": a} for a in audios])]

    def speech(self, audios: list[str]) -> list[list[list[float]]]:
        return [r["regions"] for r in edit_call_sync("speech", items=[{"audio": a} for a in audios])]

    def align(self, items: list[dict]) -> list[list[dict]]:
        return [r["words"] for r in edit_call_sync("align", items=items)]


class SpaceEditEngine:
    """``zonos3.tts.edit_runtime.EditEngine`` of the Space: uploads decode and the speaker row embeds on this process's
    CPU; encoding the clip, each region's fill (the edit LoRA on) and the decode run on the GPU session."""

    model = EDIT_MODEL

    async def decode_audio(self, audio: bytes) -> tuple[np.ndarray, int]:
        wav, sample_rate = await asyncio.to_thread(_decode_audio_bytes, audio)
        return wav[0].numpy(), int(sample_rate)

    async def embed(self, audio: bytes) -> torch.Tensor:
        return await asyncio.to_thread(embed_audio, audio)

    async def encode(self, audio: bytes, max_seconds: float) -> np.ndarray:
        codes = await edit_call(EDIT_REQUEST.get()[0], "encode", audio=audio, max_seconds=float(max_seconds))
        return np.asarray(codes, dtype=np.int64)

    async def check(self, target_text: str, doses, condition_buckets, condition_values) -> set[str]:
        P.validate_phoneme_groups(CONTEXT.text_tokenizer_spec, target_text)
        raw = set()
        if doses is not None:
            for region in doses:
                check_controls(ASSETS.steer_bundle, region)
            raw = {did for region in doses for did in region if ASSETS.steer_bundle.directions[did].space == RAW_SPACE}
        P.resolve_condition_buckets(CONTEXT, condition_buckets, condition_values)
        return raw

    async def fill(self, job: FillJob) -> tuple[np.ndarray, int]:
        embedding, delta = job.speaker_embedding, None
        if job.controls:
            resolved = resolve_controls(ASSETS.steer_bundle, job.controls, None, text=job.controls_text,
                                        speaker_embedding=embedding, sources=CONTEXT.steering_sources,
                                        hidden_size=CONTEXT.hidden_size)
            embedding, delta = resolved.speaker_embedding, resolved.hidden_delta
        request = {"text": job.text, "language": job.language, "text_normalization": job.text_normalization,
                   **job.sampling, "seed": job.seed, "max_tokens": job.max_tokens, "speaker_mode": "embedding",
                   "condition_buckets": job.condition_buckets, "condition_values": job.condition_values,
                   "stream": False}
        return await edit_call(EDIT_REQUEST.get()[0], "fill", body=request, codes=np.asarray(job.window_codes),
                               spans=job.spans, fills=[np.asarray(f) for f in job.fills], min_frames=int(job.min_frames),
                               embedding=embedding, hidden_delta=delta)

    async def decode(self, codes: np.ndarray) -> np.ndarray:
        return await edit_call(EDIT_REQUEST.get()[0], "decode", codes=np.asarray(codes))


EDIT_RUNTIME = EditRuntime(SpaceEditEngine(), InProcessAsr(), InProcessAligner())


async def edit_route(request: Request, run):
    """An edit route (zonos3.server.edit): its GPU jobs run on behalf of its visitor (their X-IP-Token)."""
    EDIT_REQUEST.set((request, asyncio.get_running_loop()))
    return await edit_errors(run(EDIT_RUNTIME))


async def audio_of(take: Take):
    """The take's PCM chunks until its audio ends (``take.last`` says how)."""
    while (message := await take.out.get())[0] == "audio":
        yield message[1]
    take.last = message
    if message[0] != "end":
        print(f"[gpu] take ended early: {message}", flush=True)


def pcm16_wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(pcm, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue()


# ----------------------------------------------------------------------------------------------------- app

app = gr.Server(title="ZONOS3 Mini", openapi_url=None, docs_url=None, redoc_url=None)


@app.post("/v2/edit/transcribe")
async def edit_transcribe(req: EditTranscribeRequest, request: Request):
    """Qwen3-ASR transcript of the audio to edit (chunk-wise past 30 s) and its words' FuseAlign timings."""
    return await edit_route(request, lambda rt: rt.transcribe(req))


@app.post("/v2/edit/align")
async def edit_align(req: EditAlignRequest, request: Request):
    """FuseAlign forced alignment of the audio to ``text``: word timings, presences and edit tokens."""
    return await edit_route(request, lambda rt: rt.align(req))


@app.post("/v2/edit")
async def edit(req: EditRequest, request: Request):
    """Regenerate the edited regions of recorded speech (the edit LoRA on); unedited codec frames are kept exactly."""
    return await edit_route(request, lambda rt: rt.edit(req))


@app.get("/v2/capabilities")
async def get_capabilities():
    return CAPABILITIES


@app.get("/v2/controls")
async def get_controls():
    return CONTROLS


@app.get("/v2/speakers")
async def list_speakers(x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
    session = SESSIONS.get(session_of(x_tts_session_id) or "", {})
    voices = sorted(DEFAULT_VOICES.values(), key=lambda v: v.label.lower())
    voices += sorted(session.values(), key=lambda v: v.created_at, reverse=True)
    return {"speakers": [serialize(v) for v in voices]}


@app.post("/v2/speakers")
async def create_speaker(req: SpeakerCreateRequest,
                         x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
    try:
        session_id = session_of(x_tts_session_id)
        if session_id is None:
            raise ValueError("Cached speaker operations require the X-TTS-Session-ID header.")
        if req.speaker_audio_base64 and req.speaker_embedding_base64:
            raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")
        speaker_id = f"spk_{uuid.uuid4().hex[:12]}"
        if req.speaker_embedding_base64:
            name = req.speaker_embedding_name
            embedding = load_embedding_vector(decode_base64_blob(req.speaker_embedding_base64,
                                                                 "speaker_embedding_base64"),
                                              expected_dim=CONFIG.speaker_embedding_dim, file_name=name)
            voice = Voice(speaker_id, label_of(req.label, name, "Speaker embedding"), "embedding", time.time(),
                          False, name, embedding=embedding)
        elif req.speaker_audio_base64:
            name = req.speaker_audio_name
            audio = decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64")
            voice = Voice(speaker_id, label_of(req.label, name, "Speaker audio"), "audio", time.time(), False, name,
                          audio, preview_wav=await asyncio.to_thread(_transcode_audio_bytes_to_wav, audio))
            await voice_embedding(voice)
        else:
            raise ValueError("Provide an audio file or saved embedding file to cache.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    SESSIONS.setdefault(session_id, {})[speaker_id] = voice
    return serialize(voice)


def label_of(explicit: str | None, name: str | None, fallback: str) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if name and Path(name).stem.strip():
        return Path(name).stem.strip()
    return fallback


@app.get("/v2/speakers/{speaker_id}/preview")
async def preview(speaker_id: str, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
    try:
        voice = find_voice(session_of(x_tts_session_id), speaker_id)
        if voice.source_type != "audio":
            raise HTTPException(status_code=404, detail="This voice has no preview audio.")
        if voice.preview_wav is None:
            voice.preview_wav = await asyncio.to_thread(_transcode_audio_bytes_to_wav, voice.audio)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=voice.preview_wav, media_type="audio/wav")


@app.get("/v2/speakers/{speaker_id}/embedding")
async def embedding(speaker_id: str, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
    try:
        vector = await voice_embedding(find_voice(session_of(x_tts_session_id), speaker_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": speaker_id, "embedding": [round(float(x), 6) for x in vector.reshape(-1)]}


@app.post("/v2/generate")
async def generate(req: GenerateRequest, request: Request,
                   x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
    """The take on the GPU session (GpuSessions): float32 PCM streamed as it is decoded, or buffered (stream=false, and
    16-bit WAV)."""
    if req.stream and req.format == "wav":
        raise HTTPException(status_code=400, detail="format=wav needs stream=false")
    try:
        body = await prepare(req.model_dump(), session_of(x_tts_session_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    take = GPU.submit(body, request)
    kind, payload = await take.out.get()
    if kind == "skipped":
        return Response(status_code=499)  # the studio replaced this take while it waited for the GPU
    if kind == "failed":
        if isinstance(payload, gr.Error):  # ZeroGPU refused or lost the session (quota, queue, worker)
            status = 429 if payload.title.startswith("ZeroGPU quota") else 503
            raise HTTPException(status_code=status, detail=payload.message) from payload
        raise payload
    if kind != "start":
        raise HTTPException(status_code=400 if kind == "invalid" else 500, detail=payload)
    seed, reading = payload
    headers = {"X-Audio-Sample-Rate": str(SAMPLE_RATE), "X-Audio-Channels": "1",
               "X-Audio-Format": "pcm16" if req.format == "wav" else "float32", "X-Seed": str(seed)}
    if req.control_spans:
        headers["X-Control-Span-Follow"] = "reading" if reading else "rate"
    if req.stream:  # a failure mid-take ends the stream: the status line is already sent
        return StreamingResponse((pcm.astype(np.float32).tobytes() async for pcm in audio_of(take)),
                                 media_type="audio/pcm", headers=headers)
    pcm = [chunk async for chunk in audio_of(take)]
    kind, payload = take.last
    if kind != "end":
        raise HTTPException(status_code=400 if kind == "invalid" else 500, detail=payload)
    pcm = np.concatenate(pcm) if pcm else np.zeros(0, dtype=np.float32)
    if req.format == "wav":
        return Response(content=pcm16_wav(pcm), media_type="audio/wav", headers=headers)
    return Response(content=pcm.astype(np.float32).tobytes(), media_type="audio/pcm", headers=headers)


# The studio's pages (/ is the chooser, /studio, /lite: never cached) and its hashed assets (immutable). Explicit
# routes, no catch-all: Gradio's own routes (/gradio_api/..., which launch() calls) stay reachable.
#
# A private Space opens inside huggingface.co as https://<space>.hf.space/?__sign=<jwt>; the HF proxy trades that for
# its `spaces-jwt` cookie, which authorizes every later request (assets, the other pages, /v2). That cookie is not
# Partitioned, so a browser that blocks third-party cookies drops it in the huggingface.co frame and every later
# request 404s. A page opened with __sign sets the same token again as a Partitioned (CHIPS) cookie, which such
# browsers keep for this frame.
JWT = re.compile(r"[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")


def page(name: str, request: Request) -> FileResponse:
    response = FileResponse(STUDIO / name, media_type="text/html", headers={"Cache-Control": "no-store, max-age=0"})
    sign = request.query_params.get("__sign", "")
    if JWT.fullmatch(sign):
        response.headers.append("Set-Cookie",
                                f"spaces-jwt={sign}; Path=/; Secure; HttpOnly; SameSite=None; Partitioned")
    return response


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def chooser(request: Request):
    return page("index.html", request)


@app.api_route("/studio", methods=["GET", "HEAD"], include_in_schema=False)
async def studio(request: Request):
    return page("studio.html", request)


@app.api_route("/lite", methods=["GET", "HEAD"], include_in_schema=False)
async def lite(request: Request):
    return page("lite.html", request)


@app.api_route("/assets/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
async def asset(path: str):
    root = (STUDIO / "assets").resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})


app.launch(ssr_mode=False)
