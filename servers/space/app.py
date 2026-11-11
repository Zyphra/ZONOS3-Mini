"""ZONOS3 Mini on a ZeroGPU Space: the Voice Studio (studio/dist) over the v2 HTTP API, served by the plain-PyTorch
reference (zonos3.reference) with generation on ZeroGPU.

``gr.Server`` is the FastAPI app; its ``launch()`` runs through ``gr.Blocks.launch``, where ``spaces`` starts ZeroGPU.
The routes are the part of servers/api/openapi.yaml the studio and the lite page use when the server has no plugins or project
store (``/v2/capabilities`` says so and the studio adapts): GET ``/v2/capabilities``, ``/v2/controls``,
``/v2/speakers``, ``/v2/speakers/{id}/preview``, ``/v2/speakers/{id}/embedding``, POST ``/v2/speakers``,
``/v2/generate`` (prefix-streamed float32 PCM, chunks ramping from 1 frame), ``/v2/phonemize`` (the gold lexicon, espeak-ng
guesses) and speech editing (``/v2/edit/transcribe``, ``/v2/edit/align``, ``/v2/edit``: the edit LoRA per request).
Pronunciation groups read the release phoneme tokens, appended at load after the edit vocabulary. Speaker embeddings
run on this process's CPU; generation, edit regions, Qwen3-ASR and FuseAlign run in a held GPU session
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
import io
import multiprocessing
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from fastapi import Header, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fusealign_hub.aligner import CHECKPOINT_FILE as FUSEALIGN_CHECKPOINT
from fusealign_hub.aligner import HF_REPO as FUSEALIGN_REPO
from gradio.context import LocalContext
from qwen_asr.inference.utils import SUPPORTED_LANGUAGES
from zonos3.generation import SAMPLE_RATE
from zonos3.models.accent_format import load_adapter
from zonos3.models.text_extra import TextExtraRows
from zonos3.reference.accents import AccentState
from zonos3.reference.lm import Zonos3LM
from zonos3.reference.synth import (
    EditLayout,
    ReferenceAssets,
    ReferenceContext,
    _speaker_embedding,
    decode_base64_blob,
    edit_region,
    load_embedding_vector,
    output_gain,
    slerp,
    synthesize,
)
from zonos3.server.capabilities import (
    CapabilityFacts,
    capabilities_document,
    condition_spec,
    decoder_capabilities,
    dual_codec_hop,
)
from zonos3.server.errors import call as edit_errors
from zonos3.server.schemas import (
    EditAlignRequest,
    EditRequest,
    EditTranscribeRequest,
    GenerateRequest,
    PhonemizeRequest,
    SpeakerCreateRequest,
)
from zonos3.server.speech import check_format, prefix_stream_supported, resolve_decoder, speech_headers
from zonos3.server.studio import resolve_studio_file, studio_cache_control
from zonos3.server.voices import (
    DefaultVoice,
    SessionVoice,
    new_speaker_id,
    normalize_session_id,
    read_voice_file,
    require_session_id,
    scan_default_voices,
    serialize_default_voice,
    serialize_session_voice,
    speaker_label,
    voice_preview,
    voice_sidecar,
    voice_speaker_modes,
)
from zonos3.tokenizer import matched_codec
from zonos3.tokenizer.dual_reference import encode_dual_reference
from zonos3.tokenizer.matched_codec import decode_model
from zonos3.tts import plan as P
from zonos3.tts.audio import _decode_audio_bytes, _transcode_audio_bytes_to_wav, pcm16_wav_bytes
from zonos3.tts.directions import RAW_SPACE
from zonos3.tts.edit import edit_marker_rows
from zonos3.tts.edit_runtime import EditModel, EditRuntime, FillJob
from zonos3.tts.markup import trained_on_tags
from zonos3.tts.phonemes import PhonemeLexicon
from zonos3.tts.steering import (
    check_controls,
    controls_document,
    disabled_controls,
    resolve_controls,
)
from zonos3.utils.release import resolve_dir_or_local, resolve_file
from zonos3.workers import asr as asr_worker
from zonos3.workers import fusealign as fusealign_worker

ROOT = Path(__file__).resolve().parent
STUDIO = ROOT / "studio"
# One device object everywhere: synthesize() moves the LM when lm.device != device, and ReferenceAssets caches the
# decoder by str(device). On ZeroGPU this process only emulates CUDA; the GPU exists inside @spaces.GPU calls.
DEVICE = torch.device("cuda:0")
# The Space serves speech editing and pronunciation groups, whose edit LoRA, phoneme tokens and lexicon are only in the
# private release repo (the public Zyphra/ZONOS3-Mini has the plain LM): it keeps loading from there (HF_TOKEN secret).
SPACE_RELEASE_REPO = "Zyphra/ZONOS3_Mini_release_ckpts"
SPACE_LM = f"hf://{SPACE_RELEASE_REPO}/lm/rate30_20k_r2grpo"
CPU = torch.device("cpu")

# The edit LoRA (zonos-accent-lora/3) brings the vocabulary it was trained with: its span-end audio row and edit-marker
# text rows, which plain takes never read or sample (so they run exactly as on the release LM). Edit regions add the
# LoRA per request (EDIT_LORA). The phoneme tokens (zonos-text-extra/1, trained on the frozen release LM) follow the
# edit marker rows: pronunciation groups (U+E000 IPA U+E001) read them, text without groups never does. Grown on the
# CPU, then moved: ZeroGPU's emulated CUDA refuses mixing its tensors with CPU ones in this process.
EDIT_ADAPTER = load_adapter(resolve_file(f"{SPACE_LM}/edit_lora/z3m_omni3b_edit_lora.pt"))
PHONEME_ROWS = TextExtraRows.load(resolve_file(f"{SPACE_LM}/phoneme_tokens/phoneme_tokens_s15000.pt"))
LM = (
    Zonos3LM.load(SPACE_LM, device="cpu", dtype=torch.float32)
    .grow(EDIT_ADAPTER.grown)
    .append_text_rows(PHONEME_ROWS)
    .to(DEVICE)
)
CONFIG = LM.config
ASSETS = ReferenceAssets.load(steer_bundle=str(ROOT / "steering" / "studio_bundle_mect60k_x10.json"))
CONTEXT = ReferenceContext(LM, ASSETS, DEVICE)  # the request rules' view of the checkpoint (no GPU work)
EDIT_LORA = AccentState.for_model(LM, EDIT_ADAPTER, 1.0)
# The gold pronunciation lexicon behind "Phonemize selection"; espeak-ng (espeakng-loader) guesses the words it lacks.
_LEXICON = f"hf://{SPACE_RELEASE_REPO}/lexicons/gold_phonemes_v1"
resolve_file(f"{_LEXICON}/manifest.json")
LEXICON = PhonemeLexicon(str(Path(resolve_file(f"{_LEXICON}/data.mdb")).parent))
ASSETS.decoder(CONFIG.audio_codec, DEVICE)
ASSETS.speaker_encoder(CONFIG.speaker_embedding_codec, CONFIG.speaker_embedding_dim, CPU)
# A clip in reference mode is encoded by the dual codec inside the GPU call; fetch its teacher folder now.
_DUAL = next(e for e in matched_codec.registry().values() if e.get("family") == "dual")
resolve_dir_or_local(_DUAL["teacher"])


# --------------------------------------------------------------------------------------------- capabilities


# The Space decodes with the codec's default decoder only (zonos3.reference.synth): a request's `decoder` may name it.
DECODERS = [matched_codec.decoder_id(CONFIG.audio_codec)]
EDIT_MODEL = EditModel.from_config(
    CONFIG,
    hop=int(matched_codec.entry(CONFIG.audio_codec)["hop"]),
    max_spans=edit_marker_rows(CONFIG.text_extra_tokens, CONTEXT.text_tokenizer_spec),
)


def capabilities() -> dict:
    """GET /v2/capabilities (zonos3.server.capabilities) of this Space: the release LM with its speaker paths, the edit
    LoRA and the gold lexicon; no plugins, no project store."""
    c = CONFIG
    modes = list(CONTEXT.speaker_modes)
    decoder, decoders = decoder_capabilities(c.audio_codec, DECODERS)
    return capabilities_document(
        CapabilityFacts(
            text_tokenizer_spec=CONTEXT.text_tokenizer_spec or None,
            text_normalization_enabled=ASSETS.normalizer() is not None,
            accents=CONTEXT.accent_names,
            speaker_modes=modes,
            speaker_embedding_dim=c.speaker_embedding_dim,
            speaker_embedding_codec=c.speaker_embedding_codec,
            n_codebooks=c.n_codebooks,
            codebook_sizes=list(c.codebook_sizes),
            audio_codec=c.audio_codec,
            max_tokens=CONTEXT.max_frames,
            prefix_stream=prefix_stream_supported(c.audio_codec),
            silence_hop=dual_codec_hop(c.audio_codec),
            tag_cfg_available=True,
            tag_cfg_trained=trained_on_tags(resolve_dir_or_local(SPACE_LM)),
            decoder=decoder,
            decoders=decoders,
            condition_tokens=None
            if c.condition_count == 0
            else condition_spec(
                names=c.condition_names,
                count=c.condition_count,
                bucket_counts=c.condition_bucket_counts,
                edges=c.condition_edges,
                flags=getattr(c, "condition_flags", None),
                units=c.condition_units,
            ),
            speaker_audio_upload=bool(modes),
            speaker_cfg_available=bool(modes),
            laci_languages=CONTEXT.laci_languages,
            plugins=[],
            default_voices_enabled=True,
            edit_model=EDIT_MODEL,
            edit_available=True,
            edit_transcribe=True,
            lexicon=LEXICON,
            projects={"enabled": False},
        )
    )


CAPABILITIES = capabilities()
CONTROLS = controls_document(ASSETS.steer_bundle, speaker_embedding_dim=CONFIG.speaker_embedding_dim)


# --------------------------------------------------------------------------------------------------- voices


# Default voices (default_voices/, zonos3.server.voices: audio files and embeddings, ids of their relative paths) and
# session voices (POST /v2/speakers, per X-TTS-Session-ID).
DEFAULT_VOICES = scan_default_voices(ROOT / "default_voices", {})
SESSIONS: dict[str, dict[str, SessionVoice]] = {}


def serialize(voice: DefaultVoice | SessionVoice) -> dict:
    modes = voice_speaker_modes(list(CONTEXT.speaker_modes), voice.source_type)
    disabled = disabled_controls(ASSETS.steer_bundle, voice.speaker_id)
    if isinstance(voice, DefaultVoice):
        return serialize_default_voice(
            voice, modes=modes, embedding_dim=CONFIG.speaker_embedding_dim, disabled=disabled
        )
    return serialize_session_voice(
        speaker_id=voice.speaker_id,
        label=voice.label,
        source_type=voice.source_type,
        original_name=voice.original_name,
        created_at=voice.created_at,
        has_preview=bool(voice.audio_bytes),
        embedding_dim=CONFIG.speaker_embedding_dim,
        modes=modes,
        disabled=disabled,
    )


def find_voice(session_id: str | None, speaker_id: str) -> DefaultVoice | SessionVoice:
    voice = DEFAULT_VOICES.get(speaker_id)
    if voice is not None:
        return voice
    voice = SESSIONS.get(require_session_id(session_id), {}).get(speaker_id)
    if voice is None:
        raise ValueError(f"Unknown cached speaker id '{speaker_id}'.")
    return voice


def voice_audio(voice: DefaultVoice | SessionVoice) -> bytes:
    """An audio voice's clip as uploaded / on disk."""
    return read_voice_file(voice) if isinstance(voice, DefaultVoice) else voice.audio_bytes


def embed_audio(audio: bytes) -> torch.Tensor:
    """The MECT speaker embedding of a clip, on this process's CPU (no GPU quota spent on it)."""
    req = P.Request(speaker_audio_base64=base64.b64encode(audio).decode("ascii"), speaker_embedding_base64=None)
    return _speaker_embedding(LM, req, ASSETS, CPU)


async def voice_embedding(voice: DefaultVoice | SessionVoice) -> torch.Tensor:
    if voice.embedding is None:
        if voice.source_type == "embedding":  # a default .npy/.npz voice (session ones carry theirs)
            voice.embedding = load_embedding_vector(
                read_voice_file(voice), expected_dim=CONFIG.speaker_embedding_dim, file_name=voice.original_name
            )
        else:
            voice.embedding = await asyncio.to_thread(embed_audio, voice_audio(voice))
    return voice.embedding.detach().to(torch.float32).clone()


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
        raise ValueError(
            "Provide speaker conditioning via upload, one cached speaker id, or two cached ids for blending."
        )
    if blended:
        if not req.speaker_blend_embedding_id_a or not req.speaker_blend_embedding_id_b:
            raise ValueError("Provide both cached speaker ids to blend between two speakers.")
        a = await voice_embedding(find_voice(session_id, req.speaker_blend_embedding_id_a))
        b = await voice_embedding(find_voice(session_id, req.speaker_blend_embedding_id_b))
        return slerp(a, b, 0.5 if req.speaker_blend_t is None else float(req.speaker_blend_t))
    if cached:
        return await voice_embedding(find_voice(session_id, req.speaker_embedding_id))
    if req.speaker_embedding_base64:
        return load_embedding_vector(
            decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64"),
            expected_dim=CONFIG.speaker_embedding_dim,
            file_name=req.speaker_embedding_name,
        )
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
    ids = {
        "speaker_embedding_id": None,
        "speaker_blend_embedding_id_a": None,
        "speaker_blend_embedding_id_b": None,
        "speaker_blend_t": None,
    }
    if mode == "embedding":
        vector = await resolve_embedding(session_id, req)
        if vector is not None:
            body.update(
                ids,
                speaker_audio_base64=None,
                speaker_audio_name=None,
                speaker_embedding_base64=npy_base64(vector),
                speaker_embedding_name=None,
            )
        return body
    if req.speaker_embedding_base64 or req.speaker_blend_embedding_id_a or req.speaker_blend_embedding_id_b:
        raise ValueError(
            "This checkpoint conditions on raw reference audio and does not accept speaker embedding "
            "files or embedding blends."
        )
    if req.speaker_embedding_id:
        if req.speaker_audio_base64 or req.speaker_reference_codes is not None:
            raise ValueError("Provide either reference audio or codes, or a cached speaker id, not both.")
        voice = find_voice(session_id, req.speaker_embedding_id)
        if voice.source_type != "audio":
            raise ValueError(f"Speaker '{voice.label}' is an embedding; speaker_mode=reference needs reference audio.")
        body.update(ids)
        codes = voice_sidecar(voice.path, ".codes.npy") if isinstance(voice, DefaultVoice) else None
        if codes is not None:  # a default voice's shipped reference prefix codes
            body["speaker_reference_codes"] = np.load(codes).tolist()
        else:
            body.update(
                speaker_audio_base64=base64.b64encode(voice_audio(voice)).decode("ascii"),
                speaker_audio_name=voice.original_name,
            )
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

        _WORKERS["asr"] = Qwen3ASRModel.from_pretrained(
            resolve_dir_or_local("hf://Qwen/Qwen3-ASR-1.7B"),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=32,
            max_new_tokens=1024,
        )
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
        layout = EditLayout(
            torch.as_tensor(args["codes"], dtype=torch.int64),
            args["spans"],
            [torch.as_tensor(f, dtype=torch.int32) for f in args["fills"]],
        )
        embedding = args["embedding"]
        rows, eos_frame = edit_region(
            LM,
            args["body"],
            assets=ASSETS,
            layout=layout,
            min_frames=args["min_frames"],
            speaker_embedding=None if embedding is None else embedding.to(DEVICE),
            hidden_delta=args["hidden_delta"],
            accents=EDIT_LORA,
            device=DEVICE,
        )
        n_cb = CONFIG.n_codebooks
        # the finishing row is not part of the generated span (it only completes the delayed codebooks' flush)
        out = torch.stack(rows[:-1])[:, :n_cb].numpy() if len(rows) > 1 else np.zeros((0, n_cb), dtype=np.int32)
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
# ZeroGPU caps how long one call may hold a GPU, per visitor: one without a quota token (not signed in on huggingface.co,
# or the Space reached outside it) gets 60 s as ZeroGPU counts them, 40 of ours on its RTX PRO 6000 nodes (which ask for
# 1.5x a call's duration). A session refused as too long reopens at this length; such a session is not renewed.
CAPPED_SESSION_SECONDS = 40
_FORK = multiprocessing.get_context("fork")  # spaces forks its GPU workers, which inherit both pipes
TAKE_RX, TAKE_TX = _FORK.Pipe(duplex=False)  # (take id, request body) or None (close), main process -> session
RESULT_RX, RESULT_TX = _FORK.Pipe(duplex=False)  # (take id, kind, payload), session -> main process


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
            synthesize(
                LM,
                body,
                assets=ASSETS,
                device=DEVICE,
                on_plan=lambda plan: RESULT_TX.send((job_id, "start", (plan.seed, plan.follow_tokens is not None))),
                on_audio=lambda pcm: RESULT_TX.send((job_id, "audio", pcm)),
            )
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
            if self.open() and self.seconds >= SESSION_SECONDS and self.left() < RENEW_SECONDS:
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
        seconds = max(SESSION_SECONDS, need + MARGIN_SECONDS + 1)
        # a session that idled out as the take was sent, or that ZeroGPU refused as too long for this visitor, left it in
        # the pipe for the next one
        for _ in range(3):
            if not self.open():
                self.start(take.request, seconds)
            if sender is None:
                sender = asyncio.create_task(asyncio.to_thread(TAKE_TX.send, (self.ids, take.body)))
            done = await self.pump(self.ids, take.out)
            if done is not None:
                await sender
                self.last = time.monotonic()
                if not done:
                    take.out.put_nowait(("error", "the GPU session ended mid-take"))
                return
            if (exc := self.session.exception()) is not None:
                if seconds > CAPPED_SESSION_SECONDS and isinstance(exc, gr.Error) and exc.title == "ZeroGPU illegal duration":
                    seconds = CAPPED_SESSION_SECONDS
                    continue
                await self.drain(sender)
                raise exc
        await self.drain(sender)
        raise RuntimeError("the GPU session ended without running the take")

    def start(self, request: Request, seconds: int) -> None:
        """A session on behalf of ``request``'s visitor (ZeroGPU reads the X-IP-Token from Gradio's request context)."""
        context = contextvars.Context()
        context.run(LocalContext.request.set, gr.Request(request=request))
        self.session = asyncio.create_task(asyncio.to_thread(gpu_session, seconds), context=context)
        self.session.add_done_callback(
            lambda s: print(f"[gpu] session closed: {s.exception() or s.result()} take(s)", flush=True)
        )
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
            resolved = resolve_controls(
                ASSETS.steer_bundle,
                job.controls,
                None,
                text=job.controls_text,
                speaker_embedding=embedding,
                sources=CONTEXT.steering_sources,
                hidden_size=CONTEXT.hidden_size,
            )
            embedding, delta = resolved.speaker_embedding, resolved.hidden_delta
        request = {
            "text": job.text,
            "language": job.language,
            "text_normalization": job.text_normalization,
            **job.sampling,
            "seed": job.seed,
            "max_tokens": job.max_tokens,
            "speaker_mode": "embedding",
            "condition_buckets": job.condition_buckets,
            "condition_values": job.condition_values,
            "stream": False,
        }
        return await edit_call(
            EDIT_REQUEST.get()[0],
            "fill",
            body=request,
            codes=np.asarray(job.window_codes),
            spans=job.spans,
            fills=[np.asarray(f) for f in job.fills],
            min_frames=int(job.min_frames),
            embedding=embedding,
            hidden_delta=delta,
        )

    async def decode(self, codes: np.ndarray, decoder: str | None) -> np.ndarray:
        # the edit route checked `decoder`: the default (DECODERS) or None
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

    async def run(rt):
        resolve_decoder(req.decoder, DECODERS)  # 400 for an id this Space does not serve
        return await rt.edit(req)

    return await edit_route(request, run)


@app.get("/v2/capabilities")
async def get_capabilities():
    return CAPABILITIES


@app.post("/v2/phonemize")
async def phonemize(req: PhonemizeRequest):
    """Gold-lexicon IPA for the words of a text (espeak-ng guesses for the words it lacks); a pronunciation with a symbol
    outside the phoneme tokens is "unsupported"."""
    try:
        return await asyncio.to_thread(
            LEXICON.phonemize_for, req.text, req.language, CONTEXT.text_tokenizer_spec, guess=req.guess
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v2/controls")
async def get_controls():
    return CONTROLS


@app.get("/v2/speakers")
async def list_speakers(x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
    session = SESSIONS.get(normalize_session_id(x_tts_session_id) or "", {})
    voices = sorted(DEFAULT_VOICES.values(), key=lambda v: v.label.lower())
    voices += sorted(session.values(), key=lambda v: v.created_at, reverse=True)
    return {"speakers": [serialize(v) for v in voices]}


@app.post("/v2/speakers")
async def create_speaker(
    req: SpeakerCreateRequest, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")
):
    try:
        session_id = require_session_id(x_tts_session_id)
        if req.speaker_audio_base64 and req.speaker_embedding_base64:
            raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")
        if req.speaker_embedding_base64:
            name = req.speaker_embedding_name
            embedding = load_embedding_vector(
                decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64"),
                expected_dim=CONFIG.speaker_embedding_dim,
                file_name=name,
            )
            voice = SessionVoice(
                new_speaker_id(),
                speaker_label(req.label, name, fallback="Speaker embedding"),
                "embedding",
                time.time(),
                name,
                embedding=embedding,
            )
        elif req.speaker_audio_base64:
            name = req.speaker_audio_name
            audio = decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64")
            voice = SessionVoice(
                new_speaker_id(),
                speaker_label(req.label, name, fallback="Speaker audio"),
                "audio",
                time.time(),
                name,
                audio_bytes=audio,
                preview_wav=await asyncio.to_thread(_transcode_audio_bytes_to_wav, audio),
            )
            await voice_embedding(voice)
        else:
            raise ValueError("Provide an audio file or saved embedding file to cache.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    SESSIONS.setdefault(session_id, {})[voice.speaker_id] = voice
    return serialize(voice)


@app.get("/v2/speakers/{speaker_id}/preview")
async def preview(speaker_id: str, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
    try:
        voice = find_voice(normalize_session_id(x_tts_session_id), speaker_id)
        if voice.source_type != "audio":
            raise HTTPException(status_code=404, detail="This voice has no preview audio.")
        if isinstance(voice, DefaultVoice):
            wav = (await asyncio.to_thread(voice_preview, voice))[0]
        else:
            wav = voice.preview_wav
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=wav, media_type="audio/wav")


@app.get("/v2/speakers/{speaker_id}/embedding")
async def embedding(speaker_id: str, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
    try:
        vector = await voice_embedding(find_voice(normalize_session_id(x_tts_session_id), speaker_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"id": speaker_id, "embedding": [round(float(x), 6) for x in vector.reshape(-1)]}


@app.post("/v2/generate")
async def generate(
    req: GenerateRequest,
    request: Request,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """The take on the GPU session (GpuSessions): float32 PCM streamed as it is decoded, or buffered (stream=false, and
    16-bit WAV)."""
    try:
        check_format(bool(req.stream), req.format)
        decoder = resolve_decoder(req.decoder, DECODERS)
        body = await prepare(req.model_dump(), normalize_session_id(x_tts_session_id))
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
    headers = speech_headers(
        fmt=req.format,
        seed=seed,
        control_spans=bool(req.control_spans),
        follows_reading=bool(reading),
        decoder=decoder,
    )
    if req.stream:  # a failure mid-take ends the stream: the status line is already sent
        return StreamingResponse(
            (pcm.astype(np.float32).tobytes() async for pcm in audio_of(take)), media_type="audio/pcm", headers=headers
        )
    pcm = [chunk async for chunk in audio_of(take)]
    kind, payload = take.last
    if kind != "end":
        raise HTTPException(status_code=400 if kind == "invalid" else 500, detail=payload)
    pcm = np.concatenate(pcm) if pcm else np.zeros(0, dtype=np.float32)
    if req.format == "wav":
        return Response(content=pcm16_wav_bytes(pcm, SAMPLE_RATE), media_type="audio/wav", headers=headers)
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
        response.headers.append(
            "Set-Cookie", f"spaces-jwt={sign}; Path=/; Secure; HttpOnly; SameSite=None; Partitioned"
        )
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


def studio_file(path: str) -> FileResponse:
    """A file of the studio build with the servers' cache rule (zonos3.server.studio)."""
    root = STUDIO.resolve()
    target = resolve_studio_file(root, path)
    if target is None:
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(target, headers={"Cache-Control": studio_cache_control(target.relative_to(root).as_posix())})


@app.api_route("/assets/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
async def asset(path: str):
    return studio_file(f"assets/{path}")


# The studio is an installable PWA: its manifest, icons, service worker and theme script sit at the root.
@app.api_route("/icons/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
async def icon(path: str):
    return studio_file(f"icons/{path}")


@app.api_route("/manifest.webmanifest", methods=["GET", "HEAD"], include_in_schema=False)
async def manifest():
    return studio_file("manifest.webmanifest")


@app.api_route("/sw.js", methods=["GET", "HEAD"], include_in_schema=False)
async def service_worker():
    return studio_file("sw.js")


@app.api_route("/theme-boot.js", methods=["GET", "HEAD"], include_in_schema=False)
async def theme_boot():
    return studio_file("theme-boot.js")


app.launch(ssr_mode=False)
