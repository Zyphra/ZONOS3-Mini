"""Dubbing (/v2/dub/*): ``zonos3.dub.runtime.DubRuntime`` over this server's scheduler (the sgl routes).

Every take is a /v2/generate request with laci=true and the take's LACI duration target, so dub takes use studio
voices, steering controls and pronunciation groups; the source voice's prompt clip is encoded with the MLX reference
encoder. Transcripts come from the server's ASR worker (Qwen3-ASR) and word timings from its FuseAlign worker (both
shared with speech editing); the dubbing worker (SimAlign, AV-sync) runs in .venv-asr with the ASR worker
(ZONOS3_ASR_PYTHON; ZONOS3_ASR_DEVICE, default mps on macOS).
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import Response
from zonos3.server.schemas import (
    DubAnalyzeRequest,
    DubGenerateRequest,
    DubRegenerateRequest,
    DubTranscribeRequest,
    GenerateRequest,
)
from zonos3.tts.steering import RAW_SPACE, resolve_controls

from zonos3_mlx.engine import LaciTake
from zonos3_mlx.server import capabilities as caps
from zonos3_mlx.server import speakers
from zonos3_mlx.server.generate import JobRun, plan_generate
from zonos3_mlx.server.state import ServerState, asr_device

logger = logging.getLogger("zonos3_mlx.server")
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def rate_scales(state: ServerState) -> dict:
    """Voice mode ("studio" / "source") -> TTS language -> its speaking-rate scale with the tempo-steered buckets of the
    steering bundle's rate_extension (source voices only on its reference_sides); empty without one, or when it was
    measured against other row edges (sgl _dub_rate_scales)."""
    from zonos3.dub.plan import RateScale

    bundle = state.engine.bundle
    ext = None if bundle is None else bundle.rate_extension
    if ext is None:
        return {}
    edges = tuple(caps.dub_rate_edges(state))
    if len(ext.row_edges) != len(edges) or not np.allclose(ext.row_edges, edges, rtol=0, atol=1e-6):
        logger.warning("steering bundle rate_extension was measured on speaking-rate edges %s, the checkpoint has %s: "
                       "dubs plan in the row buckets only", list(ext.row_edges), list(edges))
        return {}
    source = {lang: {b: d for b, d in doses.items() if ("slower" if b < 0 else "faster") in ext.reference_sides}
              for lang, doses in ext.doses.items()}
    return {mode: {lang: RateScale.extended(edges, ext.below, ext.above, doses) for lang, doses in by_lang.items()
                   if doses}
            for mode, by_lang in (("studio", ext.doses), ("source", source))}


def runtime(state: ServerState):
    """The server's DubRuntime (created on first use; its alignment worker starts with the first call that needs it)."""
    if state.dub_runtime is None:
        from zonos3.dub.runtime import DubModel, DubRuntime, WorkerSettings

        modes = speakers.checkpoint_modes(state)
        bundle = state.engine.bundle
        settings = WorkerSettings.from_env()
        if asr_device() is not None:
            settings = WorkerSettings(python=settings.python, avsync_home=settings.avsync_home, device=asr_device())
        model = DubModel(
            name=Path(state.options.model_path).name, rate_edges=tuple(caps.dub_rate_edges(state)),
            n_buckets=state.config.condition_n_buckets if state.config.condition_count else 0,
            laci_languages=frozenset(state.engine.laci_heads), studio_voices="embedding" in modes,
            source_voice="reference" in modes,
            raw_controls=frozenset(did for did, d in (bundle.directions.items() if bundle is not None else ())
                                   if d.space == RAW_SPACE),
            phoneme_groups=caps.phonemes(state)["phoneme_tokens"] is not None,
            rate_scales=rate_scales(state))

        async def synthesize(jobs, session_id):
            return await synthesize_takes(state, jobs, session_id)

        async def encode_voice(audio, sample_rate):
            return await encode_voice_prompt(state, audio, sample_rate)

        state.dub_runtime = DubRuntime(model, synthesize, encode_voice, state.asr, state.aligner, settings)
    return state.dub_runtime


def take_request(job: dict) -> GenerateRequest:
    """The /v2/generate request of one dub take (sgl _dub_take_request): the release sampling defaults, the take's
    speaking-rate bucket in the condition rows (a tempo-steered bucket leaves the rate row unknown), LACI, and its
    voice: the reference clone (raw reference prefix codes) or a studio voice with the take's steering controls."""
    voice, bucket = job["voice"], int(job["bucket"])
    scale = job.get("rate_scale")
    row = None if scale is not None and bucket in scale.doses else bucket
    fields = {"text": job["text"], "language": job["language"], "seed": int(job["seed"]), "max_tokens": int(job["max_tokens"]),
                  "stream": False, "laci": True, "speaker_cfg_scale": 1.0, "condition_buckets": {"speaking_rate": row},
                  "controls": job["controls"]}
    if voice["mode"] == "source":
        fields.update(speaker_mode="reference", speaker_reference_codes=job["ref_codes"])
    else:
        fields.update(speaker_mode="embedding", speaker_embedding_id=voice["speaker_embedding_id"])
    return GenerateRequest(**fields)


def take_rate_steering(state: ServerState, job: dict, text: str, base_delta) -> LaciTake | None:
    """The take's extended speaking-rate scale (sgl _dub_take_rate_steering): its bucket on the scale and the steering
    delta of every bucket (a steered bucket adds its tempo dose to the take's own delta)."""
    scale = job.get("rate_scale")
    if scale is None:
        return None
    bundle, engine = state.engine.bundle, state.engine

    def delta(b: int):
        if b not in scale.doses:
            return base_delta
        tempo = resolve_controls(bundle, {bundle.rate_extension.direction: scale.doses[b]}, None, text=text,
                                 speaker_embedding=None, sources=engine.steering_sources(),
                                 hidden_size=state.config.dim).hidden_delta
        return tempo if base_delta is None else base_delta + tempo

    return LaciTake(float(job["target_frames"]), bool(job["adjust_rate"]), tuple(float(e) for e in scale.edges),
                    int(scale.first), int(job["bucket"]), [delta(b) for b in range(scale.first, scale.last + 1)])


async def synthesize_takes(state: ServerState, jobs: list[dict], session_id: str | None):
    """Every dub take through the scheduler, at most twice --batch in flight so studio requests interleave. All
    requests are planned before the first is queued (HTTP 400 for a bad voice / control / text)."""
    if state.dub_slots is None:
        state.dub_slots = asyncio.Semaphore(max(1, 2 * state.options.batch))
    gain = np.float32(state.engine.output_gain(None))
    plans = []
    for job in jobs:
        req = take_request(job)
        plan = await plan_generate(state, req, session_id,
                                   laci_take=LaciTake(float(job["target_frames"]), bool(job["adjust_rate"])))
        take = take_rate_steering(state, job, plan.text, plan.hidden_delta)
        if take is not None:
            plan.laci_take = take
            plan.hidden_delta = take.rate_deltas[take.rate_bucket - take.rate_first]
        plans.append(plan)

    async def run(plan):
        async with state.dub_slots:
            take = JobRun(state, plan, "whole").start()
            try:
                data = await take.wait()
            finally:
                take.abort()
        if take.record is None:
            raise RuntimeError("dub take: the scheduler sent no LACI record")
        audio = np.frombuffer(data, dtype=np.float32) * gain if data else None
        return audio, take.record

    return list(await asyncio.gather(*(run(plan) for plan in plans)))


async def encode_voice_prompt(state: ServerState, audio: np.ndarray, sample_rate: int) -> list[list[int]]:
    """Raw reference prefix codes of the source voice's prompt clip."""
    from zonos3.dub.plan import VOICE_PROMPT_MAX_S

    buf = io.BytesIO()
    sf.write(buf, audio, int(sample_rate), format="WAV", subtype="FLOAT")
    codes = await speakers.encode_reference(state, buf.getvalue(), max_seconds=VOICE_PROMPT_MAX_S)
    return codes.long().tolist()


async def upload(request: Request, name: str, directory: str) -> Path:
    """Stream a request body (a file named ``name``) into ``directory``; 413 past the dub's upload cap."""
    from zonos3.dub.plan import MAX_VIDEO_BYTES

    declared = request.headers.get("content-length")
    if declared is not None and int(declared) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")
    suffix = Path(name).suffix.lower()
    path = Path(directory) / f"input{suffix if suffix in VIDEO_SUFFIXES | {'.wav', '.flac', '.mp3', '.ogg'} else '.bin'}"
    size = 0
    with open(path, "wb") as fh:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                raise HTTPException(status_code=413, detail="upload too large")
            fh.write(chunk)
    return path


async def call(awaitable):
    """Run a DubRuntime call, mapping its errors to HTTP statuses."""
    from zonos3.dub.runtime import AlignWorkerError, UnknownId
    from zonos3.tts.edit_align import WorkerError

    try:
        return await awaitable
    except UnknownId:
        raise HTTPException(status_code=404, detail="unknown ref_id/dub_id; upload or generate again") from None
    except AlignWorkerError as exc:
        raise HTTPException(status_code=503, detail=f"alignment worker: {exc}") from exc
    except WorkerError as exc:  # the ASR / FuseAlign worker is unavailable (the message says which)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def router(state: ServerState) -> APIRouter:
    r = APIRouter()

    def ready():
        if not caps.dub_available(state):
            raise HTTPException(status_code=400, detail="Dubbing is not enabled on this server: it needs --laci-heads "
                                                        "and a checkpoint with a speaking-rate condition row")
        return runtime(state)

    @r.get("/v2/dub/capabilities")
    async def capabilities():
        """Target languages, rate buckets, lip-sync, voice modes and limits of the dub (zonos3.dub.runtime)."""
        return {**(await asyncio.to_thread(runtime(state).capabilities)), "enabled": caps.dub_available(state)}

    @r.post("/v2/dub/ref")
    async def ref(request: Request, name: str = "audio"):
        """A dub reference: raw audio bytes (any sound file), or a video whose audio track is taken."""
        rt = ready()
        with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
            path = await upload(request, name, tmp)
            return await call(asyncio.to_thread(rt.add_ref, str(path)))

    @r.post("/v2/dub/lips/{ref_id}")
    async def lips(ref_id: str, request: Request, name: str = "video"):
        """Face track + lip embeddings of the reference's video, for generate / regenerate with lipsync (AV-sync)."""
        rt = ready()

        async def run():
            rt.refs.get(ref_id)  # 404 before reading the body
            with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
                video = await upload(request, name, tmp)
                return await asyncio.to_thread(rt.prepare_lips, ref_id, str(video))

        return await call(run())

    @r.post("/v2/dub/transcribe")
    async def transcribe(req: DubTranscribeRequest):
        return await call(ready().transcribe(req.ref_id, req.ref_lang))

    @r.post("/v2/dub/analyze")
    async def analyze(req: DubAnalyzeRequest):
        return await call(ready().analyze(req.model_dump()))

    @r.post("/v2/dub/generate")
    async def generate(req: DubGenerateRequest,
                       x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        return await call(ready().generate(req.model_dump(), x_tts_session_id))

    @r.post("/v2/dub/regenerate")
    async def regenerate(req: DubRegenerateRequest,
                         x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        return await call(ready().regenerate(req.model_dump(), x_tts_session_id))

    @r.post("/v2/dub/mux/{dub_id}")
    async def mux(dub_id: str, request: Request, name: str = "video"):
        """The video with the dub as its audio track (raw video bytes in, MP4 out)."""
        rt = ready()
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "video"

        async def run():
            rt.dubs.get(dub_id)  # 404 before reading the body
            with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
                video, out = await upload(request, name, tmp), Path(tmp) / "out.mp4"
                await asyncio.to_thread(rt.mux, dub_id, str(video), str(out))
                return out.read_bytes()

        data = await call(run())
        return Response(data, media_type="video/mp4",
                        headers={"Content-Disposition": f'attachment; filename="{stem}_dub.mp4"'})

    return r
