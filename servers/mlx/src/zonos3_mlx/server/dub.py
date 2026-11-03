"""Dubbing: the MLX parts of ``zonos3.dub.runtime.DubRuntime`` (served by ``zonos3.server.dub.make_router``).

Every take is a /v2/generate request with laci=true and the take's LACI duration target, so dub takes use studio
voices, steering controls and pronunciation groups; the source voice's prompt clip is encoded with the MLX reference
encoder. Transcripts come from the server's ASR worker (Qwen3-ASR) and word timings from its FuseAlign worker (both
shared with speech editing); the dubbing worker (SimAlign, AV-sync) runs in .venv-asr with the ASR worker
(ZONOS3_ASR_PYTHON; ZONOS3_ASR_DEVICE, default mps on macOS).
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import numpy as np
import soundfile as sf
from zonos3.server.schemas import GenerateRequest
from zonos3.tts.directions import RAW_SPACE
from zonos3.tts.steering import resolve_controls

from zonos3_mlx.engine import LaciTake
from zonos3_mlx.server import capabilities as caps
from zonos3_mlx.server import speakers
from zonos3_mlx.server.generate import JobRun, plan_generate
from zonos3_mlx.server.state import ServerState, asr_device

UNAVAILABLE = ("Dubbing is not enabled on this server: it needs --laci-heads and a checkpoint with a speaking-rate "
               "condition row")


def runtime(state: ServerState):
    """The server's DubRuntime (created on first use; its alignment worker starts with the first call that needs it)."""
    if state.dub_runtime is None:
        from zonos3.dub.runtime import DubModel, DubRuntime, WorkerSettings, rate_scales

        modes = state.engine.speaker_modes
        bundle = state.engine.steer_bundle
        settings = WorkerSettings.from_env()
        if asr_device() is not None:
            settings = WorkerSettings(python=settings.python, avsync_home=settings.avsync_home, device=asr_device())
        model = DubModel(
            name=Path(state.options.model_path).name, rate_edges=tuple(caps.dub_rate_edges(state)),
            n_buckets=(state.config.condition_bucket_counts[state.config.condition_names.index("speaking_rate")]
                       if "speaking_rate" in state.config.condition_names else 0),
            laci_languages=state.engine.laci_languages, studio_voices="embedding" in modes,
            source_voice="reference" in modes,
            raw_controls=frozenset(did for did, d in (bundle.directions.items() if bundle is not None else ())
                                   if d.space == RAW_SPACE),
            phoneme_groups=caps.phonemes(state)["phoneme_tokens"] is not None,
            rate_scales=rate_scales(None if bundle is None else bundle.rate_extension, caps.dub_rate_edges(state)))

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
    bundle, engine = state.engine.steer_bundle, state.engine

    def delta(b: int):
        if b not in scale.doses:
            return base_delta
        tempo = resolve_controls(bundle, {bundle.rate_extension.direction: scale.doses[b]}, None, text=text,
                                 speaker_embedding=None, sources=engine.steering_sources,
                                 hidden_size=engine.hidden_size).hidden_delta
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
