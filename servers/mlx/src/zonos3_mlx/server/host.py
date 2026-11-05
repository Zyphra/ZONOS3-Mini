"""The MLX server's plugin host (zonos3.server.plugins.Host): its facts, LACI takes and reference encodes.

Every take is a /v2/generate request with laci=true and the take's LACI duration target, so takes use studio voices,
steering controls and pronunciation groups; references are encoded with the MLX reference encoder. Plugins share the
server's ASR worker (Qwen3-ASR) and FuseAlign worker with speech editing; ``worker_device`` (ZONOS3_ASR_DEVICE,
default mps on macOS) is the device of their own helper workers.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
from zonos3.server.plugins import HostFacts, LaciTakeRequest, TakeResult
from zonos3.tts.directions import RAW_SPACE
from zonos3.tts.steering import resolve_controls

from zonos3_mlx.engine import LaciTake
from zonos3_mlx.server import capabilities as caps
from zonos3_mlx.server import speakers
from zonos3_mlx.server.generate import JobRun, plan_generate
from zonos3_mlx.server.state import ServerState, asr_device


def rate_scale(state: ServerState, take: LaciTakeRequest, text: str, base_delta) -> LaciTake | None:
    """The take on its extended speaking-rate scale (``take.rate_*``; sgl _laci_take_rate_scale): its bucket on the
    scale and the steering delta of every bucket (a steered bucket adds its tempo dose to the take's own delta)."""
    if take.rate_edges is None:
        return None
    bundle, engine = state.engine.steer_bundle, state.engine
    if take.rate_doses and (bundle is None or bundle.rate_extension is None):
        raise ValueError("rate_doses need a steering bundle with a rate_extension")

    def delta(b: int):
        if b not in take.rate_doses:
            return base_delta
        tempo = resolve_controls(bundle, {bundle.rate_extension.direction: take.rate_doses[b]}, None, text=text,
                                 speaker_embedding=None, sources=engine.steering_sources,
                                 hidden_size=engine.hidden_size).hidden_delta
        return tempo if base_delta is None else base_delta + tempo

    first = int(take.rate_first)
    last = first + len(take.rate_edges)
    return LaciTake(float(take.target_frames), bool(take.adjust_rate), tuple(float(e) for e in take.rate_edges),
                    first, int(take.rate_bucket), [delta(b) for b in range(first, last + 1)])


class MlxHost:
    """zonos3.server.plugins.Host over a ServerState."""

    def __init__(self, state: ServerState):
        self.state = state
        self.asr = state.asr
        self.aligner = state.aligner

    def facts(self) -> HostFacts:
        state = self.state
        bundle = state.engine.steer_bundle
        names = state.config.condition_names
        return HostFacts(
            model_name=Path(state.options.model_path).name,
            rate_edges=tuple(caps.speaking_rate_edges(state)),
            rate_buckets=(state.config.condition_bucket_counts[names.index("speaking_rate")]
                          if "speaking_rate" in names else 0),
            laci_languages=frozenset(state.engine.laci_languages),
            speaker_modes=tuple(m for m in ("reference", "embedding") if m in state.engine.speaker_modes),
            raw_controls=frozenset(did for did, d in (bundle.directions.items() if bundle is not None else ())
                                   if d.space == RAW_SPACE),
            phoneme_tokens=caps.phonemes(state)["phoneme_tokens"] is not None,
            rate_extension=None if bundle is None else bundle.rate_extension,
            worker_device=asr_device())

    async def synthesize_laci(self, takes: Sequence[LaciTakeRequest], session_id: str | None) -> list[TakeResult]:
        """Every take through the scheduler, at most twice --batch in flight so studio requests interleave. All
        requests are planned before the first is queued (ValueError for a bad voice / control / text)."""
        state = self.state
        if state.laci_slots is None:
            state.laci_slots = asyncio.Semaphore(max(1, 2 * state.options.batch))
        gain = np.float32(state.engine.output_gain(None))
        plans = []
        for take in takes:
            plan = await plan_generate(state, take.request, session_id,
                                       laci_take=LaciTake(float(take.target_frames), bool(take.adjust_rate)))
            scaled = rate_scale(state, take, plan.text, plan.hidden_delta)
            if scaled is not None:
                plan.laci_take = scaled
                plan.hidden_delta = scaled.rate_deltas[scaled.rate_bucket - scaled.rate_first]
            plans.append(plan)

        async def run(plan):
            async with state.laci_slots:
                job = JobRun(state, plan, "whole").start()
                try:
                    data = await job.wait()
                finally:
                    job.abort()
            if job.record is None:
                raise RuntimeError("LACI take: the scheduler sent no LACI record")
            audio = np.frombuffer(data, dtype=np.float32) * gain if data else None
            return audio, job.record

        return list(await asyncio.gather(*(run(plan) for plan in plans)))

    async def encode_reference(self, audio: np.ndarray, sample_rate: int, max_seconds: float) -> list[list[int]]:
        """Raw reference prefix codes of a mono float32 clip."""
        buf = io.BytesIO()
        sf.write(buf, audio, int(sample_rate), format="WAV", subtype="FLOAT")
        codes = await speakers.encode_reference(self.state, buf.getvalue(), max_seconds=max_seconds)
        return codes.long().tolist()
