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

import numpy as np
import soundfile as sf
from zonos3.server.host import host_facts, laci_slots, rate_scale
from zonos3.server.plugins import HostFacts, LaciTakeRequest, TakeResult
from zonos3.tts.edit_align import asr_device

from zonos3_mlx.engine import LaciTake
from zonos3_mlx.server import capabilities as caps
from zonos3_mlx.server import speakers
from zonos3_mlx.server.generate import JobRun, plan_generate
from zonos3_mlx.server.state import ServerState


class MlxHost:
    """zonos3.server.plugins.Host over a ServerState."""

    def __init__(self, state: ServerState):
        self.state = state
        self.asr = state.asr
        self.aligner = state.aligner

    def facts(self) -> HostFacts:
        state = self.state
        return host_facts(
            model_path=state.options.model_path,
            condition_tokens=caps.condition_tokens(state),
            speaker_modes=state.engine.speaker_modes,
            bundle=state.engine.steer_bundle,
            laci_languages=state.engine.laci_languages,
            text_tokenizer_spec=state.config.text_tokenizer_spec,
            worker_device=asr_device(),
        )

    async def synthesize_laci(self, takes: Sequence[LaciTakeRequest], session_id: str | None) -> list[TakeResult]:
        """Every take through the scheduler, at most twice --batch in flight so studio requests interleave. All
        requests are planned before the first is queued (ValueError for a bad voice / control / text)."""
        state = self.state
        if state.laci_slots is None:
            state.laci_slots = laci_slots(state.options.batch)
        gain = np.float32(state.engine.output_gain(None))
        engine = state.engine
        plans = []
        for take in takes:
            plan = await plan_generate(
                state, take.request, session_id, laci_take=LaciTake(float(take.target_frames), bool(take.adjust_rate))
            )
            scale = rate_scale(
                take,
                engine.steer_bundle,
                text=plan.text,
                base_delta=plan.hidden_delta,
                sources=engine.steering_sources,
                hidden_size=engine.hidden_size,
            )
            if scale is not None:
                plan.laci_take = LaciTake(
                    float(take.target_frames),
                    bool(take.adjust_rate),
                    scale.edges,
                    scale.first,
                    scale.bucket,
                    scale.deltas,
                )
                plan.hidden_delta = scale.delta
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
