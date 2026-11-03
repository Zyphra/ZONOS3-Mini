"""Speech editing (/v2/edit/transcribe, /v2/edit/align, /v2/edit): ``zonos3.tts.edit_runtime.EditRuntime`` over the
MLX engine.

The runtime (shared with the sgl and ggml servers) does the ASR / FuseAlign work, the word diff, the windows and the
splice; this module is its ``EditEngine``: uploads are decoded with ``decode_pcm``, the clip is encoded with the MLX
reference encoder, every region is one span-end-terminated job on the scheduler over the training edit layout
(``plan_request(edit=...)``), and the spliced codes are decoded once.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import torch
from zonos3.reference.synth import Sampling, resolve_condition_buckets, validate_phoneme_groups
from zonos3.tts.edit_runtime import EditRuntime, FillJob
from zonos3.tts.steering import RAW_SPACE, check_controls, resolve_controls

from zonos3_mlx.audio_io import decode_pcm
from zonos3_mlx.engine import EditLayout, ResolvedSpeaker, plan_request
from zonos3_mlx.server import capabilities as caps
from zonos3_mlx.server import speakers
from zonos3_mlx.server.generate import JobRun
from zonos3_mlx.server.state import ServerState

EDIT_DECODE_MAX_SECONDS = 600  # uploads are decoded this far, so the length check sees them
SAMPLE_RATE = 44100


class MlxEditEngine:
    """The MLX server's ``zonos3.tts.edit_runtime.EditEngine``."""

    def __init__(self, state: ServerState):
        self.state = state
        self.model = caps.edit_model(state)

    async def decode_audio(self, audio: bytes) -> tuple[np.ndarray, int]:
        return await asyncio.to_thread(decode_pcm, audio, max_seconds=EDIT_DECODE_MAX_SECONDS), SAMPLE_RATE

    async def embed(self, audio: bytes) -> torch.Tensor:
        return await speakers.embed_audio(self.state, audio)

    async def encode(self, audio: bytes, max_seconds: float) -> np.ndarray:
        codes = await speakers.encode_reference(self.state, audio, max_seconds=max_seconds)
        return codes.to(torch.int64).numpy()

    async def check(self, target_text: str, doses: list[dict[str, float]] | None, condition_buckets,
                    condition_values) -> set[str]:
        c, bundle = self.state.config, self.state.engine.bundle
        validate_phoneme_groups(c.text_tokenizer_spec or None, target_text)
        raw = set()
        if doses is not None:
            if bundle is None:
                raise ValueError("This server has no steering bundle (place steer.gguf next to the backbone or pass "
                                 "--steer); `controls` are unavailable.")
            for region in doses:
                check_controls(bundle, region)
            raw = {did for region in doses for did in region if bundle.directions[did].space == RAW_SPACE}
        resolve_condition_buckets(c, condition_buckets, condition_values)  # as plan_request reads them
        return raw

    async def fill(self, job: FillJob) -> tuple[np.ndarray, int]:
        state, engine = self.state, self.state.engine
        c = state.config
        embedding, delta = job.speaker_embedding, None
        if job.controls:
            resolved = resolve_controls(engine.bundle, job.controls, None, text=job.controls_text,
                                        speaker_embedding=embedding, sources=engine.steering_sources(),
                                        hidden_size=c.dim)
            embedding, delta = resolved.speaker_embedding, resolved.hidden_delta
        request = {"text": job.text, "language": job.language, "text_normalization": job.text_normalization,
                   **job.sampling, "seed": job.seed, "max_tokens": job.max_tokens,
                   "condition_buckets": job.condition_buckets, "condition_values": job.condition_values,
                   "stream": False}
        layout = EditLayout(torch.as_tensor(job.window_codes, dtype=torch.int64), job.spans,
                            [torch.as_tensor(f, dtype=torch.int32) for f in job.fills], job.min_frames)
        speaker = ResolvedSpeaker("embedding", embedding, None)
        plan = await asyncio.to_thread(plan_request, engine, request, speaker, edit=layout, hidden_delta=delta)
        sampling = replace(Sampling.from_request(plan.request), min_frames=int(job.min_frames))
        run = JobRun(state, plan, None, sampling, stop_id=c.span_end_id).start()
        try:
            await run.wait()
        finally:
            run.abort()
        # the finishing row is not part of the generated span (it only completes the delayed codebooks' flush)
        rows = (torch.stack([torch.as_tensor(r) for r in run.rows[:-1]])[:, : c.n_codebooks].numpy()
                if len(run.rows) > 1 else np.zeros((0, c.n_codebooks), dtype=np.int32))
        return rows, -1 if run.eos_frame is None else run.eos_frame

    async def decode(self, codes: np.ndarray) -> np.ndarray:
        engine = self.state.engine
        audio = await self.state.run(engine.decoder.decode, np.asarray(codes, dtype=np.int64))
        return audio * np.float32(engine.output_gain(None))


def runtime(state: ServerState) -> EditRuntime:
    """The server's EditRuntime (created on first use, over the server's ASR and FuseAlign workers)."""
    if state.edit_runtime is None:
        state.edit_runtime = EditRuntime(MlxEditEngine(state), state.asr, state.aligner)
    return state.edit_runtime
