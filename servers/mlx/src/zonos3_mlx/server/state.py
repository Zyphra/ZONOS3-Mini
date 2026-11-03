"""The MLX server's state: the engine, the scheduler, the server options and the per-process caches."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from zonos3.tts.edit_align import AsrWorker, FuseAlignWorker, fusealign_installed

from zonos3_mlx.engine import Engine
from zonos3_mlx.scheduler import Scheduler


def asr_device() -> str | None:
    """The device of the Qwen3-ASR and dubbing workers: ``ZONOS3_ASR_DEVICE``, else ``mps`` on macOS (else the
    workers' default)."""
    return os.environ.get("ZONOS3_ASR_DEVICE") or ("mps" if sys.platform == "darwin" else None)


@dataclass
class ServerOptions:
    model_path: str
    batch: int = 4
    max_frames: int = 2000
    stream_decode_default: str = "prefix"  # "prefix" | "whole"
    default_voices_dir: Path | None = None
    studio_dir: Path | None = None
    phoneme_lexicon: str | None = None
    text_normalization: bool = False
    prefix: tuple[int, int, int] = (15, 3, 450)  # chunk, hold-back, context frames
    # whether the FuseAlign worker's interpreter has fusealign_hub (speech editing and dubbing need it)
    fusealign: bool = field(default_factory=fusealign_installed)


@dataclass
class ServerState:
    engine: Engine
    scheduler: Scheduler
    options: ServerOptions
    projects: object | None = None  # zonos3.server.projects.ProjectStore
    lexicon: object | None = None  # zonos3.tts.phonemes.PhonemeLexicon
    # speakers: session voices and default-voice entries (zonos3_mlx.server.speakers)
    sessions: dict = field(default_factory=dict)
    sessions_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    default_voices: dict = field(default_factory=dict)
    default_lock: threading.Lock = field(default_factory=threading.Lock)
    # reference codes of encoded clips, by sha1 of the audio bytes (LRU of 64)
    reference_codes: OrderedDict = field(default_factory=OrderedDict)
    # speech editing and dubbing: the server's ASR worker (zonos3/workers/asr.py, Qwen3-ASR in .venv-asr; every speech
    # recognition of both) and FuseAlign worker (zonos3/workers/fusealign.py; every forced alignment of both); each
    # starts on its first call
    asr: AsrWorker = field(default_factory=lambda: AsrWorker(device=asr_device()))
    aligner: FuseAlignWorker = field(default_factory=FuseAlignWorker)
    dub_runtime: object | None = None
    dub_slots: asyncio.Semaphore | None = None

    @property
    def config(self):
        return self.engine.config

    async def run(self, fn, *args, **kwargs):
        """Run ``fn`` on the scheduler's worker thread (every MLX call) and await its result."""
        return await asyncio.wrap_future(self.scheduler.submit(fn, *args, **kwargs))

    def close(self) -> None:
        self.scheduler.stop()
        self.asr.close()
        if self.dub_runtime is not None:
            self.dub_runtime.close()
        self.aligner.close()
        if self.engine.normalizer is not None:
            self.engine.normalizer.close()
