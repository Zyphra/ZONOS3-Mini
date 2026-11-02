"""The MLX server's state: the engine, the scheduler, the server options and the per-process caches."""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from zonos3_mlx.engine import Engine
from zonos3_mlx.scheduler import Scheduler


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
    # speech editing and dubbing
    mms_workers: OrderedDict = field(default_factory=OrderedDict)
    mms_lock: threading.Lock = field(default_factory=threading.Lock)
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
        for worker in self.mms_workers.values():
            worker.close()
        if self.dub_runtime is not None:
            self.dub_runtime.close()
        if self.engine.normalizer is not None:
            self.engine.normalizer.close()
