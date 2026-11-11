"""The MLX server's state: the engine, the scheduler, the server options and the per-process caches."""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from zonos3.tts.edit_align import AsrWorker, FuseAlignWorker, fusealign_installed
from zonos3.tts.stream import StreamSettings

from zonos3_mlx.engine import Engine
from zonos3_mlx.scheduler import Scheduler


@dataclass
class ServerOptions:
    model_path: str
    batch: int = 4
    default_voices_dir: Path | None = None
    studio_dir: Path | None = None
    phoneme_lexicon: str | None = None
    prefix: StreamSettings = field(default_factory=StreamSettings)  # ZONOS3_PREFIX_STREAM_*
    # whether the FuseAlign worker's interpreter has fusealign_hub (speech editing and plugins need it)
    fusealign: bool = field(default_factory=fusealign_installed)
    plugins: tuple[str, ...] = ()  # --plugins: zonos3_<name> packages to mount


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
    # speech editing and plugins: the server's ASR worker (zonos3/workers/asr.py, Qwen3-ASR in .venv-asr; every speech
    # recognition; optional: editing without it needs the clip's transcript) and FuseAlign worker
    # (zonos3/workers/fusealign.py; every forced alignment); each starts on its first call
    asr: AsrWorker = field(default_factory=AsrWorker)
    aligner: FuseAlignWorker = field(default_factory=FuseAlignWorker)
    edit_runtime: object | None = None  # zonos3.tts.edit_runtime.EditRuntime (zonos3_mlx.server.edit.runtime)
    laci_slots: asyncio.Semaphore | None = None  # LACI takes in flight (zonos3.server.host.laci_slots, on first use)
    plugins: list = field(default_factory=list)  # loaded zonos3.server.plugins.Plugin

    @property
    def config(self):
        return self.engine.config

    async def run(self, fn, *args, **kwargs):
        """Run ``fn`` on the scheduler's worker thread (every MLX call but the streamed decodes) and await its result."""
        return await asyncio.wrap_future(self.scheduler.submit(fn, *args, **kwargs))

    def close(self) -> None:
        self.scheduler.stop()
        self.asr.close()
        for plugin in self.plugins:
            plugin.close()
        self.aligner.close()
