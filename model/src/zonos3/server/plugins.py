"""The plugin seam of every server: ``--plugins NAME...`` imports ``zonos3_<NAME>`` and mounts the FastAPI router of
its ``plugin()`` over the server's ``Host``.

A plugin serves only ``/v2/<name>/*`` and ``/<name>``, ``/<name>/*``. It reaches the served model through ``Host``: the
server's facts (``HostFacts``), LACI takes with a duration target (``LaciTakeRequest``), the codec encode of a
reference clip and the server's shared ASR / FuseAlign workers. The sgl and mlx servers implement ``Host`` in-process;
the ggml server runs plugins in its Python host process (zonos3/workers/pyhost.py) and forwards their paths there.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

from fastapi import APIRouter, FastAPI

from zonos3.server.schemas import GenerateRequest
from zonos3.tts.steering import RateExtension

if TYPE_CHECKING:
    import numpy as np

    from zonos3.tts.edit_align import AsrWorker, FuseAlignWorker

HOST_API = 1  # bumped on any breaking change of Host / HostFacts / LaciTakeRequest / Plugin
CATCH_ALL = "/{path:path}"  # the studio's catch-all route, kept last
_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class HostFacts:
    """What a plugin may know of the served checkpoint and server."""

    model_name: str  # checkpoint dir name
    rate_edges: tuple[float, ...]  # speaking-rate condition-row edges (UTF-8 B/s); () without that row
    rate_buckets: int  # that row's bucket count (0 without)
    laci_languages: frozenset[str]  # TTS languages with LACI heads (--laci-heads)
    speaker_modes: tuple[str, ...]  # subset of ("reference", "embedding")
    raw_controls: frozenset[str]  # steering ids in RAW_SPACE (move the speaker embedding)
    phoneme_tokens: bool  # pronunciation groups are readable
    rate_extension: RateExtension | None  # zonos3.tts.steering.RateExtension of the bundle
    worker_device: str | None = None  # device for helper workers (mlx: asr_device()); None = their default

    def to_json(self) -> dict:
        ext = self.rate_extension
        if ext is not None:
            ext = {
                "direction": ext.direction,
                "row_edges": list(ext.row_edges),
                "below": list(ext.below),
                "above": list(ext.above),
                "doses": {lang: {str(b): float(d) for b, d in doses.items()} for lang, doses in ext.doses.items()},
                "reference_sides": list(ext.reference_sides),
            }
        return {
            "model_name": self.model_name,
            "rate_edges": list(self.rate_edges),
            "rate_buckets": self.rate_buckets,
            "laci_languages": sorted(self.laci_languages),
            "speaker_modes": list(self.speaker_modes),
            "raw_controls": sorted(self.raw_controls),
            "phoneme_tokens": self.phoneme_tokens,
            "rate_extension": ext,
            "worker_device": self.worker_device,
        }

    @classmethod
    def from_json(cls, d: dict) -> HostFacts:
        ext = d.get("rate_extension")
        if ext is not None:
            ext = RateExtension(
                direction=ext["direction"],
                row_edges=tuple(float(e) for e in ext["row_edges"]),
                below=tuple(float(e) for e in ext["below"]),
                above=tuple(float(e) for e in ext["above"]),
                doses={lang: {int(b): float(x) for b, x in doses.items()} for lang, doses in ext["doses"].items()},
                reference_sides=tuple(ext["reference_sides"]),
            )
        return cls(
            model_name=d["model_name"],
            rate_edges=tuple(float(e) for e in d["rate_edges"]),
            rate_buckets=int(d["rate_buckets"]),
            laci_languages=frozenset(d["laci_languages"]),
            speaker_modes=tuple(d["speaker_modes"]),
            raw_controls=frozenset(d["raw_controls"]),
            phoneme_tokens=bool(d["phoneme_tokens"]),
            rate_extension=ext,
            worker_device=d.get("worker_device"),
        )


@dataclass(frozen=True)
class LaciTakeRequest:
    """One LACI take with a duration target (zonos3.tts.laci.DurationTarget), optionally on an extended
    speaking-rate scale (zonos3.tts.laci.RateBuckets)."""

    request: GenerateRequest  # laci=True, stream=False; planned exactly as /v2/generate
    target_frames: float
    adjust_rate: bool = True
    rate_edges: tuple[float, ...] | None = None  # extended scale: None = retakes move along the row
    rate_first: int = 0
    rate_bucket: int | None = None  # the take's bucket on the scale
    # steered bucket -> dose of facts.rate_extension.direction (ValueError without it)
    rate_doses: Mapping[int, float] = field(default_factory=dict)

    def to_json(self, out: str) -> dict:
        """The take as the ggml server's ``laci_takes`` engine call carries it (WAV written to ``out``)."""
        scale = None
        if self.rate_edges is not None:
            scale = {
                "edges": [float(e) for e in self.rate_edges],
                "first": int(self.rate_first),
                "bucket": self.rate_bucket,
                "doses": {str(b): float(d) for b, d in self.rate_doses.items()},
            }
        return {
            "request": self.request.model_dump(exclude_unset=True),
            "target_frames": float(self.target_frames),
            "adjust_rate": bool(self.adjust_rate),
            "scale": scale,
            "out": out,
        }


# float32 mono at zonos3.generation.SAMPLE_RATE with the server's output gain (None: no audio), the LACI record
TakeResult = tuple["np.ndarray | None", dict]


class Host(Protocol):
    asr: AsrWorker  # zonos3.tts.edit_align: the server's shared workers
    aligner: FuseAlignWorker

    def facts(self) -> HostFacts: ...

    async def synthesize_laci(self, takes: Sequence[LaciTakeRequest], session_id: str | None) -> list[TakeResult]:
        """Plan every take first (ValueError -> 400 before any is queued), then run them on the scheduler with at most
        2x the running-request cap in flight so other requests interleave."""

    async def encode_reference(self, audio: np.ndarray, sample_rate: int, max_seconds: float) -> list[list[int]]:
        """Raw-reference prefix codes [[code]*C]*frames of a mono float32 clip (the server's codec encode)."""


class Plugin(Protocol):
    name: str  # [a-z][a-z0-9_]*; serves only /v2/<name>/* and /<name>, /<name>/*
    host_api: int  # must equal HOST_API
    keepalive_paths: tuple[str, ...]  # POST paths (prefix match, as KeepaliveMiddleware) that honour X-Keepalive

    def router(self, host: Host) -> APIRouter: ...

    def close(self) -> None: ...


def load(names: Sequence[str]) -> list[Plugin]:
    """Each name: importlib.import_module(f"zonos3_{name}").plugin(). ValueError (startup error) for a bad name,
    a missing package ("plugin 'NAME': zonos3_NAME is not installed in this interpreter"), name or host_api mismatch."""
    loaded: list[Plugin] = []
    for name in names:
        if not _NAME.fullmatch(name):
            raise ValueError(f"plugin {name!r}: a plugin name is [a-z][a-z0-9_]*")
        if any(p.name == name for p in loaded):
            raise ValueError(f"plugin {name!r} is listed twice")
        package = f"zonos3_{name}"
        try:
            module = importlib.import_module(package)
        except ModuleNotFoundError as exc:
            if exc.name != package:
                raise
            raise ValueError(f"plugin {name!r}: {package} is not installed in this interpreter") from None
        plugin = module.plugin()
        if plugin.name != name:
            raise ValueError(f"plugin {name!r}: {package}.plugin() is named {plugin.name!r}")
        if plugin.host_api != HOST_API:
            raise ValueError(f"plugin {name!r}: built for host API {plugin.host_api}, this server has {HOST_API}")
        loaded.append(plugin)
    return loaded


def _owned(path: str, name: str) -> bool:
    return any(path == root or path.startswith(root + "/") for root in (f"/v2/{name}", f"/{name}"))


def mount(app: FastAPI, loaded: Sequence[Plugin], host: Host, keepalive_paths: list[str]) -> None:
    """include_router(p.router(host)) for each plugin; ValueError if a route is outside /v2/<name> or /<name>;
    then move every route whose path is "/{path:path}" (the studio catch-all) back to the end of app.router.routes;
    keepalive_paths.extend(p.keepalive_paths)."""
    for plugin in loaded:
        router = plugin.router(host)
        stray = [route.path for route in router.routes if not _owned(route.path, plugin.name)]
        if stray:
            raise ValueError(f"plugin {plugin.name!r} serves {stray} outside /v2/{plugin.name} and /{plugin.name}")
        app.include_router(router)
    routes = app.router.routes
    routes[:] = [r for r in routes if getattr(r, "path", None) != CATCH_ALL] + [
        r for r in routes if getattr(r, "path", None) == CATCH_ALL
    ]
    for plugin in loaded:
        keepalive_paths.extend(plugin.keepalive_paths)
