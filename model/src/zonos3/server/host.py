"""The plugin host facts and LACI take rules shared by the Python servers' ``zonos3.server.plugins.Host`` (sgl
``_SglHost``, mlx ``MlxHost``; each keeps its own take scheduling and reference encode)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from zonos3.server.plugins import HostFacts, LaciTakeRequest
from zonos3.tts.directions import RAW_SPACE
from zonos3.tts.steering import SteeringBundle, resolve_controls


def speaking_rate_edges(condition_tokens: dict | None) -> list[float]:
    """The speaking-rate row's edges (UTF-8 B/s) of a checkpoint's condition rows
    (zonos3.server.capabilities.condition_spec); [] without that row."""
    return [] if condition_tokens is None else [float(e) for e in condition_tokens["edges"].get("speaking_rate") or ()]


@dataclass(frozen=True)
class RateScale:
    """A LACI take on an extended speaking-rate scale: the scale's edges, its first bucket, the take's bucket on it and
    the steering delta of every bucket (``first`` .. ``first + len(edges)``)."""

    edges: tuple[float, ...]
    first: int
    bucket: int
    deltas: list

    @property
    def delta(self):
        """The steering delta of the take's own bucket."""
        return self.deltas[self.bucket - self.first]


def rate_scale(take: LaciTakeRequest, bundle: SteeringBundle | None, *, text: str, base_delta, sources: tuple[int, ...],
               hidden_size: int) -> RateScale | None:
    """Put a LACI take on an extended speaking-rate scale (take.rate_edges, README "Extended speaking-rate buckets"): in
    a steered bucket the take's steering delta (``base_delta``) gains the bucket's tempo dose; the take carries the
    scale, its bucket on it and its delta in every bucket (LACI reports the bucket on the scale, a steered one's row id
    is only "unknown"; a rate-adjustable take's retakes switch to another bucket's delta). None without a scale."""
    if take.rate_edges is None:
        return None
    if take.rate_doses and (bundle is None or bundle.rate_extension is None):
        raise ValueError("rate_doses need a steering bundle with a rate_extension")

    def delta(b: int):
        if b not in take.rate_doses:
            return base_delta
        tempo = resolve_controls(bundle, {bundle.rate_extension.direction: take.rate_doses[b]}, None, text=text,
                                 speaker_embedding=None, sources=sources, hidden_size=hidden_size).hidden_delta
        return tempo if base_delta is None else base_delta + tempo

    if take.rate_bucket is None:
        raise ValueError("a LACI take on an extended speaking-rate scale needs its rate_bucket")
    first = int(take.rate_first)
    edges = tuple(float(e) for e in take.rate_edges)
    return RateScale(edges, first, int(take.rate_bucket), [delta(b) for b in range(first, first + len(edges) + 1)])


def laci_slots(batch: int) -> asyncio.Semaphore:
    """LACI takes in flight: twice the server's running-request cap, so other requests interleave."""
    return asyncio.Semaphore(max(1, 2 * int(batch or 1)))


def host_facts(*, model_path: str, condition_tokens: dict | None, speaker_modes: Iterable[str],
               bundle: SteeringBundle | None, laci_languages: Iterable[str], text_tokenizer_spec: dict | None,
               worker_device: str | None = None) -> HostFacts:
    """What a plugin may know of the served checkpoint and server."""
    phonemes = None if not text_tokenizer_spec else text_tokenizer_spec.get("phonemes")
    return HostFacts(
        model_name=PurePosixPath(str(model_path).rstrip("/")).name,
        rate_edges=tuple(speaking_rate_edges(condition_tokens)),
        rate_buckets=int(condition_tokens["bucket_counts"].get("speaking_rate", 0)) if condition_tokens else 0,
        laci_languages=frozenset(laci_languages),
        speaker_modes=tuple(m for m in ("reference", "embedding") if m in speaker_modes),
        raw_controls=frozenset(
            did for did, d in (bundle.directions.items() if bundle is not None else ()) if d.space == RAW_SPACE
        ),
        phoneme_tokens=phonemes is not None,
        rate_extension=bundle.rate_extension if bundle is not None else None,
        worker_device=worker_device,
    )
