"""Steering bundles: several direction sets served together, driven by one request field.

A bundle file (``--steer-bundle``) names direction sets and carries the UI metadata
the studio reads from ``/tts/controls``::

    {"name": "studio_mect60k",
     "sets": {"identity": "speaker_attributes_mect60k", "delivery": "ears_styles_mect60k", ...},
     "timing": {"delivery.tempo": [[-2, 2.2], [0, 1.0], [2, 0.63]]},
     "ui": {...}}

Set directories are relative to the bundle file. A request's ``controls`` maps
``<set>.<direction>`` to a raw dose. Directions of ``raw`` sets move the speaker
embedding (their deltas summed, the result rescaled to the embedding's norm, as one
raw emotion direction does); ``attnres_source`` directions add to their Full
AttnRes bank source on every generated frame. ``control_spans`` apply hidden
directions to the frames generated while a character range of the text is spoken:
frames are placed by character position at ``frames_per_char`` (the client's
measurement from an earlier take, or a default), stretched inside each span by the
``timing`` table of the controls that change the pace, with linear ramps of
``RAMP_FRAMES`` at both edges. ``control_span_follow: "reading"`` places the same
span deltas by the model's reading position instead (zonos3.tts.span_follow).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

from .emotion import EmotionDirections
from .prompt import nonverbal_marker_spans

HIDDEN_SPACE = "attnres_source"
RAW_SPACE = "raw"
RAMP_FRAMES = 3                # 0.2 s at 15 frames per second
DEFAULT_FRAMES_PER_CHAR = 1.0  # release LM, embedding mode, unsteered (15 fps, ~15 characters per second)


@dataclass
class Direction:
    id: str
    set: str
    name: str
    space: str
    vector: torch.Tensor
    source_index: int | None


@dataclass(frozen=True)
class RateExtension:
    """Tempo-steered speaking-rate buckets past the condition row (bundle ``rate_extension``;
    docs/guide.md "Extended speaking-rate buckets"): the pace direction, the row edges they continue, the nominal edges past
    the row (``below[k]`` the lower edge of bucket -k, ``above[k]`` the upper edge of bucket n-1+k for the row's n
    buckets) and per TTS language the buckets measured safe with their dose (bucket -> dose, contiguous from the row).
    ``reference_sides`` ("slower" / "faster"): the sides that also serve reference-mode voices (a dub's source voice);
    the buckets were validated in embedding mode."""

    direction: str
    row_edges: tuple[float, ...]
    below: tuple[float, ...]
    above: tuple[float, ...]
    doses: Dict[str, Dict[int, float]]
    reference_sides: tuple[str, ...] = ("slower", "faster")


@dataclass
class SteeringBundle:
    name: str
    path: Path
    sets: Dict[str, EmotionDirections]
    directions: Dict[str, Direction]
    timing: Dict[str, List[List[float]]] = field(default_factory=dict)
    ui: dict = field(default_factory=dict)
    rate_extension: RateExtension | None = None

    @property
    def hidden_sources(self) -> List[int]:
        return sorted({int(d.source_index) for d in self.sets.values() if d.space == HIDDEN_SPACE})

    @property
    def raw_dim(self) -> int | None:
        dims = {int(d.dim) for d in self.sets.values() if d.space == RAW_SPACE}
        return dims.pop() if dims else None


def load_bundle(path: str | Path) -> SteeringBundle:
    path = Path(path).expanduser()
    spec = json.loads(path.read_text())
    sets: Dict[str, EmotionDirections] = {}
    directions: Dict[str, Direction] = {}
    raw_dims = set()
    for set_name, rel in spec["sets"].items():
        if "." in set_name:
            raise ValueError(f"{path}: set name {set_name!r} must not contain '.'")
        directory = (path.parent / rel).resolve()
        loaded = EmotionDirections.load(directory)
        if loaded is None or loaded.is_empty():
            raise ValueError(f"{path}: set {set_name!r} has no directions at {directory}")
        if loaded.space not in (RAW_SPACE, HIDDEN_SPACE):
            raise ValueError(f"{path}: set {set_name!r} is {loaded.space!r}; bundles take raw or attnres_source sets")
        if loaded.space == RAW_SPACE:
            raw_dims.add(int(loaded.dim))
        sets[set_name] = loaded
        for name, vec in {**loaded.named, **loaded.axes}.items():
            did = f"{set_name}.{name}"
            directions[did] = Direction(did, set_name, name, loaded.space, vec,
                                        loaded.source_index if loaded.space == HIDDEN_SPACE else None)
    if len(raw_dims) > 1:
        raise ValueError(f"{path}: raw sets disagree on the embedding dim: {sorted(raw_dims)}")
    timing = spec.get("timing", {})
    unknown = sorted(set(timing) - set(directions))
    if unknown:
        raise ValueError(f"{path}: timing for unknown directions {unknown}")
    rate_extension = None
    if spec.get("rate_extension"):
        r = spec["rate_extension"]
        if r["direction"] not in directions or directions[r["direction"]].space != HIDDEN_SPACE:
            raise ValueError(f"{path}: rate_extension direction {r['direction']!r} is not a hidden direction of the bundle")
        rate_extension = RateExtension(
            direction=r["direction"], row_edges=tuple(float(e) for e in r["row_edges"]),
            below=tuple(float(e) for e in r["below"]), above=tuple(float(e) for e in r["above"]),
            doses={lang: {int(b): float(d) for b, d in doses.items()} for lang, doses in r["languages"].items()},
            reference_sides=tuple(r.get("reference_sides", ("slower", "faster"))),
        )
    return SteeringBundle(name=spec.get("name", path.stem), path=path, sets=sets, directions=directions,
                          timing=timing, ui=spec.get("ui", {}), rate_extension=rate_extension)


def _duration_factor(bundle: SteeringBundle, controls: Mapping[str, float]) -> float:
    """Expected duration multiplier of a set of controls (product over the timed ones)."""
    factor = 1.0
    for did, table in bundle.timing.items():
        dose = float(controls.get(did, 0.0))
        if dose:
            xs, ys = zip(*sorted((float(x), float(y)) for x, y in table))
            factor *= float(np.interp(dose, xs, ys))
    return factor


@dataclass
class ResolvedControls:
    """What a request's controls do: the moved speaker embedding (or the original), the hidden
    delta laid out per steered source (flattened; None if no hidden control), the span
    schedule tensor [K, 3 + n_sources*hidden] (start frame, end frame, ramp frames, delta) or None,
    and the character range [start, end) of each schedule row (zonos3.tts.span_follow places
    the rows by the reading position instead)."""

    speaker_embedding: torch.Tensor | None
    hidden_delta: torch.Tensor | None
    schedule: torch.Tensor | None
    span_chars: list[tuple[int, int]] | None = None


def check_controls(bundle: SteeringBundle, controls: Mapping[str, float], span_ids: Sequence[str] = ()) -> None:
    """Raise ValueError for an id the bundle does not serve (in ``controls`` or ``span_ids``)
    or a non-finite dose in ``controls``."""
    for did in list(controls) + list(span_ids):
        if did not in bundle.directions:
            raise ValueError(f"Unknown control {did!r}; see /tts/controls.")
    for did, v in controls.items():
        if not np.isfinite(v):
            raise ValueError(f"Control {did!r} must be finite, got {v!r}.")


def resolve_controls(
    bundle: SteeringBundle,
    controls: Mapping[str, float] | None,
    spans: Sequence[Mapping] | None,
    *,
    text: str,
    speaker_embedding: torch.Tensor | None,
    sources: Sequence[int],
    hidden_size: int,
    frames_per_char: float | None = None,
) -> ResolvedControls:
    controls = {k: float(v) for k, v in (controls or {}).items() if float(v) != 0.0}
    spans = list(spans or [])
    check_controls(bundle, controls, [d for s in spans for d in s.get("controls", {})])

    def hidden_vector(doses: Mapping[str, float]) -> torch.Tensor | None:
        parts = [(bundle.directions[d], v) for d, v in doses.items()
                 if float(v) != 0.0 and bundle.directions[d].space == HIDDEN_SPACE]
        if not parts:
            return None
        out = torch.zeros(len(sources), hidden_size, dtype=torch.float32)
        for d, v in parts:
            if d.vector.numel() != hidden_size:
                raise ValueError(f"Control {d.id!r} has dim {d.vector.numel()}, the model's hidden size is {hidden_size}.")
            out[list(sources).index(int(d.source_index))] += float(v) * d.vector
        return out.flatten()

    # speaker embedding: every raw control at once, norm kept
    raw = {d: v for d, v in controls.items() if bundle.directions[d].space == RAW_SPACE}
    embedding = speaker_embedding
    if raw:
        if speaker_embedding is None:
            names = ", ".join(sorted(raw))
            raise ValueError(f"Controls {names} move the speaker embedding and need speaker_mode=embedding.")
        base = speaker_embedding.detach().to(torch.float32, copy=True).reshape(-1).cpu()
        delta = torch.zeros_like(base)
        for did, v in raw.items():
            vec = bundle.directions[did].vector
            if vec.numel() != base.numel():
                raise ValueError(f"Control {did!r} has dim {vec.numel()}, the speaker embedding has {base.numel()}.")
            delta += v * vec
        moved = base + delta
        embedding = (moved * (torch.linalg.vector_norm(base) / torch.linalg.vector_norm(moved).clamp_min(1e-8))).contiguous()

    hidden = hidden_vector(controls)

    schedule, span_chars = None, None
    if spans:
        n = len(text)
        rate = float(frames_per_char or DEFAULT_FRAMES_PER_CHAR) * _duration_factor(bundle, controls)
        if rate <= 0:
            raise ValueError("control_span_rate must be positive.")
        cleaned = []
        for s in spans:
            start, end = int(s["start"]), int(s["end"])
            doses = {k: float(v) for k, v in s.get("controls", {}).items() if float(v) != 0.0}
            if not 0 <= start < end <= n:
                raise ValueError(f"control span [{start}, {end}) is outside the text (0..{n}).")
            for did in doses:
                if bundle.directions[did].space != HIDDEN_SPACE:
                    raise ValueError(f"Control {did!r} moves the speaker embedding and cannot vary inside a request; "
                                     "only hidden-space controls take spans.")
            cleaned.append((start, end, doses))
        cleaned.sort()
        for (s0, e0, _), (s1, _e1, _) in zip(cleaned, cleaned[1:]):
            if s1 < e0:
                raise ValueError(f"control spans [{s0}, {e0}) and [{s1}, ...) overlap.")
        # frame position of every character boundary: span characters run at their own pace; nonverbal markers
        # (zonos3.tts.prompt: "<laugh/>", "<shouting>" ...) take no time (control_span_rate counts spoken characters)
        per_char = np.full(n, rate)
        for start, end, doses in cleaned:
            per_char[start:end] = rate * _duration_factor(bundle, doses)
        for start, end in nonverbal_marker_spans(text):
            per_char[start:end] = 0.0
        frame_at = np.concatenate([[0.0], np.cumsum(per_char)])
        rows, chars = [], []
        for start, end, doses in cleaned:
            vec = hidden_vector(doses)
            if vec is None:
                continue
            head = torch.tensor([round(frame_at[start]), round(frame_at[end]), RAMP_FRAMES], dtype=torch.float32)
            rows.append(torch.cat([head, vec]))
            chars.append((start, end))
        schedule = torch.stack(rows) if rows else None
        span_chars = chars if rows else None
    return ResolvedControls(speaker_embedding=embedding, hidden_delta=hidden, schedule=schedule, span_chars=span_chars)


def schedule_delta(schedule: torch.Tensor | None, frame: int) -> torch.Tensor | None:
    """Sum of the span deltas active at generated frame ``frame``: weight 1 inside [start, end), ramping
    linearly to 0 over ``ramp`` frames on both sides."""
    if schedule is None:
        return None
    start, end, ramp = schedule[:, 0], schedule[:, 1], schedule[:, 2].clamp_min(1.0)
    f = float(frame)
    w = torch.minimum((f - (start - ramp)) / ramp, ((end + ramp) - f) / ramp).clamp(0.0, 1.0)
    if not bool((w > 0).any()):
        return None
    return (w[:, None] * schedule[:, 3:]).sum(0)


def controls_document(bundle: SteeringBundle, *, speaker_embedding_dim: int | None) -> dict:
    """The /tts/controls payload: sets, directions (raw vectors included for client-side geometry) and UI metadata."""
    sets = {}
    for name, d in bundle.sets.items():
        entry = {"space": d.space, "modes": ["embedding"] if d.space == RAW_SPACE else ["embedding", "reference"]}
        if d.space == HIDDEN_SPACE:
            entry["source_index"] = int(d.source_index)
        sets[name] = entry
    directions = {}
    for did, d in bundle.directions.items():
        kind = "axis" if d.name in bundle.sets[d.set].axes else "named"
        entry = {"set": d.set, "space": d.space, "kind": kind, "norm": float(torch.linalg.vector_norm(d.vector))}
        if d.space == RAW_SPACE:
            entry["vector"] = [round(float(x), 6) for x in d.vector]
        directions[did] = entry
    return {"bundle": bundle.name, "embedding_dim": speaker_embedding_dim, "sets": sets, "directions": directions,
            "timing": bundle.timing, "ui": bundle.ui}
