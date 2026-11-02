"""Companion GGUFs -> the Python request-logic objects: emotion / accent directions, the steering bundle, accent adapters.

Layouts are those ``tools/convert_gguf.py`` writes (``steer``, ``directions``, ``accents``); the rebuilt dataclasses are
the ones zonos3.tts.emotion / zonos3.tts.steering load from directories, so every request path stays in zonos3.tts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from zonos3.tts.emotion import EmotionCalibration, EmotionDirections
from zonos3.tts.steering import HIDDEN_SPACE, Direction, RateExtension, SteeringBundle

from zonos3_mlx.gguf_io import GGUFFile

DIR_ARCH, STEER_ARCH, ACC_ARCH = "zonos3_directions", "zonos3_steer", "zonos3_accents"


def _expect(f: GGUFFile, arch: str) -> None:
    if f.arch != arch:
        raise ValueError(f"{f.path}: general.architecture is {f.arch!r}, expected {arch!r}")


def _vector(f: GGUFFile, name: str) -> torch.Tensor:
    return torch.from_numpy(np.array(f.array(name), dtype=np.float32).reshape(-1))


def _calibration(text: str) -> EmotionCalibration | None:
    if not text:
        return None
    data = json.loads(text)
    return EmotionCalibration(
        by_speaker={k: {e: float(v) for e, v in d.items()} for k, d in data.get("by_speaker", {}).items()},
        default={e: float(v) for e, v in data.get("default", {}).items()},
        global_default=float(data.get("global_default", 1.0)),
    )


def load_directions(path: str | Path) -> tuple[EmotionDirections, EmotionCalibration | None]:
    """A ``zonos3_directions`` GGUF (emotion.gguf, accent-directions.gguf) and its calibration."""
    f = GGUFFile(path)
    _expect(f, DIR_ARCH)
    k = lambda key: f"{DIR_ARCH}.{key}"  # noqa: E731
    space = f.meta(k("space"))
    source = int(f.meta(k("source_index")))
    names, axes = list(f.meta(k("names"), [])), list(f.meta(k("axes"), []))
    ref = float(f.meta(k("ref_base_norm")))
    lda = {}
    if space == "lda":
        lda = {f"lda_{p}": torch.from_numpy(f.array(f"dir.lda.{p}").astype(np.float32)) for p in ("weight", "bias", "pinv")}
    directions = EmotionDirections(
        dim=int(f.meta(k("dim"))),
        named={n: _vector(f, f"dir.named.{i}") for i, n in enumerate(names)},
        axes={n: _vector(f, f"dir.axis.{i}") for i, n in enumerate(axes)},
        tiers={str(a): str(b) for a, b in json.loads(f.meta(k("tiers"), "{}")).items()},
        presets={str(p): {c: float(w) for c, w in v.items()} for p, v in json.loads(f.meta(k("presets"), "{}")).items()},
        ref_base_norm=None if math.isnan(ref) else ref,
        space=space,
        source_index=None if source < 0 else source,
        input_dim=int(f.meta(k("input_dim"))),
        **lda,
    )
    return directions, _calibration(f.meta(k("calibration"), ""))


def load_steer(path: str | Path) -> SteeringBundle:
    """A ``zonos3_steer`` GGUF (steer.gguf) as the bundle zonos3.tts.steering.load_bundle builds."""
    f = GGUFFile(path)
    _expect(f, STEER_ARCH)
    k = lambda key: f"{STEER_ARCH}.{key}"  # noqa: E731
    set_names, spaces = f.meta(k("set.names")), f.meta(k("set.spaces"))
    sources = [int(v) for v in f.meta(k("set.sources"))]
    ids, kinds = f.meta(k("direction.ids")), f.meta(k("direction.kinds"))
    groups: dict[str, dict[str, dict[str, torch.Tensor]]] = {s: {"named": {}, "axis": {}} for s in set_names}
    for i, (did, kind) in enumerate(zip(ids, kinds)):
        set_name, _, name = did.partition(".")
        groups[set_name][kind][name] = _vector(f, f"steer.dir.{i}")
    sets, directions = {}, {}
    for set_name, space, source in zip(set_names, spaces, sources):
        named, axes = groups[set_name]["named"], groups[set_name]["axis"]
        dim = int(next(iter({**named, **axes}.values())).numel())
        sets[set_name] = EmotionDirections(dim=dim, named=named, axes=axes, space=space,
                                           source_index=source if space == HIDDEN_SPACE else None)
        for name, vec in {**named, **axes}.items():
            did = f"{set_name}.{name}"
            directions[did] = Direction(did, set_name, name, space, vec, source if space == HIDDEN_SPACE else None)
    rate_extension = None
    raw = f.meta(k("rate_extension"), "")
    if raw:
        r = json.loads(raw)
        if r["direction"] not in directions or directions[r["direction"]].space != HIDDEN_SPACE:
            raise ValueError(f"{path}: rate_extension direction {r['direction']!r} is not a hidden direction of the bundle")
        rate_extension = RateExtension(
            direction=r["direction"], row_edges=tuple(float(e) for e in r["row_edges"]),
            below=tuple(float(e) for e in r["below"]), above=tuple(float(e) for e in r["above"]),
            doses={lang: {int(b): float(d) for b, d in doses.items()} for lang, doses in r["languages"].items()},
            reference_sides=tuple(r.get("reference_sides", ("slower", "faster"))),
        )
    return SteeringBundle(name=f.meta(k("name")), path=Path(path), sets=sets, directions=directions,
                          timing=json.loads(f.meta(k("timing"), "{}")), ui=json.loads(f.meta(k("ui"), "{}")),
                          rate_extension=rate_extension)


# ----------------------------------------------------------------------------------------------------- accents


@dataclass
class _Slot:
    linear: dict[str, tuple[mx.array, mx.array]]  # GGUF base name -> (down [K, in], up [out, K])
    experts: dict[int, dict[str, tuple[mx.array, mx.array]]]  # layer -> gate|up|down -> (down [E, r, in], up [E, out, r])
    router_bias: dict[int, mx.array]


class AccentBank:
    """The adapters of a ``zonos3_accents`` GGUF (accents.gguf)."""

    def __init__(self, names: list[str], formats: list[str], slots: list[_Slot]):
        self.names, self.formats, self._slots = names, formats, slots

    @classmethod
    def load(cls, path: str | Path) -> "AccentBank":
        f = GGUFFile(path)
        _expect(f, ACC_ARCH)
        names = list(f.meta(f"{ACC_ARCH}.names"))
        formats = list(f.meta(f"{ACC_ARCH}.format"))
        slots = [_Slot({}, {}, {}) for _ in names]
        pairs: dict[tuple[int, str], dict[str, mx.array]] = {}
        for tname in f.names():
            if not tname.startswith("acc."):
                continue
            _, slot, rest = tname.split(".", 2)
            value = mx.array(f.array(tname).astype(np.float32))
            if rest.endswith(".router_bias"):
                layer = int(rest.split(".")[1])
                slots[int(slot)].router_bias[layer] = value
                continue
            base, _, part = rest.rpartition(".")
            pairs.setdefault((int(slot), base), {})[part] = value
        for (slot, base), parts in pairs.items():
            down, up = parts["lora_down"], parts["lora_up"]
            if base.endswith(("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")):
                layer = int(base.split(".")[1])
                kind = base.rsplit(".", 1)[1].removeprefix("ffn_").removesuffix("_exps")
                slots[slot].experts.setdefault(layer, {})[kind] = (down, up)
            else:
                slots[slot].linear[base] = (down, up)
        return cls(names, formats, slots)

    def state(self, name: str, strength: float) -> "AccentState":
        if name not in self.names:
            raise KeyError(name)
        return AccentState(self._slots[self.names.index(name)], float(strength))


class AccentState:
    """One adapter at one strength on the MLX LM: the hook points of zonos3.reference.accents.AccentState."""

    def __init__(self, slot: _Slot, strength: float):
        self._slot, self.strength = slot, strength

    def linear(self, base: str, x: mx.array, y: mx.array) -> mx.array:
        factors = self._slot.linear.get(base)
        if factors is None:
            return y
        down, up = factors
        return y + ((x @ down.T) * self.strength) @ up.T

    def router_bias(self, layer: int, bias: mx.array) -> mx.array:
        shift = self._slot.router_bias.get(layer)
        return bias if shift is None else bias + self.strength * shift

    def expert_shift(self, layer: int, kind: str, choices: mx.array, rows: mx.array, y: mx.array) -> mx.array:
        """``y + strength * up[e] @ (down[e] @ row)`` per routed pair: rows [n, k, 1, in], choices [n, k]."""
        factors = self._slot.experts.get(layer, {}).get(kind)
        if factors is None:
            return y
        down, up = factors
        t = rows @ down[choices].swapaxes(-1, -2)  # [n, k, 1, r]
        return y + self.strength * (t @ up[choices].swapaxes(-1, -2))
