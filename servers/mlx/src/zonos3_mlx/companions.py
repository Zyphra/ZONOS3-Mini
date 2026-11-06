"""Companion GGUFs -> the Python request-logic objects: the steering bundle (direction sets + calibrations), accent
adapters.

Layouts are those ``tools/convert_gguf.py`` writes (``steer``, ``accents``); the rebuilt dataclasses are the ones
zonos3.tts.directions / zonos3.tts.steering load from directories, so every request path stays in zonos3.tts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from zonos3.models.accent_format import FORMAT_GROWN
from zonos3.tts.directions import HIDDEN_SPACE, SPACES, Calibration, DirectionSet
from zonos3.tts.steering import SteeringBundle, build_bundle

from zonos3_mlx.gguf_io import GGUFFile

STEER_ARCH, ACC_ARCH = "zonos3_steer", "zonos3_accents"


def _expect(f: GGUFFile, arch: str) -> None:
    if f.arch != arch:
        raise ValueError(f"{f.path}: general.architecture is {f.arch!r}, expected {arch!r}")


def _vector(f: GGUFFile, name: str) -> torch.Tensor:
    return torch.from_numpy(np.array(f.array(name), dtype=np.float32).reshape(-1))


def load_steer(path: str | Path) -> SteeringBundle:
    """A ``zonos3_steer`` GGUF (steer.gguf) as the bundle zonos3.tts.steering.load_bundle builds."""
    f = GGUFFile(path)
    _expect(f, STEER_ARCH)
    k = lambda key: f"{STEER_ARCH}.{key}"  # noqa: E731
    set_names, spaces = f.meta(k("set.names")), f.meta(k("set.spaces"))
    sources = [int(v) for v in f.meta(k("set.sources"))]
    calibrations = f.meta(k("set.calibrations"))
    if len(calibrations) != len(set_names):
        raise ValueError(f"{path}: {len(calibrations)} set calibrations for {len(set_names)} sets")
    ids, kinds = f.meta(k("direction.ids")), f.meta(k("direction.kinds"))
    groups: dict[str, dict[str, dict[str, torch.Tensor]]] = {s: {"named": {}, "axis": {}} for s in set_names}
    for i, (did, kind) in enumerate(zip(ids, kinds)):
        set_name, _, name = did.partition(".")
        groups[set_name][kind][name] = _vector(f, f"steer.dir.{i}")
    sets = {}
    for set_name, space, source, calibration in zip(set_names, spaces, sources, calibrations):
        if space not in SPACES:
            raise ValueError(f"{path}: set {set_name!r} is {space!r}; bundles take raw or attnres_source sets")
        named, axes = groups[set_name]["named"], groups[set_name]["axis"]
        dim = int(next(iter({**named, **axes}.values())).numel()) if named or axes else 0
        sets[set_name] = DirectionSet(dim=dim, named=named, axes=axes, space=space,
                                      source_index=source if space == HIDDEN_SPACE else None,
                                      calibration=Calibration.from_document(json.loads(calibration))
                                      if calibration else None)
    raw = f.meta(k("rate_extension"), "")
    return build_bundle(Path(path), f.meta(k("name")), sets, timing=json.loads(f.meta(k("timing"), "{}")),
                        ui=json.loads(f.meta(k("ui"), "{}")), rate_extension=json.loads(raw) if raw else None)


# ----------------------------------------------------------------------------------------------------- accents


@dataclass
class _Slot:
    linear: dict[str, tuple[mx.array, mx.array]]  # GGUF base name -> (down [K, in], up [out, K])
    experts: dict[int, dict[str, tuple[mx.array, mx.array]]]  # layer -> gate|up|down -> (down [E, r, in], up [E, out, r])
    router_bias: dict[int, mx.array]


class AccentBank:
    """The adapters of a ``zonos3_accents`` GGUF (accents.gguf). A ``zonos-accent-lora/3`` slot (at most one) is the
    speech-edit adapter (``edit``): edit fills run on it at strength 1, and it is not in ``accent_names``."""

    def __init__(self, names: list[str], formats: list[str], slots: list[_Slot]):
        self.names, self.formats, self._slots = names, formats, slots
        edits = [name for name, fmt in zip(names, formats) if fmt == FORMAT_GROWN]
        if len(edits) > 1:
            raise ValueError(f"accents.gguf holds more than one speech-edit adapter ({', '.join(edits)})")
        self.edit = edits[0] if edits else None
        self.accent_names = [name for name in names if name != self.edit]

    @classmethod
    def load(cls, path: str | Path, *, head_rows: int | None = None) -> "AccentBank":
        """``head_rows``: the backbone's output head, to fit the head factor of an adapter trained before the vocabulary
        grew (zonos3.models.accent_format.fit_head: zero rows at each block's span-end row)."""
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
                if base == "output" and head_rows is not None and up.shape[0] < head_rows:
                    n_codebooks = head_rows - up.shape[0]
                    if up.shape[0] % n_codebooks == 0:  # zonos3.models.accent_format.fit_head
                        k = up.shape[1]
                        blocks = up.reshape(n_codebooks, -1, k)
                        up = mx.concatenate([blocks, mx.zeros((n_codebooks, 1, k), up.dtype)],
                                            axis=1).reshape(head_rows, k)
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
