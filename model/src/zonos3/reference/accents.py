"""Accent adapters on the reference LM: ``base + strength * adapter`` (the sgl server's per-request shifts).

A target linear adds ``strength * ((x @ down.T) @ up)`` to its flat ``[out, in]`` product, the router adds
``strength * shift`` to its balancing biases, and each routed (token, expert) pair adds
``strength * up[e] @ (down[e] @ row)`` to its gate/up pre-activation (w13, SonicExperts interleaved rows) and its
expert output (w2) -- zonos3_sgl.layers.accent ``accent_shift`` / ``accent_router_bias`` / ``expert_pair_shift``.
An adapter loaded with its table shifts (``load_adapter(tables=True)``) also adds ``strength * shift[id]`` to the
embedding rows a prompt reads and ``strength * normalized @ shift.T`` to the output head's logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from zonos3.models.accent_format import AccentAdapter


@dataclass(frozen=True)
class AccentState:
    adapter: AccentAdapter
    strength: float

    @classmethod
    def for_model(cls, lm, adapter: AccentAdapter, strength: float) -> "AccentState":
        """The adapter's factors on ``lm``'s device and dtype, checked against its weights."""
        params = dict(lm.named_parameters())
        device, dtype = lm.device, lm.dtype

        def weight(module: str) -> torch.Tensor:
            found = params.get(module + ".weight")
            if found is None:
                raise ValueError(f"accent {adapter.name!r} targets {module!r}, which the model lacks")
            return found

        linear = {}
        for module, (down, up) in adapter.linear.items():
            if adapter.shapes[module] != tuple(weight(module).shape):
                raise ValueError(f"{adapter.name}: {module}.weight shape {adapter.shapes[module]} != model "
                                 f"{tuple(weight(module).shape)}")
            linear[module] = (down.to(device, dtype), up.to(device, dtype))
        experts = {}
        for module, weights in adapter.experts.items():
            served = {"w13": params.get(module + ".gate_up_proj"), "w2": params.get(module + ".down_proj")}
            if served["w13"] is None:
                raise ValueError(f"accent {adapter.name!r} has expert factors on {module!r}, which is not MoE experts")
            for name, (up, down) in weights.items():
                if (up.shape[0], up.shape[1], down.shape[2]) != tuple(served[name].shape):
                    raise ValueError(f"{adapter.name}: {module}.{name} factors do not shift the model's "
                                     f"[E, out, in] {tuple(served[name].shape)}")
            experts[module] = {name: (up.to(device, dtype), down.to(device, dtype)) for name, (up, down) in weights.items()}
        router_bias = {}
        for module, shift in adapter.router_bias.items():
            biases = params.get(module + ".balancing_biases")
            if biases is None or biases.shape != shift.shape:
                raise ValueError(f"accent {adapter.name!r} shifts {module}.balancing_biases, not a router of the model")
            router_bias[module] = shift.to(device, torch.float32)
        tables = {}
        for name, shift in adapter.tables.items():
            table = params.get(name)
            if table is None or table.shape != shift.shape:
                raise ValueError(f"accent {adapter.name!r} shifts {name} {tuple(shift.shape)}, which the model has as "
                                 f"{None if table is None else tuple(table.shape)} (grow its vocabulary first)")
            tables[name] = shift.to(device, dtype)
        moved = AccentAdapter(adapter.name, adapter.format, linear, adapter.shapes, experts, router_bias, tables)
        return cls(moved, float(strength))

    def embedding(self, index: int, ids: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Embedder ``index``'s rows of ``ids`` with the adapter's table shift."""
        shift = self.adapter.tables.get(f"multi_embedder.embedders.{index}.weight")
        return rows if shift is None else rows + self.strength * F.embedding(ids, shift)

    def linear(self, module: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        factors = self.adapter.linear.get(module)
        if factors is not None:
            down, up = factors
            y = y + ((x @ down.T) * self.strength) @ up
        table = self.adapter.tables.get(module + ".weight")
        return y if table is None else y + self.strength * (x @ table.T)

    def router_bias(self, module: str, bias: torch.Tensor) -> torch.Tensor:
        shift = self.adapter.router_bias.get(module)
        return bias if shift is None else bias + self.strength * shift

    def expert_shift(self, module: str, weight: str, expert: int, row: torch.Tensor) -> torch.Tensor | None:
        factors = self.adapter.experts.get(module, {}).get(weight)
        if factors is None:
            return None
        up, down = factors
        return self.strength * (up[expert] @ (down[expert] @ row))
