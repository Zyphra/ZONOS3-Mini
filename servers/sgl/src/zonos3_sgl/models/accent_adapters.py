"""Per-request accent adapters: batched low-rank shifts on the served model.

An accent adapter (zonos3.models.accent_format: the file format and its parser)
shifts linear weights by ``up @ down``, router balancing biases by a vector and,
in ``/2``, every MoE expert weight by its own ``up_e @ down_e``. A request names
one adapter and a strength alpha and is served as ``base + alpha * adapter`` --
the merge the adapter was scored as offline -- while sharing batches and CUDA
graphs with requests that use other adapters, other strengths or none.

Every loaded adapter owns ``rank`` columns of one shared K-wide space. An
adapted op stores ``down`` [K, in] and ``up`` [K, out], zero where an adapter
leaves the op alone, and adds ``((x @ down.T) * mask) @ up`` to its output. The
mask [tokens, K] holds each token's request strength on its adapter's columns
and zero elsewhere. It is built once per forward from per-token slots and
strengths, so the cost of mixing adapters is fixed by K, not by how a batch is
composed. Slot 0 is the base model: its rows add exact zeros.

Name mapping. An adapter's module names are the checkpoint's, so they resolve
by attribute path on the served model. Each ``TARGETS`` entry names the op
class it may resolve to; every one of those ops adds its shift to the flat
``[out, in]`` product before splitting it, which is the product lingua's LoRA
adapts:

    attention.wq / gater          Attention.wq / gater (SimpleLinear)
    attention.wkv                 Attention.wkv (ChunkedLinear [2, kv, in] as [2*kv, in]; K|V split after)
    attention.wo                  Attention.wo (SimpleLinearOProj) or OriginalGDN.wo
    attention.in_proj             OriginalGDN.in_proj ([2056, in]; qkv|z|a|b split after), also on the
                                  replayed 64-row chunk tail (see zonos3_sgl.models.gdn)
    feed_forward.w_in / w_out     FeedForward.w_in (ChunkedLinear [2, inter, in]; h|gate split after) /
                                  w_out (LinearRowParallel)
    feed_forward.latent_down/up   MoEFeedForward.latent_down / latent_up (SimpleLinear around the experts)
    router.router_linear          Router.router_linear (its logits feed both the softmax and the
                                  quantile top-k)
    multi_output                  MultiOutputHead (shift on the real rows after min_rows padding)

Router bias shifts become a per-slot table added per token to the routing bias.

Expert factors (``/2``, SonicExperts layout: w13 rows interleave gate/up, as the
exact path's ``_sonic_w13``) are served on the exact SonicMoE path
(``ZONOS3_USE_TRAINING_SONIC_EXPERTS=1``) inside its two grouped GEMMs, as the
addend C of ``D = A @ B + C`` (zonos3_sgl.layers.accent.sonic_experts_with_accents):
each routed (token, expert) pair of a request on adapter a adds
``alpha * up[a, e] @ (down[a, e] @ x)`` to its gate/up pre-activation and
``alpha * up2[a, e] @ (down2[a, e] @ h)`` to its expert output before the top-k
combine. Adapters are grouped by expert rank (one ``ExpertRankGroup`` per rank
and layer, ``[adapters * E, ...]`` factor stacks indexed by (adapter, expert));
a Triton kernel computes each pair's term from its own factors
(zonos3_sgl.kernel.triton.expert_lora). Pairs of no adapter or of an adapter
without expert factors get an exact zero addend, and the kernels with a zero
addend give the bits of the plain SonicMoE call.

Adapters are files ``<accent>.pt`` in one directory; the file stem is the name
requests use. Tensor parallelism is not supported.
"""

from __future__ import annotations

from pathlib import Path

import torch
from zonos3.models.accent_format import (
    EXPERT_WEIGHTS,
    TARGETS,
    fit_head,
    list_accent_adapters,
    load_adapter,
    lora_target,
)


def _served_classes() -> dict[str, tuple[tuple[type, ...], tuple[type, ...]]]:
    """lingua LoRA target -> (owner module classes, op classes) that apply its shift."""
    from zonos3_sgl.layers.linear import ChunkedLinear, LinearRowParallel

    from .gdn import OriginalGDN, _Weight
    from .zonos3 import (
        Attention,
        FeedForward,
        MoEFeedForward,
        MultiOutputHead,
        Router,
        SimpleLinear,
        SimpleLinearOProj,
        Zonos3ForCausalLM,
        _RouterLinear,
    )

    return {
        "attention.wq": ((Attention,), (SimpleLinear,)),
        "attention.wkv": ((Attention,), (ChunkedLinear,)),
        "attention.wo": ((Attention, OriginalGDN), (SimpleLinearOProj, _Weight)),
        "attention.gater": ((Attention,), (SimpleLinear,)),
        "attention.in_proj": ((OriginalGDN,), (_Weight,)),
        "feed_forward.w_in": ((FeedForward,), (ChunkedLinear,)),
        "feed_forward.w_out": ((FeedForward,), (LinearRowParallel,)),
        "feed_forward.latent_down": ((MoEFeedForward,), (SimpleLinear,)),
        "feed_forward.latent_up": ((MoEFeedForward,), (SimpleLinear,)),
        "router.router_linear": ((Router,), (_RouterLinear,)),
        "multi_output": ((Zonos3ForCausalLM,), (MultiOutputHead,)),
    }


def _resolve(model, name: str):
    """(owner, op) for a module name, e.g. ``layers.3.attention.wq``."""
    from zonos3_sgl.layers.base import OPList

    owner, obj = None, model
    for part in name.split("."):
        owner = obj
        if isinstance(obj, OPList) and part.isdigit():
            index = int(part)
            obj = obj.op_list[index] if index < len(obj.op_list) else None
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            raise ValueError(f"accent adapter targets {name!r}, which the served model lacks")
    return owner, obj


def _resolve_linear(model, module: str):
    """The served op that applies the shift of lingua LoRA module ``module``."""
    target = lora_target(module)
    if target is None:
        raise ValueError(
            f"accent adapter shifts {module!r}, outside the LoRA targets {', '.join(TARGETS)}"
        )
    owner, op = _resolve(model, module)
    owners, ops = _served_classes()[target]
    if not isinstance(owner, owners) or not isinstance(op, ops):
        raise ValueError(
            f"accent adapter target {module!r} resolves to {type(owner).__name__}."
            f"{type(op).__name__}, which does not apply {target} shifts"
        )
    return op


class ExpertRankGroup:
    """Expert factors of the adapters with one expert rank on one MoE layer.

    Group ``local * E + e`` is expert e of the group's ``local``-th adapter;
    ``first[slot]`` is ``local * E`` for the group's slots and -1 for every
    other slot. ``factors[w]`` holds ``down`` [G, in, rank] and ``up``
    [G, rank, out]: the per-expert ``down.T`` / ``up.T``, so a pair reads
    contiguous rows (zonos3_sgl.kernel.triton.expert_lora).
    """

    def __init__(self, rank: int, first: torch.Tensor, factors: dict):
        self.rank = rank
        self.first = first  # [slots + 1] int64
        self.factors = factors  # weight name -> (down, up)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for pair in self.factors.values() for t in pair)


class ExpertAccents:
    """One MoE layer's expert factors, one ExpertRankGroup per expert rank."""

    def __init__(self, num_experts: int, groups: list[ExpertRankGroup]):
        self.num_experts = num_experts
        self.groups = groups

    def nbytes(self) -> int:
        return sum(group.nbytes() for group in self.groups)


class AccentAdapters:
    """Loaded adapters: their slots and the K-wide column layout."""

    def __init__(
        self,
        names: list[str],
        column_slots: torch.Tensor,
        ops: dict | None = None,
        expert_ops: dict | None = None,
    ):
        self.names = names
        self.slots = {name: index + 1 for index, name in enumerate(names)}
        self.column_slots = column_slots  # [K] int32, slot of each column
        self.ops = ops or {}  # module name -> adapted op
        self.expert_ops = expert_ops or {}  # experts module name -> FusedGroupedExperts

    def nbytes(self) -> int:
        """Bytes of the linear factors and expert factors (the bias tables are tiny)."""
        total = sum(t.numel() * t.element_size() for op in self.ops.values() for t in op._accent)
        total += sum(op._accent_experts.nbytes() for op in self.expert_ops.values())
        return total

    def slot(self, name: str | None) -> int:
        if name is None:
            return 0
        try:
            return self.slots[name]
        except KeyError:
            raise ValueError(
                f"unknown accent {name!r}; loaded: {', '.join(self.names)}"
            ) from None

    def mask(self, slots: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """[tokens, K]: each token's strength on its adapter's columns."""
        hit = self.column_slots.unsqueeze(0) == slots.unsqueeze(1)
        return hit.to(dtype) * scales.to(dtype).unsqueeze(1)


def attach_accent_adapters(
    model, directory: str | Path, device: torch.device, dtype: torch.dtype
) -> AccentAdapters:
    """Load every adapter in ``directory`` onto ``model``'s ops."""
    names = list_accent_adapters(directory)
    factors: dict[str, list[tuple[int, int, torch.Tensor, torch.Tensor, tuple[int, ...]]]] = {}
    biases: dict[str, list[tuple[int, torch.Tensor]]] = {}
    experts: dict[str, dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]] = {}
    widths = []
    head = getattr(model, "multi_output", None)
    for index, name in enumerate(names):
        adapter = load_adapter(Path(directory) / f"{name}.pt")
        if head is not None:
            adapter = fit_head(adapter, head.weight.shape[0])
        slot = index + 1
        for router, shift in adapter.router_bias.items():
            biases.setdefault(router, []).append((slot, shift))
        for module, (down, up) in adapter.linear.items():
            factors.setdefault(module, []).append((slot, sum(widths), down, up, adapter.shapes[module]))
        for module, weights in adapter.experts.items():
            experts.setdefault(module, {})[slot] = weights
        widths.append(adapter.rank)

    column_slots = torch.repeat_interleave(
        torch.arange(1, len(names) + 1, dtype=torch.int32), torch.tensor(widths)
    )
    width = int(column_slots.numel())
    ops = {}
    for module, entries in factors.items():
        op = ops[module] = _resolve_linear(model, module)
        weight = op.weight
        out_features, in_features = weight[..., 0].numel(), weight.shape[-1]
        down = torch.zeros(width, in_features)
        up = torch.zeros(width, out_features)
        for slot, start, pair_down, pair_up, shape in entries:
            if shape != tuple(weight.shape):
                raise ValueError(
                    f"{names[slot - 1]}: {module}.weight shape {shape} != served {tuple(weight.shape)}"
                )
            rank = pair_down.shape[0]
            down[start : start + rank] = pair_down
            up[start : start + rank] = pair_up
        op._accent = (down.to(device, dtype), up.to(device, dtype))
    for target, entries in biases.items():
        _, router = _resolve(model, target)
        if not hasattr(router, "balancing_biases"):
            raise ValueError(f"accent adapter shifts {target}.balancing_biases, not a router")
        table = torch.zeros(len(names) + 1, router.balancing_biases.numel())
        for slot, shift in entries:
            table[slot] = shift
        router._accent_bias = table.to(device)
    expert_ops = _attach_expert_factors(model, experts, len(names), device, dtype)
    adapters = AccentAdapters(names, column_slots.to(device), ops, expert_ops)
    model._accents = adapters
    return adapters


def _attach_expert_factors(
    model, experts: dict[str, dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]], num_slots: int, device, dtype
) -> dict:
    """Per-expert factors (experts module -> slot -> weight -> (up, down)) -> an ``ExpertAccents`` on each
    FusedGroupedExperts."""
    from .zonos3 import FusedGroupedExperts, MoEFeedForward

    expert_ops = {}
    for module, per_slot in experts.items():
        owner, op = _resolve(model, module)
        if not isinstance(owner, MoEFeedForward) or not isinstance(op, FusedGroupedExperts):
            raise ValueError(f"accent adapter expert factors on {module!r}, which is not MoE experts")
        if not op._use_sonic_experts:
            raise ValueError(
                "accent adapters with expert factors are served on the exact SonicMoE path only "
                "(ZONOS3_USE_TRAINING_SONIC_EXPERTS=1)"
            )
        served = {"w13": tuple(op.gate_up_proj.shape), "w2": tuple(op.down_proj.shape)}
        ranks: dict[int, list[int]] = {}
        for slot, weights in sorted(per_slot.items()):
            slot_ranks = set()
            for weight, (pair_up, pair_down) in weights.items():
                num, out_features, in_features = served[weight]
                rank = pair_down.shape[-2]
                if (
                    tuple(pair_up.shape) != (num, out_features, rank)
                    or tuple(pair_down.shape) != (num, rank, in_features)
                ):
                    raise ValueError(
                        f"slot {slot}: {module}.{weight} factors up {tuple(pair_up.shape)}, "
                        f"down {tuple(pair_down.shape)} do not shift the served [E, out, in] {served[weight]}"
                    )
                slot_ranks.add(rank)
            if len(slot_ranks) != 1:
                raise ValueError(f"slot {slot}: {module} expert factors mix ranks {sorted(slot_ranks)}")
            ranks.setdefault(slot_ranks.pop(), []).append(slot)

        groups = []
        num = op.num_experts
        for rank, slots in sorted(ranks.items()):
            first = torch.full((num_slots + 1,), -1, dtype=torch.int64)
            factors = {}
            for weight in EXPERT_WEIGHTS:
                _, out_features, in_features = served[weight]
                # A weight an adapter leaves alone keeps zero factors (exact zero shift).
                down = torch.zeros(len(slots) * num, in_features, rank, dtype=dtype)
                up = torch.zeros(len(slots) * num, rank, out_features, dtype=dtype)
                for local, slot in enumerate(slots):
                    first[slot] = local * num
                    pair = per_slot[slot].get(weight)
                    if pair is not None:
                        rows = slice(local * num, (local + 1) * num)
                        down[rows] = pair[1].transpose(1, 2).to(dtype)
                        up[rows] = pair[0].transpose(1, 2).to(dtype)
                factors[weight] = (down.to(device), up.to(device))
            groups.append(ExpertRankGroup(rank, first.to(device), factors))
        op._accent_experts = ExpertAccents(num, groups)
        expert_ops[module] = op
    return expert_ops


BLOCK_ROWS = 16


@torch.no_grad()
def ensure_batch_invariance(adapters: AccentAdapters, rows=(1, 2, 3, 4, 5, 8, 16, 64, 256)) -> list[str]:
    """Make every adapted op's side path give a row the same bits at any row count.

    The exact-mode contract is that a request's logits do not depend on its
    batch. cuBLAS picks GEMM kernels by shape, so for some shapes a row's sum is
    rounded differently at different row counts (on H100: the LM head's
    [rows, K] @ [K, 40980] up product below 6-16 rows, up to 64 rows at K=1120).
    Each op is probed with 1..256 rows against a 1024-row product; failing ops
    run their side path in fixed-shape BLOCK_ROWS-row pieces (one kernel per
    GEMM whatever the batch) and are probed again. Returns the switched module
    names. Rows without an accent are exact either way: their mask row is zero.

    Expert side paths (zonos3_sgl.kernel.triton.expert_lora, one program per pair
    and output tile) are probed the same way, 1..256 routed pairs against 1024
    on every slot (each rank's adapters, /1 adapters, no accent); they have no
    fallback and must pass.
    """
    from zonos3_sgl.layers.accent import accent_shift, expert_pair_shift

    generator = torch.Generator(device=adapters.column_slots.device).manual_seed(20261015)
    slots = adapters.column_slots.unique()
    switched = []

    def invariant(op) -> bool:
        down, _ = op._accent
        x = torch.randn(1024, down.shape[1], device=down.device, generator=generator).to(down.dtype)
        # Every row on one adapter's columns, as in serving.
        pick = torch.randint(len(slots), (1024,), device=down.device, generator=generator)
        mask = adapters.mask(slots[pick], torch.ones(1024, device=down.device), down.dtype)
        reference = accent_shift(op, x, mask)
        return all(torch.equal(accent_shift(op, x[:n], mask[:n]), reference[:n]) for n in rows)

    for name, op in adapters.ops.items():
        if invariant(op):
            continue
        op._accent_block_rows = BLOCK_ROWS
        switched.append(name)
        if not invariant(op):
            raise RuntimeError(f"accent adapter side path of {name} is not batch-invariant")

    for name, op in adapters.expert_ops.items():
        accents = op._accent_experts
        device = accents.groups[0].first.device
        num_slots = accents.groups[0].first.numel()
        for weight, (down, _) in accents.groups[0].factors.items():
            x = torch.randn(1024, down.shape[1], device=device, generator=generator).to(down.dtype)
            tokens = torch.arange(1024, device=device, dtype=torch.int32)
            pair_slots = torch.randint(num_slots, (1024,), device=device, generator=generator).int()
            experts = torch.randint(accents.num_experts, (1024,), device=device, generator=generator).int()
            scales = torch.rand(1024, device=device, generator=generator) + 0.5
            reference = expert_pair_shift(accents, weight, x, tokens, experts, pair_slots, scales)
            for n in rows:
                shift = expert_pair_shift(
                    accents, weight, x[:n], tokens[:n], experts[:n], pair_slots[:n], scales[:n]
                )
                if not torch.equal(shift, reference[:n]):
                    raise RuntimeError(
                        f"accent expert side path of {name}.{weight} is not batch-invariant at {n} pairs"
                    )
    return switched
