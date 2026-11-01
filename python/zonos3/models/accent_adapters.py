"""Per-request accent adapters: batched low-rank shifts on the served model.

An accent adapter (lingua ``apps.zonos.accent_adapter``, format
``zonos-accent-lora/1`` or ``/2``) shifts linear weights by ``up @ down``,
router balancing biases by a vector and, in ``/2``, every MoE expert weight by
its own ``up_e @ down_e``. A request names one adapter and a strength alpha and
is served as ``base + alpha * adapter`` -- the merge the adapter was scored as
offline -- while sharing batches and CUDA graphs with requests that use other
adapters, other strengths or none.

Every loaded adapter owns ``rank`` columns of one shared K-wide space. An
adapted op stores ``down`` [K, in] and ``up`` [K, out], zero where an adapter
leaves the op alone, and adds ``((x @ down.T) * mask) @ up`` to its output. The
mask [tokens, K] holds each token's request strength on its adapter's columns
and zero elsewhere. It is built once per forward from per-token slots and
strengths, so the cost of mixing adapters is fixed by K, not by how a batch is
composed. Slot 0 is the base model: its rows add exact zeros.

Name mapping. ``scripts/convert_lingua_checkpoint.py`` keeps every non-expert
tensor under its lingua key, so an adapter key resolves by attribute path on the
served model. ``TARGETS`` mirrors lingua's ``apps.zonos.lora.DEFAULT_TARGETS``
and names the op class each target may resolve to; every one of those ops adds
its shift to the flat ``[out, in]`` product before splitting it, which is the
product lingua's LoRA adapts:

    attention.wq / gater          Attention.wq / gater (SimpleLinear)
    attention.wkv                 Attention.wkv (ChunkedLinear [2, kv, in] as [2*kv, in]; K|V split after)
    attention.wo                  Attention.wo (SimpleLinearOProj) or OriginalGDN.wo
    attention.in_proj             OriginalGDN.in_proj ([2056, in]; qkv|z|a|b split after), also on the
                                  replayed 64-row chunk tail (see zonos3.models.gdn)
    feed_forward.w_in / w_out     FeedForward.w_in (ChunkedLinear [2, inter, in]; h|gate split after) /
                                  w_out (LinearRowParallel)
    feed_forward.latent_down/up   MoEFeedForward.latent_down / latent_up (SimpleLinear around the experts)
    router.router_linear          Router.router_linear (its logits feed both the softmax and the
                                  quantile top-k)
    multi_output                  MultiOutputHead (shift on the real rows after min_rows padding)

Exact shifts: ``<router>.balancing_biases`` (float32 quantile thresholds) becomes
a per-slot table added per token to the routing bias. As in the converter, the
legacy ZeroMean parametrization ``.parametrizations.balancing_biases.original``
maps to ``balancing_biases`` with its mean removed, and the training-only router
metrics (``ent_denom``, ``normalized_entropy``) are dropped. Any other moved
tensor is refused.

Expert factors (``/2``): ``layers.N.feed_forward.experts.{w13,w2}`` carry
``up`` [E, out, r] and ``down`` [E, r, in] in lingua's SonicExperts layout
(w13 rows interleave gate/up, as the exact path's ``_sonic_w13``), ``shape``
(E, out, in) and ``experts`` E. They are served on the exact SonicMoE path
(``ZONOS3_USE_TRAINING_SONIC_EXPERTS=1``) inside its two grouped GEMMs, as the
addend C of ``D = A @ B + C`` (zonos3.layers.accent.sonic_experts_with_accents):
each routed (token, expert) pair of a request on adapter a adds
``alpha * up[a, e] @ (down[a, e] @ x)`` to its gate/up pre-activation and
``alpha * up2[a, e] @ (down2[a, e] @ h)`` to its expert output before the top-k
combine. Adapters are grouped by expert rank (one ``ExpertRankGroup`` per rank
and layer, ``[adapters * E, ...]`` factor stacks indexed by (adapter, expert));
a Triton kernel computes each pair's term from its own factors
(zonos3.kernel.triton.expert_lora). Pairs of no adapter or of an adapter
without expert factors get an exact zero addend, and the kernels with a zero
addend give the bits of the plain SonicMoE call.

Adapters are files ``<accent>.pt`` in one directory; the file stem is the name
requests use. Tensor parallelism is not supported.
"""

from __future__ import annotations

from pathlib import Path

import torch

FORMAT = "zonos-accent-lora/1"
FORMAT_EXPERTS = "zonos-accent-lora/2"
FORMATS = (FORMAT, FORMAT_EXPERTS)
# SonicExperts weights lingua's apps.zonos.lora adapts per expert.
EXPERT_WEIGHTS = ("w13", "w2")

_BALANCING_BIASES = ".balancing_biases"
_LEGACY_BALANCING_BIASES = ".parametrizations.balancing_biases.original"
# Router state the converter drops (training metrics, never used for routing).
_DROPPED_ROUTER_STATE = (".router.ent_denom", ".router.normalized_entropy")


def _served_classes() -> dict[str, tuple[tuple[type, ...], tuple[type, ...]]]:
    """lingua LoRA target -> (owner module classes, op classes) that apply its shift."""
    from zonos3.layers.linear import ChunkedLinear, LinearRowParallel

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


# Mirrors lingua apps.zonos.lora.DEFAULT_TARGETS.
TARGETS = (
    "attention.wq",
    "attention.wkv",
    "attention.wo",
    "attention.gater",
    "attention.in_proj",
    "feed_forward.w_in",
    "feed_forward.w_out",
    "feed_forward.latent_down",
    "feed_forward.latent_up",
    "router.router_linear",
    "multi_output",
)


def lora_target(module: str) -> str | None:
    """The TARGETS entry a module name matches (lingua's suffix rule), or None."""
    for target in TARGETS:
        if module == target or module.endswith("." + target):
            return target
    return None


def served_exact_name(name: str) -> tuple[str | None, bool]:
    """Served tensor an exact shift lands on, and whether to remove its mean.

    Mirrors convert_lingua_checkpoint._output_key / convert_model_state for the
    router state; None means the converter drops the tensor.
    """
    if any(part in name for part in _DROPPED_ROUTER_STATE):
        return None, False
    if name.endswith(_LEGACY_BALANCING_BIASES):
        return name.removesuffix(_LEGACY_BALANCING_BIASES) + _BALANCING_BIASES, True
    return name, False


def list_accent_adapters(directory: str | Path) -> list[str]:
    """Adapter names in a directory, in slot order (slot = index + 1)."""
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"accent adapter directory {path} does not exist")
    names = sorted(file.stem for file in path.glob("*.pt"))
    if not names:
        raise FileNotFoundError(f"no accent adapters (*.pt) in {path}")
    return names


def _resolve(model, name: str):
    """(owner, op) for a module name, e.g. ``layers.3.attention.wq``."""
    from zonos3.layers.base import OPList

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
    contiguous rows (zonos3.kernel.triton.expert_lora).
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
    factors: dict[str, list[tuple[int, int, dict]]] = {}
    biases: dict[str, list[tuple[int, torch.Tensor]]] = {}
    experts: dict[str, list[tuple[int, dict]]] = {}
    widths = []
    for index, name in enumerate(names):
        path = Path(directory) / f"{name}.pt"
        adapter = torch.load(path, map_location="cpu", weights_only=True)
        if adapter.get("format") not in FORMATS:
            raise ValueError(f"{path}: not a {' / '.join(FORMATS)} adapter")
        slot = index + 1
        for target, shift in adapter["exact"].items():
            served, zero_mean = served_exact_name(target)
            if served is None:
                continue
            if not served.endswith(".router" + _BALANCING_BIASES):
                raise ValueError(
                    f"{path}: shifts {target}; accent serving applies only linear "
                    "factors and router balancing biases"
                )
            shift = shift.float()
            if zero_mean:
                shift = shift - shift.mean()
            router = served.removesuffix(_BALANCING_BIASES)
            biases.setdefault(router, []).append((slot, shift))
        rank = 0
        for target, pair in adapter["factors"].items():
            if ".experts." in target or "experts" in pair:
                if adapter["format"] != FORMAT_EXPERTS:
                    raise ValueError(f"{path}: {FORMAT} adapter shifts MoE expert weights ({target})")
                experts.setdefault(target, []).append((slot, pair))
                continue
            rank = max(rank, pair["down"].shape[-2])
            factors.setdefault(target, []).append((slot, sum(widths), pair))
        widths.append(rank)

    column_slots = torch.repeat_interleave(
        torch.arange(1, len(names) + 1, dtype=torch.int32), torch.tensor(widths)
    )
    width = int(column_slots.numel())
    ops = {}
    for target, entries in factors.items():
        if not target.endswith(".weight"):
            raise ValueError(f"accent adapter factor {target!r} is not a weight")
        module = target.removesuffix(".weight")
        op = ops[module] = _resolve_linear(model, module)
        weight = op.weight
        out_features, in_features = weight[..., 0].numel(), weight.shape[-1]
        down = torch.zeros(width, in_features)
        up = torch.zeros(width, out_features)
        for slot, start, pair in entries:
            if tuple(pair["shape"]) != tuple(weight.shape):
                raise ValueError(
                    f"{names[slot - 1]}: {target} shape {tuple(pair['shape'])} != served "
                    f"{tuple(weight.shape)}"
                )
            rank = pair["down"].shape[0]
            down[start : start + rank] = pair["down"].float()
            up[start : start + rank] = pair["up"].float().T
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
    model, experts: dict[str, list[tuple[int, dict]]], num_slots: int, device, dtype
) -> dict:
    """Per-expert factors -> an ``ExpertAccents`` on each FusedGroupedExperts."""
    from .zonos3 import FusedGroupedExperts, MoEFeedForward

    layers: dict[str, dict[int, dict[str, dict]]] = {}
    for target, entries in experts.items():
        module, _, weight = target.rpartition(".")
        if not module.endswith("feed_forward.experts") or weight not in EXPERT_WEIGHTS:
            raise ValueError(
                f"accent adapter shifts expert tensor {target!r}; servable expert factors are "
                f"<layer>.feed_forward.experts.{{{','.join(EXPERT_WEIGHTS)}}}"
            )
        for slot, pair in entries:
            layers.setdefault(module, {}).setdefault(slot, {})[weight] = pair

    expert_ops = {}
    for module, per_slot in layers.items():
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
            for weight, pair in weights.items():
                num, out_features, in_features = served[weight]
                rank = pair["down"].shape[-2]
                if (
                    tuple(pair["shape"]) != served[weight]
                    or pair.get("experts") != num
                    or tuple(pair["up"].shape) != (num, out_features, rank)
                    or tuple(pair["down"].shape) != (num, rank, in_features)
                ):
                    raise ValueError(
                        f"slot {slot}: {module}.{weight} factors up {tuple(pair['up'].shape)}, "
                        f"down {tuple(pair['down'].shape)}, shape {tuple(pair['shape'])}, experts "
                        f"{pair.get('experts')} do not shift the served [E, out, in] {served[weight]}"
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
                        down[rows] = pair["down"].transpose(1, 2).to(dtype)
                        up[rows] = pair["up"].transpose(1, 2).to(dtype)
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

    Expert side paths (zonos3.kernel.triton.expert_lora, one program per pair
    and output tile) are probed the same way, 1..256 routed pairs against 1024
    on every slot (each rank's adapters, /1 adapters, no accent); they have no
    fallback and must pass.
    """
    from zonos3.layers.accent import accent_shift, expert_pair_shift

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
