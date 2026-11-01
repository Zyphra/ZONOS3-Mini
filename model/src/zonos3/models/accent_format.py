"""Accent adapter files (lingua ``apps.zonos.accent_adapter``, ``zonos-accent-lora/1`` and ``/2``).

An adapter shifts linear weights by ``up @ down``, router balancing biases by a vector and, in ``/2``, every MoE
expert weight by its own ``up_e @ down_e``. A request served with adapter strength alpha runs ``base + alpha *
adapter``. ``load_adapter`` parses one ``<accent>.pt`` (the file stem is the name requests use) into module-keyed
factors; the servers and the reference apply them (zonos3_sgl.models.accent_adapters, zonos3.reference.accents).

Module names are the checkpoint's: ``tools/convert_lingua_checkpoint.py`` keeps every non-expert tensor under its
lingua key. ``TARGETS`` mirrors lingua's ``apps.zonos.lora.DEFAULT_TARGETS``. Exact shifts: ``<router>.balancing_biases``
(float32 quantile thresholds); as in the converter, the legacy ZeroMean parametrization
``.parametrizations.balancing_biases.original`` maps to ``balancing_biases`` with its mean removed, and the training-only
router metrics (``ent_denom``, ``normalized_entropy``) are dropped. Any other exact shift is refused.

Expert factors (``/2``): ``layers.N.feed_forward.experts.{w13,w2}`` carry ``up`` [E, out, r] and ``down`` [E, r, in] in
lingua's SonicExperts layout (w13 rows interleave gate/up), ``shape`` (E, out, in) and ``experts`` E.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

FORMAT = "zonos-accent-lora/1"
FORMAT_EXPERTS = "zonos-accent-lora/2"
FORMATS = (FORMAT, FORMAT_EXPERTS)
# SonicExperts weights lingua's apps.zonos.lora adapts per expert.
EXPERT_WEIGHTS = ("w13", "w2")

BALANCING_BIASES = ".balancing_biases"
_LEGACY_BALANCING_BIASES = ".parametrizations.balancing_biases.original"
# Router state the converter drops (training metrics, never used for routing).
_DROPPED_ROUTER_STATE = (".router.ent_denom", ".router.normalized_entropy")

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
        return name.removesuffix(_LEGACY_BALANCING_BIASES) + BALANCING_BIASES, True
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


@dataclass(frozen=True)
class AccentAdapter:
    """One parsed adapter.

    ``linear``: LoRA module (e.g. ``layers.3.attention.wq``) -> (down [K, in], up [K, out]) float32, the shift being
    ``(x @ down.T) @ up`` on the flat ``[out, in]`` product; ``shapes``: the weight shape each linear shift was trained
    on. ``experts``: experts module (``layers.N.feed_forward.experts``) -> weight (``w13`` / ``w2``) -> (up [E, out, r],
    down [E, r, in]) float32 in the SonicExperts layout. ``router_bias``: router module (``layers.N.feed_forward.router``)
    -> float32 [experts] added to its balancing biases."""

    name: str
    format: str
    linear: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    experts: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    router_bias: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def rank(self) -> int:
        """Columns the adapter's linear factors take (their largest rank)."""
        return max((down.shape[0] for down, _ in self.linear.values()), default=0)


def load_adapter(path: str | Path) -> AccentAdapter:
    """Parse ``<accent>.pt``; raises ValueError for a file no server can apply."""
    path = Path(path)
    adapter = torch.load(path, map_location="cpu", weights_only=True)
    fmt = adapter.get("format")
    if fmt not in FORMATS:
        raise ValueError(f"{path}: not a {' / '.join(FORMATS)} adapter")
    router_bias: dict[str, torch.Tensor] = {}
    for target, shift in adapter["exact"].items():
        served, zero_mean = served_exact_name(target)
        if served is None:
            continue
        if not served.endswith(".router" + BALANCING_BIASES):
            raise ValueError(
                f"{path}: shifts {target}; accent serving applies only linear "
                "factors and router balancing biases"
            )
        shift = shift.float()
        if zero_mean:
            shift = shift - shift.mean()
        router_bias[served.removesuffix(BALANCING_BIASES)] = shift
    linear: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    experts: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for target, pair in adapter["factors"].items():
        if ".experts." in target or "experts" in pair:
            if fmt != FORMAT_EXPERTS:
                raise ValueError(f"{path}: {FORMAT} adapter shifts MoE expert weights ({target})")
            module, _, weight = target.rpartition(".")
            if not module.endswith("feed_forward.experts") or weight not in EXPERT_WEIGHTS:
                raise ValueError(
                    f"accent adapter shifts expert tensor {target!r}; servable expert factors are "
                    f"<layer>.feed_forward.experts.{{{','.join(EXPERT_WEIGHTS)}}}"
                )
            up, down = pair["up"], pair["down"]
            shape = tuple(pair["shape"])
            if (
                len(shape) != 3
                or pair.get("experts") != shape[0]
                or tuple(up.shape) != (shape[0], shape[1], down.shape[-2])
                or tuple(down.shape) != (shape[0], down.shape[-2], shape[2])
            ):
                raise ValueError(
                    f"{path.stem}: {target} factors up {tuple(up.shape)}, down {tuple(down.shape)}, shape {shape}, "
                    f"experts {pair.get('experts')} do not shift the [E, out, in] weight {shape}"
                )
            experts.setdefault(module, {})[weight] = (up.float(), down.float())
            continue
        if not target.endswith(".weight"):
            raise ValueError(f"accent adapter factor {target!r} is not a weight")
        module = target.removesuffix(".weight")
        if lora_target(module) is None:
            raise ValueError(
                f"accent adapter shifts {module!r}, outside the LoRA targets {', '.join(TARGETS)}"
            )
        linear[module] = (pair["down"].float(), pair["up"].float().T)
        shapes[module] = tuple(pair["shape"])
    return AccentAdapter(path.stem, fmt, linear, shapes, experts, router_bias)
