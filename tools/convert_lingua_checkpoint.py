#!/usr/bin/env python3
"""Convert a Lingua distributed checkpoint into a minisgl checkpoint.

The converter reads only the model tensors from a PyTorch Distributed
Checkpoint (DCP). Optimizer, dataloader, and training state are intentionally
excluded. The source checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "checkpoints"
CHECKPOINT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

OUTPUT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# Fields understood by zonos3.utils.hf.Zonos3Config. Keeping params.json
# deliberately small makes release checkpoints independent of training-only
# configuration.
MODEL_CONFIG_FIELDS = {
    "n_layers",
    "dim",
    "head_dim",
    "n_heads",
    "n_kv_heads",
    "use_xsa",
    "attention_pattern",
    "residual_type",
    "block_attnres_block_size",
    "block_attnres_backend",
    "ffn_dim_multiplier",
    "multiple_of",
    "moe_ffn_dim_multiplier",
    "moe_multiple_of",
    "norm_eps",
    "rope_theta",
    "max_seqlen",
    "n_codebooks",
    "codebook_size",
    "codebook_sizes",
    "audio_codec",
    "eoa_id",
    "audio_pad_id",
    "span_end_id",
    "text_vocab",
    "text_extra_tokens",
    "speaker_enabled",
    "speaker_reference_prefix_enabled",
    "speaker_reference_max_seconds",
    "speaker_embedding_dim",
    "speaker_embedding_codec",
    "speaker_lda_dim",
    "speaker_background_token_enabled",
    "accurate_mode_token_enabled",
    "speaking_rate_num_buckets",
    "speaking_rate_buckets",
    "quality_num_buckets",
    "quality_features",
    "quality_buckets",
    "quality_dropout",
    "condition_tokens_enabled",
    "condition_n_buckets",
    "condition_count",
    "condition_names",
    "condition_edges",
    "condition_units",
    "moe_router_topk",
    "norm_topk_prob",
    "special_topk_layers",
    "moe_n_experts",
    "moe_alpha",
    "moe_router_dim",
    "moe_router_type",
    "moe_router_use_eda",
    "moe_balancing_strategy",
    "moe_start_from_layer",
    "moe_end_from_layer",
    "loss_softcap",
}


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return -1, path.name


def resolve_checkpoint(path: Path) -> Path:
    """Resolve a checkpoint directory or select the latest complete child."""
    path = path.expanduser().resolve()
    if (path / ".metadata").is_file():
        return path

    checkpoint_root = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    complete = [child for child in checkpoint_root.iterdir() if child.is_dir() and (child / ".metadata").is_file()]
    if not complete:
        raise FileNotFoundError(f"No complete distributed checkpoints found under {checkpoint_root}")
    return max(complete, key=_checkpoint_sort_key)


def resolve_training_config(checkpoint: Path, explicit_config: Path | None) -> Path:
    """Find the training config associated with a DCP checkpoint."""
    if explicit_config is not None:
        config = explicit_config.expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(f"Training config does not exist: {config}")
        return config

    candidates = [
        checkpoint / "config.yaml",
        checkpoint.parent / "config.yaml",
        checkpoint.parent.parent / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not infer config.yaml from the checkpoint. Pass it explicitly with --config.")


def default_checkpoint_name(checkpoint: Path) -> str:
    run_name = checkpoint.parent.parent.name if checkpoint.parent.name == "checkpoints" else "model"
    raw_name = f"{run_name}-step-{checkpoint.name}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip("-.")


def validate_checkpoint_name(name: str) -> str:
    if not CHECKPOINT_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"Checkpoint name must be one path component containing only letters, numbers, '.', '_', or '-': {name!r}"
        )
    return name


def load_training_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in training config: {config_path}")
    model = config.get("model", config)
    if not isinstance(model, dict):
        raise ValueError(f"Expected a model mapping in training config: {config_path}")
    return config


def merge_checkpoint_model_params(training_config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Overlay the resolved model defaults persisted by the training job.

    YAML configs commonly omit dataclass defaults such as ``text_vocab``.
    Those defaults still affect checkpoint tensor shapes, so inference config
    must prefer the fully resolved ``params.json`` saved beside the DCP.
    """
    params_path = checkpoint / "params.json"
    if not params_path.is_file():
        return training_config
    payload = json.loads(params_path.read_text())
    resolved_model = payload.get("model")
    if not isinstance(resolved_model, dict):
        raise ValueError(f"Expected a model mapping in {params_path}")

    merged_config = dict(training_config)
    yaml_model = training_config.get("model", training_config)
    if not isinstance(yaml_model, dict):
        raise ValueError("Expected a model mapping in the training config")
    merged_model = dict(yaml_model)
    merged_model.update(
        {key: value for key, value in resolved_model.items() if key in MODEL_CONFIG_FIELDS and value is not None}
    )
    merged_config["model"] = merged_model
    return merged_config


# Training-only knobs the runtime hard-codes. Every supported value is listed; anything else is refused loudly.
RUNTIME_ASSUMED_MODEL_FIELDS = {
    "qwen_gating_style": {"headwise"},
    "embedding_norm": {True},
    "attn_out_norm": {False},
    "ffn_out_norm": {False},
    "sliding_window": {-1},
    "flash_msa_every_n_layers": {0},
    "noise_codebook_size": {None},
    "moe_qb_ema": {0.0},
}


def check_runtime_assumptions(model_config: dict[str, Any]) -> None:
    for key, allowed in RUNTIME_ASSUMED_MODEL_FIELDS.items():
        if key in model_config and model_config[key] not in allowed:
            raise ValueError(
                f"model.{key}={model_config[key]!r} is not implemented by the inference runtime "
                f"(supported: {sorted(map(str, allowed))}); refusing to convert a checkpoint it would run wrongly."
            )
    if model_config.get("codebook_sizes") is not None:
        sizes = [int(v) for v in model_config["codebook_sizes"]]
        if len(sizes) != int(model_config.get("n_codebooks", 0)) or max(sizes) != int(
            model_config.get("codebook_size", 0)
        ):
            raise ValueError("codebook_sizes must have one entry per codebook with the largest equal to codebook_size")


def build_params(training_config: dict[str, Any], dtype_name: str) -> dict[str, Any]:
    model_config = training_config.get("model", training_config)
    check_runtime_assumptions(model_config)
    v = model_config.get("text_conditioning")
    if v is not None and v != "bytes":
        raise ValueError(
            f"ZONOS3 converts byte / SuperBPE text checkpoints only; this run uses text_conditioning={v!r}"
        )
    params = {key: value for key, value in model_config.items() if key in MODEL_CONFIG_FIELDS and value is not None}
    if "moe_router_use_eda" not in params and "use_eda" in model_config:
        params["moe_router_use_eda"] = bool(model_config["use_eda"])

    raw_balancing_strategy = str(model_config.get("moe_balancing_strategy", "legacy")).strip().lower().replace("-", "_")
    strategy_aliases = {
        "current": "quantile",
        "quantile": "quantile",
        "qbalancing": "quantile",
        "megatron": "quantile",
        "megatron_qbalancing": "quantile",
        "megatron_quantile": "quantile",
        "quantile_balancing": "quantile",
        "old": "legacy",
        "legacy": "legacy",
        "aux": "legacy",
        "aux_loss": "legacy",
    }
    if raw_balancing_strategy not in strategy_aliases:
        raise ValueError(f"Unsupported model.moe_balancing_strategy={raw_balancing_strategy!r}.")
    data_config = training_config.get("data", {})
    if data_config.get("text_tokenizer") == "superbpe":
        spec = {
            "kind": "superbpe",
            "path": data_config["text_tokenizer_path"],
            "sha256": data_config["text_tokenizer_sha256"],
            "vocab_size": data_config["text_tokenizer_vocab_size"],
            "offset": 4,
            "bos": 2,
            "eos": 3,
        }
        from zonos3.tts.superbpe import validate_spec

        validate_spec(spec, int(params["text_vocab"]))
        params["text_tokenizer_spec"] = spec
    if isinstance(data_config, dict):
        raw_audio_codec = data_config.get("codec")
        audio_codec = str(raw_audio_codec) if raw_audio_codec is not None else None
        if audio_codec is not None:
            params["audio_codec"] = audio_codec
            from zonos3.tokenizer.matched_codec import codec_names

            if audio_codec not in codec_names():
                raise ValueError(
                    f"Unsupported data.codec={audio_codec!r}: ZONOS3 converts checkpoints of the codecs in "
                    f"registries/codec_registry.json {sorted(codec_names())}"
                )
        speaker_codec = data_config.get("speaker_embedding_codec")
        if speaker_codec and params.get("speaker_enabled"):
            params["speaker_embedding_codec"] = str(speaker_codec)
        if data_config.get("speaker_reference_index_path") or any(
            isinstance(source, dict) and source.get("speaker_reference_index")
            for source in (data_config.get("sources") or {}).values()
        ):
            # Reference prefixes may be configured per compiled source (the GDN / Full-AttnRes family).
            params["speaker_reference_prefix_enabled"] = True
            params["speaker_reference_max_seconds"] = float(data_config.get("speaker_reference_max_seconds", 30.0))
        if params.get("condition_tokens_enabled"):
            # The cond3 finetunes: the runtime needs the bucket edges to map raw values (LUFS / Hz / bytes per
            # second) to the ids the tables were trained with; the loader's CONDITION_NAMES order is fixed.
            names = ["loudness", "fmax", "speaking_rate"]
            edges_path = data_config.get("condition_edges_path")
            if not edges_path or not Path(edges_path).is_file():
                raise ValueError(
                    "model.condition_tokens_enabled needs data.condition_edges_path (the bucket edges file)"
                )
            spec = json.loads(Path(edges_path).read_text())
            n_buckets = int(params.get("condition_n_buckets", 10))
            if (
                int(data_config.get("condition_n_buckets", n_buckets)) != n_buckets
                or int(spec.get("n_buckets", n_buckets)) != n_buckets
            ):
                raise ValueError("condition_n_buckets differs between model, data and the edges file")
            count = int(params.get("condition_count", len(names)))
            # The encode-start twin (2026-10-16): a fourth row, a flag (0 = mid-file crop, 1 = genuine encode start,
            # n_buckets = unknown) whose table has the same shape; the training config marks it with
            # data.condition_encode_start_max_frames. No edges: requests send the id (condition_buckets) or a boolean.
            flags = {}
            if count == len(names) + 1 and "condition_encode_start_max_frames" in data_config:
                names = names + ["encode_start"]
                flags["encode_start"] = {
                    "values": [0, 1],
                    "labels": {"0": "mid-file crop", "1": "genuine encode start"},
                    "max_frames": int(data_config.get("condition_encode_start_max_frames") or 0),
                    "note": "1 = the generation starts at frame 0 of a file's encode (the codec's cold start), "
                    "0 = a mid-file continuation, unknown = the model decides (20% of training draws)",
                }
            if count != len(names):
                raise ValueError(
                    f"condition_count={count} is not implemented by the runtime (expected {len(names)}: {names})"
                )
            edges = {}
            for name in names:
                if name in flags:
                    continue
                values = [float(x) for x in spec["edges"][name]]
                if len(values) != n_buckets - 1 or any(b < a for a, b in zip(values, values[1:])):
                    raise ValueError(f"{edges_path}: {name} needs {n_buckets - 1} sorted edges")
                edges[name] = values
            params["condition_count"] = count
            params["condition_n_buckets"] = n_buckets
            params["condition_names"] = names
            params["condition_edges"] = edges
            params["condition_units"] = {"loudness": "LUFS", "fmax": "Hz", "speaking_rate": "utf-8 bytes per second"}
            if flags:
                params["condition_units"]["encode_start"] = "flag"
            params["condition_flags"] = flags
    params.update(
        {
            "model_type": "zonos2",
            "dtype": dtype_name,
            # Lingua renormalizes the selected weights whenever top-k > 1.
            "norm_topk_prob": int(params.get("moe_router_topk", 1)) > 1,
            # Training uses Sonic MoE, while minisgl converts the weights into
            # its fused inference representation.
            "moe_impl": "sonic",
            # Older configs omitted this field and used the learned +bias rule;
            # newer q-balanced checkpoints persist thresholds used with -bias.
            "moe_balancing_strategy": strategy_aliases[raw_balancing_strategy],
        }
    )
    return {"model": params}


def allocate_model_state(
    checkpoint: Path,
) -> tuple[dict[str, torch.Tensor], int]:
    """Allocate CPU tensors matching only the model entries in DCP metadata."""
    metadata = FileSystemReader(checkpoint).read_metadata().state_dict_metadata
    model_metadata = {
        key.removeprefix("model."): value
        for key, value in metadata.items()
        if key.startswith("model.") and isinstance(value, TensorStorageMetadata)
    }
    if not model_metadata:
        raise ValueError(f"No model tensors found in distributed checkpoint: {checkpoint}")

    source_bytes = 0
    model_state: dict[str, torch.Tensor] = {}
    for key, value in model_metadata.items():
        tensor = torch.empty(tuple(value.size), dtype=value.properties.dtype, device="cpu")
        model_state[key] = tensor
        source_bytes += tensor.numel() * tensor.element_size()
    return model_state, source_bytes


def load_model_state(checkpoint: Path) -> tuple[dict[str, torch.Tensor], int]:
    model_state, source_bytes = allocate_model_state(checkpoint)
    dcp.load({"model": model_state}, checkpoint_id=checkpoint)
    return model_state, source_bytes


def _output_key(key: str) -> str | None:
    if ".router.ent_denom" in key or ".router.normalized_entropy" in key:
        return None
    if ".parametrizations.balancing_biases.original" in key:
        return key.replace(
            ".parametrizations.balancing_biases.original",
            ".balancing_biases",
        )
    if key.endswith(".feed_forward.experts.w13"):
        return key.removesuffix("w13") + "gate_up_proj"
    if key.endswith(".feed_forward.experts.w2"):
        return key.removesuffix("w2") + "down_proj"
    return key


def convert_model_state(
    model_state: dict[str, torch.Tensor],
    output_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Normalize Lingua model tensors into minisgl's native state dict."""
    converted: dict[str, torch.Tensor] = {}
    stats = {
        "source_tensors": len(model_state),
        "output_tensors": 0,
        "dropped_training_tensors": 0,
        "sonic_expert_tensors": 0,
        "balancing_bias_tensors": 0,
    }

    for source_key in sorted(model_state):
        output_key = _output_key(source_key)
        if output_key is None:
            stats["dropped_training_tensors"] += 1
            continue

        tensor = model_state[source_key]

        if ".parametrizations.balancing_biases.original" in source_key:
            # Legacy Lingua applies ZeroMean as a float32 parametrization. Save
            # the effective value used for routing, not the unconstrained value.
            tensor = tensor.float()
            tensor = tensor - tensor.mean()
            stats["balancing_bias_tensors"] += 1
        elif source_key.endswith(".router.balancing_biases"):
            # Q-balancing checkpoints already contain the effective thresholds.
            # Preserve float32 because small threshold changes affect top-k ties.
            tensor = tensor.float()
            stats["balancing_bias_tensors"] += 1
        else:
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=output_dtype)

        if source_key.endswith(".feed_forward.experts.w13"):
            if tensor.ndim != 3 or tensor.shape[1] % 2:
                raise ValueError(
                    f"Expected Sonic w13 with shape [experts, 2 * intermediate, hidden], "
                    f"got {source_key}: {tuple(tensor.shape)}"
                )
            # Sonic stores gate/up rows interleaved. minisgl's fused kernel
            # expects all gate rows followed by all up rows.
            tensor = torch.cat((tensor[:, 0::2, :], tensor[:, 1::2, :]), dim=1)
            stats["sonic_expert_tensors"] += 1
        elif source_key.endswith(".feed_forward.experts.w2"):
            stats["sonic_expert_tensors"] += 1

        if output_key in converted:
            raise ValueError(f"Multiple source tensors map to output key: {output_key}")
        converted[output_key] = tensor.contiguous()

    stats["output_tensors"] = len(converted)
    return converted, stats


def tensor_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())


def convert_checkpoint(
    checkpoint_input: Path,
    output_root: Path,
    name: str | None,
    config_input: Path | None,
    dtype_name: str,
) -> Path:
    checkpoint = resolve_checkpoint(checkpoint_input)
    config_path = resolve_training_config(checkpoint, config_input)
    training_config = load_training_config(config_path)
    training_config = merge_checkpoint_model_params(training_config, checkpoint)

    checkpoint_name = validate_checkpoint_name(name or default_checkpoint_name(checkpoint))
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / checkpoint_name
    if output_path.exists():
        raise FileExistsError(
            f"Output checkpoint already exists: {output_path}. Choose a different --name or remove it explicitly."
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{checkpoint_name}.", dir=output_root))
    try:
        print(f"Loading model tensors from {checkpoint}")
        model_state, source_bytes = load_model_state(checkpoint)
        print(f"Loaded {len(model_state)} model tensors ({source_bytes / 2**30:.2f} GiB); converting to {dtype_name}")

        converted, stats = convert_model_state(model_state, OUTPUT_DTYPES[dtype_name])
        del model_state

        model_path = staging / "model.pth"
        torch.save(converted, model_path)
        params = build_params(training_config, dtype_name)
        (staging / "params.json").write_text(json.dumps(params, indent=2) + "\n")
        shutil.copy2(config_path, staging / "config.yaml")

        manifest = {
            "format": "minisgl-zonos2",
            "source_checkpoint": str(checkpoint),
            "source_config": str(config_path),
            "source_checkpoint_params": str(checkpoint / "params.json")
            if (checkpoint / "params.json").is_file()
            else None,
            "checkpoint_step": int(checkpoint.name) if checkpoint.name.isdigit() else None,
            "output_dtype": dtype_name,
            "source_model_bytes": source_bytes,
            "output_model_bytes": tensor_bytes(converted),
            "conversion": stats,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")

        staging.rename(output_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Converted checkpoint written to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Lingua DCP checkpoint, its checkpoints directory, or a training run "
            "directory. A parent directory selects its latest complete checkpoint."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training config.yaml. Inferred from the checkpoint layout by default.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output directory name under checkpoints/. Defaults to run-name plus step.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Converted checkpoint root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(OUTPUT_DTYPES),
        default="bfloat16",
        help="Floating-point dtype for converted weights (default: bfloat16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_checkpoint(
        checkpoint_input=args.checkpoint,
        output_root=args.output_root,
        name=args.name,
        config_input=args.config,
        dtype_name=args.dtype,
    )


if __name__ == "__main__":
    main()
