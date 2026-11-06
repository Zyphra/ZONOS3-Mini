from pathlib import Path

import pytest
import torch
from zonos3.models.config import ModelConfig
from zonos3_sgl.models.weight import _normalize_zonos3_state_dict
from zonos3.utils.hf import Zonos3Config

from convert_lingua_checkpoint import (
    build_params,
    convert_model_state,
    resolve_checkpoint,
    validate_checkpoint_name,
)


def test_weight_loader_materializes_effective_zero_mean_balancing_bias():
    state = {
        "layers.3.feed_forward.router.parametrizations.balancing_biases.original": (
            torch.tensor([1.0, 2.0, 6.0])
        )
    }

    normalized = _normalize_zonos3_state_dict(state)

    assert set(normalized) == {"layers.3.feed_forward.router.balancing_biases"}
    bias = normalized["layers.3.feed_forward.router.balancing_biases"]
    torch.testing.assert_close(bias, torch.tensor([-2.0, -1.0, 3.0]))


def test_convert_model_state_normalizes_training_and_sonic_keys():
    state = {
        "layers.3.feed_forward.experts.w13": torch.arange(4.0).reshape(1, 4, 1),
        "layers.3.feed_forward.experts.w2": torch.ones(1, 1, 2),
        "layers.3.feed_forward.router.parametrizations.balancing_biases.original": (
            torch.tensor([1.0, 2.0, 6.0])
        ),
        "layers.3.feed_forward.router.ent_denom": torch.tensor(2.0),
        "out_norm.weight": torch.ones(2),
    }

    converted, stats = convert_model_state(state, torch.bfloat16)

    assert set(converted) == {
        "layers.3.feed_forward.experts.gate_up_proj",
        "layers.3.feed_forward.experts.down_proj",
        "layers.3.feed_forward.router.balancing_biases",
        "out_norm.weight",
    }
    assert converted["layers.3.feed_forward.experts.gate_up_proj"].flatten().tolist() == [
        0.0,
        2.0,
        1.0,
        3.0,
    ]
    assert converted["layers.3.feed_forward.router.balancing_biases"].float().sum().item() == 0.0
    assert converted["layers.3.feed_forward.router.balancing_biases"].dtype == torch.float32
    assert converted["layers.3.feed_forward.experts.gate_up_proj"].dtype == torch.bfloat16
    assert stats == {
        "source_tensors": 5,
        "output_tensors": 4,
        "dropped_training_tensors": 1,
        "sonic_expert_tensors": 2,
        "balancing_bias_tensors": 1,
    }


def test_build_params_preserves_architecture_and_sets_inference_semantics():
    params = build_params(
        {
            "model": {
                "n_layers": 28,
                "dim": 1024,
                "moe_n_experts": 8,
                "moe_router_topk": 2,
                "moe_ffn_dim_multiplier": 0.125,
                "moe_multiple_of": 64,
                "seed": 333,
            }
        },
        "bfloat16",
    )

    assert params == {
        "model": {
            "n_layers": 28,
            "dim": 1024,
            "moe_n_experts": 8,
            "moe_router_topk": 2,
            "moe_ffn_dim_multiplier": 0.125,
            "moe_multiple_of": 64,
            "model_type": "zonos2",
            "dtype": "bfloat16",
            "norm_topk_prob": True,
            "moe_impl": "sonic",
            "moe_balancing_strategy": "legacy",
        }
    }


def test_build_params_preserves_data_speaker_embedding_codec():
    params = build_params(
        {
            "data": {"speaker_embedding_codec": "w2vbert_sv"},
            "model": {
                "n_layers": 11,
                "dim": 1024,
                "speaker_enabled": True,
                "speaker_embedding_dim": 256,
            },
        },
        "bfloat16",
    )

    assert params["model"]["speaker_embedding_codec"] == "w2vbert_sv"


def test_build_params_keeps_span_end_id():
    params = build_params(
        {"model": {"n_layers": 2, "dim": 256, "span_end_id": 4098, "text_extra_tokens": 3}},
        "bfloat16",
    )

    assert params["model"]["span_end_id"] == 4098
    assert params["model"]["text_extra_tokens"] == 3


def test_span_end_id_grows_audio_vocab_by_one_per_codebook():
    common = dict(n_codebooks=10, codebook_size=4096, eoa_id=4096, audio_pad_id=4097, text_vocab=8196)
    base = ModelConfig.from_zonos3_config(Zonos3Config(**common))
    edit = ModelConfig.from_zonos3_config(Zonos3Config(**common, span_end_id=4098))

    assert base.span_end_id is None
    assert base.audio_vocab_size == 4098
    assert edit.span_end_id == 4098
    assert edit.audio_vocab_size == base.audio_vocab_size + 1 == 4099
    assert edit.vocab_size == base.vocab_size + 10


@pytest.mark.parametrize("span_end_id", [4096, 4097, 4099])
def test_span_end_id_must_follow_audio_pad(span_end_id):
    with pytest.raises(ValueError):
        ModelConfig.from_zonos3_config(
            Zonos3Config(n_codebooks=10, codebook_size=4096, eoa_id=4096, audio_pad_id=4097, span_end_id=span_end_id)
        )


def test_model_config_uses_independent_moe_intermediate_size():
    config = Zonos3Config(
        dim=768,
        head_dim=64,
        n_heads=12,
        n_kv_heads=3,
        ffn_dim_multiplier=0.75,
        multiple_of=64,
        moe_n_experts=256,
        moe_router_topk=8,
        moe_ffn_dim_multiplier=1 / 12,
        moe_multiple_of=64,
    )

    model_config = ModelConfig.from_zonos3_config(config)

    assert model_config.intermediate_size == 576
    assert model_config.moe_intermediate_size == 64


def test_model_config_falls_back_to_dense_intermediate_size_for_old_checkpoints():
    config = Zonos3Config(
        dim=384,
        head_dim=64,
        ffn_dim_multiplier=0.5,
        multiple_of=256,
        moe_n_experts=64,
        moe_router_topk=2,
    )

    model_config = ModelConfig.from_zonos3_config(config)

    assert model_config.intermediate_size == 256
    assert model_config.moe_intermediate_size == 256


def test_resolve_checkpoint_selects_latest_complete_checkpoint(tmp_path: Path):
    checkpoint_root = tmp_path / "checkpoints"
    incomplete = checkpoint_root / "0000000300"
    earlier = checkpoint_root / "0000000100"
    latest = checkpoint_root / "0000000200"
    incomplete.mkdir(parents=True)
    earlier.mkdir()
    latest.mkdir()
    (earlier / ".metadata").touch()
    (latest / ".metadata").touch()

    assert resolve_checkpoint(tmp_path) == latest
    assert resolve_checkpoint(checkpoint_root) == latest
    assert resolve_checkpoint(earlier) == earlier


@pytest.mark.parametrize("name", ["../escape", "nested/name", "", ".hidden"])
def test_validate_checkpoint_name_rejects_unsafe_names(name: str):
    with pytest.raises(ValueError):
        validate_checkpoint_name(name)
