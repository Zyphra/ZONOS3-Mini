"""The converter refuses training-only knobs the runtime hard-codes (frozen GDN / Full-AttnRes family, 2026-10)."""
from __future__ import annotations

import pytest
from convert_lingua_checkpoint import RUNTIME_ASSUMED_MODEL_FIELDS, check_runtime_assumptions

FROZEN_FAMILY = {
    "dim": 1024,
    "n_layers": 11,
    "attention_pattern": "gdn3_global1",
    "residual_type": "full_attnres",
    "qwen_gating_style": "headwise",
    "embedding_norm": True,
    "attn_out_norm": False,
    "ffn_out_norm": False,
    "sliding_window": -1,
    "flash_msa_every_n_layers": 0,
    "noise_codebook_size": None,
    "moe_qb_ema": 0.0,
    "n_codebooks": 10,
    "codebook_size": 4096,
    "codebook_sizes": [4096] * 10,
}


def test_frozen_family_config_passes_the_guard():
    check_runtime_assumptions(dict(FROZEN_FAMILY))


def test_absent_keys_mean_the_training_defaults_the_runtime_assumes():
    check_runtime_assumptions({"dim": 1024, "n_codebooks": 9, "codebook_size": 1024})


@pytest.mark.parametrize(
    "key, value",
    [
        ("qwen_gating_style", "global"),
        ("embedding_norm", False),
        ("attn_out_norm", True),
        ("ffn_out_norm", True),
        ("sliding_window", 4096),
        ("flash_msa_every_n_layers", 2),
        ("noise_codebook_size", 64),
        ("moe_qb_ema", 0.01),
    ],
)
def test_unsupported_training_knob_is_refused(key, value):
    with pytest.raises(ValueError):
        check_runtime_assumptions({**FROZEN_FAMILY, key: value})


@pytest.mark.parametrize("sizes", [[4096] * 9, [2048] * 10, [4096] * 9 + [8192]])
def test_codebook_sizes_must_match_the_codebook_layout(sizes):
    with pytest.raises(ValueError):
        check_runtime_assumptions({**FROZEN_FAMILY, "codebook_sizes": sizes})


def test_every_guarded_field_lists_its_supported_values():
    assert RUNTIME_ASSUMED_MODEL_FIELDS
    assert all(allowed for allowed in RUNTIME_ASSUMED_MODEL_FIELDS.values())
