"""Numeric helpers shared by the MLX models."""

from __future__ import annotations

import mlx.core as mx


def bf16_round(x: mx.array) -> mx.array:
    """A BF16 tensor boundary of the autocast recipes: round to bfloat16, keep computing in float32."""
    return x.astype(mx.bfloat16).astype(mx.float32)
