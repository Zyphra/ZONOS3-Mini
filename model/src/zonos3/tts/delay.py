"""Delay pattern of the codebook rows."""

from __future__ import annotations

import torch


def shear_up(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Remove delay: column j shifted up by j rows.

    This is the inverse of shear() - it removes the delay pattern applied
    during generation to align all codebook outputs for decoding.

    Args:
        x: Input tensor of shape (..., H, W) where H is frames and W is codebooks
        pad_id: Padding value to fill empty positions

    Returns:
        De-sheared tensor of same shape
    """
    H, W = x.shape[-2:]
    out = x.new_full(x.shape, pad_id)
    for j in range(W):
        if H > j:
            out[..., : H - j, j] = x[..., j:, j]
    return out
