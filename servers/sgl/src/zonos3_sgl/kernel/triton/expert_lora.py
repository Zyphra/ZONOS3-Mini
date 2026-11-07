"""Per-pair low-rank expert shifts for accent adapters (zonos3_sgl.layers.accent).

One program computes one output tile of one routed (token, expert) pair p:

    out[p] = alpha[t] * up[g] @ (down[g] @ rows[p or t]),   g = first[slot[t]] + e_p

in float32 (bf16 loads), rounded once to ``out``'s dtype. ``first[slot]`` is -1
for slots without factors in this rank group: those pairs load nothing and are
written as zeros (``zero_others``) or left alone. Every pair's reduction runs
in a fixed order with fixed block sizes, so its bits do not depend on the
other pairs of the launch (batch-invariant), and the launch has static shapes
and no host syncs (CUDA-graph capturable).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _expert_lora_kernel(
    rows_ptr,
    row_stride,
    tokens_ptr,
    experts_ptr,
    slots_ptr,
    scales_ptr,
    first_ptr,
    down_ptr,
    up_ptr,
    out_ptr,
    out_stride,
    IN: tl.constexpr,
    OUT: tl.constexpr,
    RANK: tl.constexpr,
    RANK_PAD: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    GATHER_ROWS: tl.constexpr,
    ZERO_OTHERS: tl.constexpr,
):
    pair = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    col_mask = cols < OUT
    out_row = out_ptr + pair.to(tl.int64) * out_stride + cols
    token = tl.load(tokens_ptr + pair)
    start = tl.load(first_ptr + tl.load(slots_ptr + token))
    if start < 0:
        # Not on this rank group's adapters: nothing to read.
        if ZERO_OTHERS:
            tl.store(out_row, tl.zeros([BLOCK_OUT], dtype=out_ptr.dtype.element_ty), mask=col_mask)
    else:
        group = (start + tl.load(experts_ptr + pair)).to(tl.int64)
        if GATHER_ROWS:
            row = token.to(tl.int64)
        else:
            row = pair.to(tl.int64)
        ranks = tl.arange(0, RANK_PAD)
        rank_mask = ranks < RANK
        hidden = tl.zeros([RANK_PAD], dtype=tl.float32)
        down = down_ptr + group * (IN * RANK)
        for k0 in range(0, IN, BLOCK_IN):
            k = k0 + tl.arange(0, BLOCK_IN)
            k_mask = k < IN
            x = tl.load(rows_ptr + row * row_stride + k, mask=k_mask, other=0.0).to(tl.float32)
            d = tl.load(
                down + k[:, None] * RANK + ranks[None, :], mask=k_mask[:, None] & rank_mask[None, :], other=0.0
            ).to(tl.float32)
            hidden += tl.sum(d * x[:, None], axis=0)
        hidden = hidden * tl.load(scales_ptr + token).to(tl.float32)
        up = tl.load(
            up_ptr + group * (RANK * OUT) + ranks[:, None] * OUT + cols[None, :],
            mask=rank_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(up * hidden[:, None], axis=0)
        tl.store(out_row, out.to(out_ptr.dtype.element_ty), mask=col_mask)


def expert_lora_shift(
    rows: torch.Tensor,
    tokens: torch.Tensor,
    experts: torch.Tensor,
    slots: torch.Tensor,
    scales: torch.Tensor,
    first: torch.Tensor,
    down: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor,
    *,
    gather_rows: bool,
    zero_others: bool,
) -> None:
    """Write the shift of every pair on this rank group into ``out`` [pairs, out_features].

    rows: [tokens, in] (``gather_rows``: row of pair p is ``tokens[p]``) or
    [pairs, in]; tokens / experts: [pairs] int32, each pair's token and
    expert; slots / scales: [tokens] int32 / float32, the token's adapter slot
    and strength; zero_others: write zeros for pairs off this rank group
    (otherwise leave them alone).
    first: [slots + 1] int64, the slot's first group or -1; down [G, in, rank]
    and up [G, rank, out] contiguous, group ``first[slot] + e``.
    """
    pairs = tokens.shape[0]
    if pairs == 0:
        return
    _, in_features, rank = down.shape
    out_features = up.shape[-1]
    assert rows.stride(-1) == 1 and out.stride(-1) == 1 and down.is_contiguous() and up.is_contiguous()
    # Fixed for every launch (never by pair count): the fastest of a sweep on
    # H100 for the gdn375k shapes, ranks 16 and 64, 64 and 2048 pairs.
    block_out = 256
    grid = (pairs, triton.cdiv(out_features, block_out))
    _expert_lora_kernel[grid](
        rows,
        rows.stride(0),
        tokens,
        experts,
        slots,
        scales,
        first,
        down,
        up,
        out,
        out.stride(0),
        IN=in_features,
        OUT=out_features,
        RANK=rank,
        RANK_PAD=max(16, triton.next_power_of_2(rank)),
        BLOCK_IN=128,
        BLOCK_OUT=block_out,
        GATHER_ROWS=gather_rows,
        ZERO_OTHERS=zero_others,
        num_warps=2,
    )
