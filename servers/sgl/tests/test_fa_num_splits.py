"""GDN-family attention runs flash-attention unsplit, so cached decode equals the packed (training) arithmetic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from zonos3_sgl.attention.fa import fa_num_splits


def test_gdn_family_runs_unsplit_and_legacy_keeps_the_heuristic(monkeypatch):
    monkeypatch.delenv("ZONOS3_FA_NUM_SPLITS", raising=False)
    assert fa_num_splits(SimpleNamespace(attention_pattern="gdn3_global1")) == 1
    assert fa_num_splits(SimpleNamespace(attention_pattern="global")) == 0
    assert fa_num_splits(SimpleNamespace()) == 0
    monkeypatch.setenv("ZONOS3_FA_NUM_SPLITS", "4")
    assert fa_num_splits(SimpleNamespace(attention_pattern="gdn3_global1")) == 4
    monkeypatch.setenv("ZONOS3_FA_NUM_SPLITS", "0")
    assert fa_num_splits(SimpleNamespace(attention_pattern="gdn3_global1")) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FA3 kernel")
def test_unsplit_decode_matches_unsplit_prefill_on_long_caches():
    """The property the parity receipts rely on: for KV lengths past 512 rows, one-row decode with num_splits=1 returns
    the same bits as the last row of the full prefill, at batch 1 and batch 4 (the auto heuristic does not)."""
    from zonos3_sgl.attention.fa import _fa_sgl_impl

    device = torch.device("cuda")
    Hq, Hk, Dh = 16, 4, 32
    for n in (640, 1398):
        g = torch.Generator(device=device).manual_seed(n)
        q_all = F.rms_norm(torch.randn(n, Hq, Dh, device=device, generator=g), (Dh,)).to(torch.bfloat16)
        k_all = F.rms_norm(torch.randn(n, Hk, Dh, device=device, generator=g), (Dh,)).to(torch.bfloat16)
        v_all = (torch.randn(n, Hk, Dh, device=device, generator=g) * 0.5).to(torch.bfloat16)
        B = 4
        k_cache = torch.zeros(B * n + 1, 1, Hk, Dh, device=device, dtype=torch.bfloat16)
        v_cache = torch.zeros_like(k_cache)
        for s in range(B):
            k_cache[s * n : (s + 1) * n, 0] = k_all
            v_cache[s * n : (s + 1) * n, 0] = v_all
        page_table = torch.arange(B * n, device=device, dtype=torch.int32).view(B, n)
        cu = torch.tensor([0, n], device=device, dtype=torch.int32)
        prefill = _fa_sgl_impl(
            q=q_all,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table[:1],
            cache_seqlens=torch.tensor([n], device=device, dtype=torch.int32),
            cu_seqlens_q=cu,
            cu_seqlens_k_new=cu,
            max_seqlen_q=n,
            softmax_scale=Dh**-0.5,
            num_splits=1,
        )
        for b in (1, 4):
            q = q_all[-1:].expand(b, Hq, Dh).contiguous()
            cu_q = torch.arange(0, b + 1, device=device, dtype=torch.int32)
            decode = _fa_sgl_impl(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table[:b],
                cache_seqlens=torch.full((b,), n, device=device, dtype=torch.int32),
                cu_seqlens_q=cu_q,
                cu_seqlens_k_new=cu_q * n,
                max_seqlen_q=1,
                softmax_scale=Dh**-0.5,
                num_splits=1,
            )
            assert torch.equal(decode[0], prefill[-1]), (n, b)
