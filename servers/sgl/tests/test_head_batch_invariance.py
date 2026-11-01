"""The multi-codebook LM head gives a request the same logits whether it shares the GEMM with 0 or 1000 other rows."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from zonos3_sgl.models.zonos3 import MultiOutputHead, head_min_rows


def test_gdn_family_pads_the_head_to_two_rows(monkeypatch):
    monkeypatch.delenv("ZONOS3_HEAD_MIN_ROWS", raising=False)
    assert head_min_rows(SimpleNamespace(attention_pattern="gdn3_global1")) == 2
    assert head_min_rows(SimpleNamespace(attention_pattern="global")) == 1
    assert head_min_rows(SimpleNamespace()) == 1  # legacy configs without the field keep their bits
    monkeypatch.setenv("ZONOS3_HEAD_MIN_ROWS", "8")
    assert head_min_rows(SimpleNamespace(attention_pattern="global")) == 8
    monkeypatch.setenv("ZONOS3_HEAD_MIN_ROWS", "1")
    assert head_min_rows(SimpleNamespace(attention_pattern="gdn3_global1")) == 1


def test_padding_only_adds_rows_and_slices_them_back():
    head = MultiOutputHead(16, 24, min_rows=2)
    head.weight = torch.randn(24, 16)
    x = torch.randn(1, 16)
    assert head.forward(x).shape == (1, 24)
    assert torch.equal(head.forward(x), torch.nn.functional.linear(torch.cat([x, torch.zeros(1, 16)]), head.weight)[:1])
    x3 = torch.randn(3, 16)
    assert torch.equal(head.forward(x3), torch.nn.functional.linear(x3, head.weight))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuBLAS kernel selection is a GPU property")
def test_frozen_family_head_is_batch_invariant_on_this_gpu():
    """Receipt-grade check of the cuBLAS behaviour the padding relies on (40980 x 1024 bf16, H100 / cu128)."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20261012)
    weight = (torch.randn(10 * 4098, 1024, device=device, generator=generator) * 0.02).to(torch.bfloat16)
    x = torch.randn(1041, 1024, device=device, generator=generator).to(torch.bfloat16)
    reference = torch.nn.functional.linear(x, weight)
    unpadded = MultiOutputHead(1024, 10 * 4098, min_rows=1)
    padded = MultiOutputHead(1024, 10 * 4098, min_rows=2)
    unpadded.weight = padded.weight = weight
    for rows in (1, 2, 3, 4, 5, 8, 16, 64, 335):
        assert torch.equal(padded.forward(x[:rows]), reference[:rows]), rows
    # the padding exists because the single-row product differs on this hardware; if cuBLAS ever stops doing that
    # the padding is harmless, so only record the fact rather than fail.
    single = unpadded.forward(x[:1])
    if not torch.equal(single, reference[:1]):
        assert (single.float() - reference[:1].float()).abs().max() <= 0.125
