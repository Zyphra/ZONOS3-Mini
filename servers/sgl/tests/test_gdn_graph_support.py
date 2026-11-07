"""Helpers behind the GDN CUDA-graph decode: static index tables and FLA cache pinning."""

from __future__ import annotations

import sys
import types
from collections import deque

import pytest
import torch

fla = pytest.importorskip("fla", reason="FLA overlay not on the path")  # noqa: F841  (OriginalGDN imports it)
from zonos3_sgl.models.gdn import CHUNK, OriginalGDN  # noqa: E402


def test_static_tables_describe_bs_windows_of_64_rows():
    tables = OriginalGDN.static_tables(torch.device("cpu"), [1, 4])
    cu, rows = tables[4]
    assert cu.tolist() == [0, 64, 128, 192, 256] and cu.dtype == torch.int32
    assert rows.tolist() == [0, 64, 128, 192] and rows.dtype == torch.long
    assert tables[1][0].tolist() == [0, CHUNK]


def test_pin_fla_cache_keeps_results_keyed_by_the_static_tensors():
    static = torch.arange(0, 3, dtype=torch.int32) * CHUNK
    other = torch.tensor([0, 10], dtype=torch.int32)
    cached: deque = deque(maxlen=4)
    kept = torch.ones(3)
    foreign = torch.zeros(2)
    cached.append(((static, 64), {}, kept))
    cached.append(((other, 64), {}, foreign))
    cached.append(((other,), {"cu_seqlens_cpu": static}, kept * 2))

    def wrapper():
        return cached  # closure over the deque, like fla.utils.tensor_cache

    module = types.ModuleType("fla_fake_index")
    module.prepare_things = wrapper
    sys.modules["fla_fake_index"] = module
    try:
        pinned: dict = {}
        added = OriginalGDN.pin_fla_cache([static], pinned)
        assert added == 2 and len(pinned) == 2
        assert id(kept) in pinned and id(foreign) not in pinned
        assert all(torch.is_tensor(t) for t in pinned.values())
        assert OriginalGDN.pin_fla_cache([static], pinned) == 0  # idempotent
    finally:
        del sys.modules["fla_fake_index"]
