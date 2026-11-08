"""Helpers behind the GDN CUDA graphs: static index tables, padded prefill layouts and FLA cache pinning."""

from __future__ import annotations

import sys
import types
from collections import deque

import pytest
import torch

fla = pytest.importorskip("fla", reason="FLA overlay not on the path")  # noqa: F841  (OriginalGDN imports it)
from fla.ops.utils.index import prepare_chunk_indices  # noqa: E402
from zonos3_sgl.models.gdn import CHUNK, OriginalGDN, PrefillLayout, chunk_tables  # noqa: E402


def test_static_tables_describe_bs_windows_of_64_rows():
    tables = OriginalGDN.static_tables(torch.device("cpu"), [1, 4])
    cu, chunks, rows = tables[4]
    assert cu.tolist() == [0, 64, 128, 192, 256] and cu.dtype == torch.int32
    assert chunks.tolist() == [[0, 0], [1, 0], [2, 0], [3, 0]] and chunks.dtype == torch.int32
    assert rows.tolist() == [0, 64, 128, 192] and rows.dtype == torch.long
    assert tables[1][0].tolist() == [0, CHUNK]


@pytest.mark.parametrize("lengths", [[1], [64], [65], [5, 63, 64, 130], [200, 1, 128]])
def test_chunk_tables_match_the_indices_fla_derives(lengths):
    cu, chunks = chunk_tables(lengths, torch.device("cpu"))
    assert cu.tolist() == torch.tensor([0] + lengths).cumsum(0).tolist()
    assert torch.equal(chunks, prepare_chunk_indices(cu, CHUNK))


def test_prefill_layout_covers_every_padded_row_once():
    whole = [0, 3, 1]
    layout = PrefillLayout.build(whole, torch.device("cpu"))
    rows = sum(CHUNK * (w + 1) for w in whole)
    covered = torch.cat([layout.whole_rows, layout.window_rows]).sort().values
    assert torch.equal(covered, torch.arange(rows))
    assert layout.whole_reqs.tolist() == [1, 2] and layout.whole_cu.tolist() == [0, 3 * CHUNK, 4 * CHUNK]
    # each window directly follows its request's complete chunks
    assert layout.window_base.tolist() == [0, 3 * CHUNK, CHUNK]
    assert layout.window_rows.view(3, CHUNK)[:, 0].tolist() == [0, 4 * CHUNK, 6 * CHUNK]


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
