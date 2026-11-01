"""CPU tests for the alignment probe helpers (zonos3.tts.align_probe)."""

from __future__ import annotations

import pytest
import torch

from types import SimpleNamespace

from zonos3.tts import align_probe
from zonos3.tts.align_probe import AlignProbe, deliver, locate_text_span, prepare

N_CB = 10
PAD_TEXT = 8196


def _prompt(prefix: int, cond: int, text_ids: list[int], tail: int = 0, bos: bool = True, eos: bool = True):
    text_col = [PAD_TEXT] * prefix + [0] * cond + ([2] if bos else []) + text_ids + ([3] if eos else []) + [PAD_TEXT] * tail
    ids = torch.zeros(len(text_col), N_CB + 1, dtype=torch.int32)
    ids[:, N_CB] = torch.tensor(text_col, dtype=torch.int32)
    return ids


def test_locate_text_span_skips_prefix_and_condition_rows():
    ids = _prompt(prefix=7, cond=3, text_ids=[10, 11, 12, 13], tail=3)
    start, length = locate_text_span(ids, N_CB)
    assert (start, length) == (10, 6)
    assert ids[start, N_CB] == 2 and ids[start + length - 1, N_CB] == 3


@pytest.mark.parametrize("kwargs", [dict(bos=False), dict(eos=False)])
def test_locate_text_span_requires_bos_and_eos(kwargs):
    with pytest.raises(ValueError):
        locate_text_span(_prompt(2, 0, [10, 11, 12], **kwargs), N_CB)


def test_locate_text_span_rejects_special_id_inside_span():
    with pytest.raises(ValueError):
        locate_text_span(_prompt(2, 0, [10, 1, 12]), N_CB)


# ------------------------------------------------------------------------------------------------ batched kernels

N_HEADS, N_KV, HEAD_DIM, PAGES, LAYERS = 16, 4, 32, 300, (3, 7)
GROUP = N_HEADS // N_KV


class _KV:
    def __init__(self, generator):
        self.k = {layer: torch.randn(PAGES, 1, N_KV, HEAD_DIM, generator=generator) for layer in LAYERS}
        self.v = {layer: torch.randn(PAGES, 1, N_KV, HEAD_DIM, generator=generator) for layer in LAYERS}

    def k_cache(self, layer):
        return self.k[layer]

    def v_cache(self, layer):
        return self.v[layer]


def _batch(probes, device_lens, generator):
    for layer in LAYERS:
        align_probe.register_attention_layer(layer)
    page_table = torch.stack([torch.randperm(PAGES, generator=generator)[:60] for _ in probes]).to(torch.int32)
    reqs = [SimpleNamespace(align_probe=p, device_len=n) for p, n in zip(probes, device_lens)]
    batch = SimpleNamespace(reqs=reqs, size=len(reqs), padded_size=len(reqs),
                            attn_metadata=SimpleNamespace(page_table=page_table))
    prepare(batch, None, N_HEADS, torch.device("cpu"))
    return batch, page_table


def _readout_reference(q, kv, page_row, probe, scale):
    """Per-request reference: sum over probe heads of the (window-renormalized) text softmax -> (argmax, centroid)."""
    slots = page_row[probe.text_start : probe.text_start + probe.text_len].long()
    total = torch.zeros(probe.text_len)
    for layer, h in probe.heads:
        logits = (kv.k_cache(layer)[slots, 0, h // GROUP] @ q[layer][h]) * scale
        window = (probe.mask or {}).get((layer, h))
        if window is not None:
            lo, hi = align_probe.clip_window(window, probe.text_len)
            idx = torch.arange(probe.text_len)
            logits = logits.masked_fill((idx < lo) | (idx > hi), float("-inf"))
        total += torch.softmax(logits, -1)
    m = float((total * torch.arange(probe.text_len)).sum() / total.sum())
    return int(total.argmax()), m


def test_batched_readout_matches_per_request_reference():
    g = torch.Generator().manual_seed(0)
    kv = _KV(g)
    probes = [
        AlignProbe(heads=((3, 1), (7, 6), (7, 13)), observer=None, text_start=5, text_len=20),
        None,  # an unprobed request in the same batch
        AlignProbe(heads=((3, 0), (7, 5)), observer=None, text_start=30, text_len=13, mask={(7, 5): (2, 6)}),
    ]
    batch, page_table = _batch(probes, [40, 40, 50], g)
    assert batch.probe.rows == [0, 2] and batch.probe_legacy is None and batch.probe.masked.shape[0] == 1
    backend = SimpleNamespace(kvcache=kv, scale=HEAD_DIM**-0.5)
    q = {layer: torch.randn(3, N_HEADS, HEAD_DIM, generator=g) for layer in LAYERS}
    for layer in LAYERS:
        align_probe._readout(layer, q[layer], batch, backend)
    out = batch.probe.out
    for i in batch.probe.rows:
        pos, m = _readout_reference({lay: q[lay][i] for lay in LAYERS}, kv, page_table[i], probes[i], backend.scale)
        assert int(out[i, 0]) == pos
        assert abs(float(out[i, 1]) - m) < 1e-4


@pytest.mark.parametrize("window", [(2, 6), (-5, 100)])
def test_masked_heads_match_explicit_reference(window):
    """Masked head outputs = softmax over all the request's keys with only the text keys outside the window at -inf
    (a covering window is plain attention); every other head and row is untouched."""
    g = torch.Generator().manual_seed(1)
    kv = _KV(g)
    masked_heads = ((3, 0), (7, 5))
    probes = [
        AlignProbe(heads=((3, 1),), observer=None, text_start=5, text_len=20),
        AlignProbe(heads=masked_heads, observer=None, text_start=30, text_len=13, mask={h: window for h in masked_heads}),
    ]
    n_keys = 50
    batch, page_table = _batch(probes, [40, n_keys], g)
    backend = SimpleNamespace(kvcache=kv, scale=HEAD_DIM**-0.5)
    for layer in LAYERS:
        q = torch.randn(2, N_HEADS, HEAD_DIM, generator=g)
        o = torch.randn(2, N_HEADS, HEAD_DIM, generator=g)
        new = align_probe.apply(layer, q, o.clone(), batch, backend)
        for i in range(2):
            for h in range(N_HEADS):
                if i != 1 or (layer, h) not in masked_heads:
                    assert torch.equal(new[i, h], o[i, h])
                    continue
                slots = page_table[1, :n_keys].long()
                logits = (kv.k_cache(layer)[slots, 0, h // GROUP] @ q[1, h]) * backend.scale
                lo, hi = align_probe.clip_window(window, 13)
                text_idx = torch.arange(n_keys) - 30
                outside = (text_idx >= 0) & (text_idx < 13) & ((text_idx < lo) | (text_idx > hi))
                p = torch.softmax(logits.masked_fill(outside, float("-inf")), -1)
                assert p[:30].sum() > 0 and p[43:].sum() > 0  # non-text keys are never masked
                torch.testing.assert_close(new[1, h], p @ kv.v_cache(layer)[slots, 0, h // GROUP])


def test_deliver_calls_observers_and_stops_at_the_observed_step():
    seen = []

    class Obs:
        def on_step(self, probe, step, pos, m):
            seen.append((step, pos, m))
            probe.stop = step >= 2

    probe = AlignProbe(heads=((3, 0),), observer=Obs(), text_start=0, text_len=4)
    req = SimpleNamespace(align_probe=probe, device_len=50, max_device_len=900)
    batch = SimpleNamespace(reqs=[req], probe=SimpleNamespace(rows=[0]))
    deliver(batch, torch.tensor([[1.0, 1.5]]), finished=())
    assert seen == [(1, 1, 1.5)] and req.max_device_len == 900
    req.device_len = 51
    deliver(batch, torch.tensor([[2.0, 2.25]]), finished=())
    assert seen[-1] == (2, 2, 2.25) and req.max_device_len == 51  # no rows left: this step's row is the last
    deliver(batch, torch.tensor([[3.0, 3.0]]), finished=())
    assert len(seen) == 2  # a stopped probe gets no more calls
