"""Weights-backed alignment-probe tests (loads the real model; GPU).

Skipped unless CUDA and ZONOS3_TEST_MODEL (a ZONOS3 Mini LM dir, e.g. $ZONOS3_RELEASE_LM) are available:

    ZONOS3_TEST_MODEL=$ZONOS3_RELEASE_LM CUDA_VISIBLE_DEVICES=3 python -m pytest -o addopts="" \
        tests/misc/test_align_probe_gpu.py -q
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL = os.environ.get("ZONOS3_TEST_MODEL")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not MODEL or not torch.cuda.is_available(), reason="needs CUDA and ZONOS3_TEST_MODEL (loads weights)"
    ),
]

TEXTS = ["The quick brown fox jumps over the lazy dog near the river bank.", "Hello there, how are you doing today?"]
HEADS = ((3, 6), (7, 4), (7, 11))


@pytest.fixture(scope="module")
def tts(tmp_path_factory):
    import json

    from zonos3_sgl.tts.llm import TTSLLM

    heads = tmp_path_factory.mktemp("laci") / "heads.json"
    heads.write_text(
        json.dumps(
            {"frames_per_token": 3.5, "heads": [{"layer": layer, "head": head, "radius": 3} for layer, head in HEADS]}
        )
    )
    # decode graphs on, carrying the probe readout and one masked-request slot; LACI for en_us
    return TTSLLM(
        MODEL,
        decode_audio=False,
        max_running_req=4,
        attention_backend="fa",
        cache_type="naive",
        cuda_graph_max_bs=4,
        align_probe_max_text=256,
        align_probe_max_masked=1,
        memory_ratio=0.35,
        laci_heads={"en_us": str(heads)},
    )


@pytest.fixture
def replays(tts, monkeypatch):
    """Counts decode-graph replays."""
    count = [0]
    runner = tts.engine.graph_runner
    original = runner.replay

    def counted(batch):
        count[0] += 1
        return original(batch)

    monkeypatch.setattr(runner, "replay", counted)
    return count


def _params(**kw):
    from zonos3_sgl.message.tts import TTSSamplingParams

    return TTSSamplingParams(**{**{"seed": 7, "max_tokens": 60, "ignore_eos": True}, **kw})


class Reduced:
    def __init__(self, stop_at: int | None = None):
        self.seen, self.stop_at = [], stop_at

    def on_step(self, probe, step, pos, m):
        self.seen.append((step, pos, m))
        if self.stop_at is not None and step >= self.stop_at:
            probe.stop = True


class Full:
    """Reference: the eager all-heads distributions reduced like the batched readout."""

    def __init__(self):
        self.seen = []

    def on_attention(self, probe, step, text_attn, text_mass):
        a = sum(text_attn[h] for h in HEADS)
        m = float((a * torch.arange(a.shape[0], device=a.device)).sum() / a.sum())
        self.seen.append((step, int(a.argmax()), m))


def _probe(observer, **kw):
    from zonos3_sgl.tts.align_probe import AlignProbe

    return AlignProbe(heads=HEADS, observer=observer, **kw)


def test_graph_readout_matches_eager_reference_and_keeps_tokens(tts, replays):
    base = tts.generate(TEXTS, _params(), decode_audio=False)
    graph_replays = replays[0]
    reduced = [Reduced(), Reduced()]
    got = tts.generate(TEXTS, _params(), decode_audio=False, align_probes=[_probe(r) for r in reduced])
    assert replays[0] > graph_replays  # the probed decode ran inside the graphs
    full = [Full(), Full()]
    ref = tts.generate(
        TEXTS, _params(), decode_audio=False, align_probes=[_probe(f, all_heads=True) for f in full]
    )  # eager
    for b, g, r in zip(base, got, ref):
        assert g["audio_tokens"] == b["audio_tokens"] == r["audio_tokens"]
    for red, f in zip(reduced, full):
        assert [s for s, _, _ in red.seen] == list(range(1, len(red.seen) + 1)) and len(red.seen) >= 55
        expected = {s: (p, m) for s, p, m in f.seen}
        same_pos = 0
        for step, pos, m in red.seen:
            p_ref, m_ref = expected[step]
            same_pos += pos == p_ref
            assert abs(m - m_ref) < 1e-3, (step, m, m_ref)
        assert same_pos >= len(red.seen) - 1  # argmax may flip only on an exact near-tie


def test_observer_stop_finishes_request_at_the_observed_step(tts):
    rec = Reduced(stop_at=20)
    res = tts.generate(TEXTS[:1], _params(max_tokens=200), decode_audio=False, align_probes=[_probe(rec)])[0]
    assert rec.seen[-1][0] == 20
    # rows 0..19: the prefill row plus 19 decode rows; the stopping step's row is the finishing (dropped) row
    assert len(res["audio_tokens"]) == 20


def test_masked_probe_runs_in_the_graphs_and_masks(tts, replays):
    window = dict.fromkeys(HEADS, (0, 3))
    plain = tts.generate(TEXTS[:1], _params(max_tokens=30), decode_audio=False)[0]["audio_tokens"]
    rec = Reduced()
    before = replays[0]
    out = tts.generate(TEXTS[:1], _params(max_tokens=30), decode_audio=False, align_probes=[_probe(rec, mask=window)])[
        0
    ]["audio_tokens"]
    assert replays[0] > before  # one masked request fits the graphs' masked slot
    assert rec.seen and all(0 <= pos <= 3 and 0.0 <= m <= 3.0 for _, pos, m in rec.seen)
    assert out != plain  # the masked heads changed the generation
    # a later unmasked batch must not see the masked slots of an earlier step
    assert tts.generate(TEXTS[:1], _params(max_tokens=30), decode_audio=False)[0]["audio_tokens"] == plain


def test_masked_requests_beyond_the_graph_slots_run_eager(tts, replays):
    window = dict.fromkeys(HEADS, (0, 3))
    recs = [Reduced(), Reduced()]
    before = replays[0]
    tts.generate(TEXTS, _params(max_tokens=30), decode_audio=False, align_probes=[_probe(r, mask=window) for r in recs])
    assert replays[0] == before  # two masked requests, one slot: eager
    assert all(rec.seen and all(0 <= pos <= 3 for _, pos, _ in rec.seen) for rec in recs)


def test_prefill_history_feeds_repetition_penalty(tts):
    cfg = tts.engine.model_config
    n_cb, size = cfg.n_codebooks, cfg.codebook_size
    first = tts.generate(TEXTS[:1], _params(max_tokens=48), decode_audio=False)[0]["audio_tokens"][:43]
    assert len(first) == 43
    rows = torch.tensor(first, dtype=torch.int32)[:, :n_cb]
    rows = torch.cat((rows, torch.full((43, 1), int(cfg.text_vocab), dtype=torch.int32)), dim=1)
    params = _params(max_tokens=8, repetition_penalty=1e4, repetition_window=96, repetition_codebooks=6)
    out = tts.generate(TEXTS[:1], params, decode_audio=False, audio_prefill_rows=[rows], audio_prefill_history=[40])[0][
        "audio_tokens"
    ]
    assert len(out) >= 5
    for cb in range(6):
        hist = {int(t) for t in rows[3:, cb] if 0 <= int(t) < size}
        new = {int(r[cb]) for r in out[:5]}
        assert not hist & new, (cb, hist & new)


def _silence(tts, n=3):
    """n prefill rows that open the audio stream (one repeated plain frame, sheared like the release silence prefill)."""
    from zonos3.tts.prompt import TTSPromptConfig, silence_rows_from_codes

    cfg = tts.engine.model_config
    return silence_rows_from_codes(
        torch.full((cfg.n_codebooks,), 7, dtype=torch.int32),
        n,
        TTSPromptConfig(n_codebooks=cfg.n_codebooks, audio_pad_id=cfg.audio_pad_id, text_vocab=int(cfg.text_vocab)),
    )


def test_laci_without_failures_matches_plain_generation(tts):
    params = _params(max_tokens=400, ignore_eos=False)
    silence = _silence(tts)
    base = tts.generate(TEXTS, params, decode_audio=False, audio_prefill_rows=[silence] * 2)
    got = tts.generate(TEXTS, params, decode_audio=False, audio_prefill_rows=[silence] * 2, laci=True)
    for b, g in zip(base, got):
        rec = g["laci"]
        assert rec is not None and b["laci"] is None
        assert rec["attempts"] == 0, rec["failures"]
        # the same stream; LACI may end it early (last text bin read + tail) but never alters a row
        assert g["audio_tokens"] == b["audio_tokens"][: len(g["audio_tokens"])]
        assert rec["end"] in ("eos", "text_done")
        if rec["end"] == "eos":
            assert g["eos_frame"] == b["eos_frame"]


def test_laci_rollbacks_regenerate_inside_the_running_batch(tts, monkeypatch):
    from zonos3.tts.laci import LaciConfig

    # deliberately aggressive thresholds: every request stalls, rolls back and regenerates under the ACI mask
    monkeypatch.setattr(
        tts, "laci_config", LaciConfig(stall_factor=1.1, min_stall_frames=10, max_retries=2, bin_tokens=4)
    )
    silence = _silence(tts)
    got = tts.generate(
        TEXTS,
        _params(max_tokens=400, ignore_eos=False),
        decode_audio=False,
        audio_prefill_rows=[silence] * 2,
        laci=True,
    )
    for g in got:
        rec = g["laci"]
        assert rec["attempts"] >= 1
        resumed = [f for f in rec["failures"] if not f["ignored"]]
        assert len(resumed) == rec["attempts"]
        assert all(3 <= f["rollback_row"] <= f["detect_row"] for f in resumed)
        assert rec["generated_total"] > rec["final_rows"]
        assert len(g["audio_tokens"]) >= rec["final_rows"] - 1  # the final row rides on the finishing message
    assert not tts.laci_requests  # every session finished


def test_guards(tts):
    with pytest.raises(ValueError):
        tts.generate(TEXTS, _params(), decode_audio=False, align_probes=[_probe(Reduced())])  # one per prompt
    with pytest.raises(ValueError):
        tts.generate(TEXTS[:1], _params(), decode_audio=False, audio_prefill_history=[3])  # more than the prefill rows
    with pytest.raises(ValueError):
        tts.generate(TEXTS[:1], _params(controls_cfg_scale=2.0), decode_audio=False, laci=True)  # no CFG twin
    with pytest.raises(ValueError):
        tts.generate(TEXTS[:1], _params(), decode_audio=False, language="es", laci=True)
