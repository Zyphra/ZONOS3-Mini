"""LACI and span-follow parity: servers/ggml `laci-replay` (src/zonos3-laci.cpp, src/zonos3-span-follow.cpp) against
zonos3.tts.laci.LaciSession and zonos3.tts.span_follow driven the same way (model/tests/test_laci.py `run_attempt`).

The ctest cases replay tests/fixtures/laci-replay/<case>.trace.json against <case>.expected.json; regenerate them with
``uv run --package zonos3-tools python servers/ggml/tests/parity/test_ggml_laci_replay.py --write-fixtures``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from zonos3.tts import span_follow as sf
from zonos3.tts.laci import DurationTarget, LaciConfig, LaciHeads, LaciSession, RateBuckets

REPO = Path(__file__).resolve().parents[4]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "laci-replay"

TEXT_LEN = 100
FPT = 4
HEADS = {
    "frames_per_token": float(FPT),
    "heads": [{"layer": 7, "head": 4, "radius": 3}, {"layer": 3, "head": 2, "radius": 5}],
}
EDGES = [10.117, 12.123, 13.286, 14.25, 15.14, 16.079, 17.094, 18.489, 21.208]  # ZONOS3 Mini release speaking rate


# ------------------------------------------------------------------------------------------------- Python drivers


def _mask(mask) -> list | None:
    return None if mask is None else [[h[0], h[1], lo, hi] for h, (lo, hi) in mask.items()]


def _tags(rows) -> list[int]:
    return [int(r[0]) for r in rows]


def run_laci(t: dict) -> dict:
    """Mirror of laci-replay's run_laci: one LaciSession driven like test_laci.run_attempt across attempts."""
    hj = t["heads_json"]
    if isinstance(hj, str):
        heads = LaciHeads.load(hj)
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(hj, f)
        heads = LaciHeads.load(f.name)
        Path(f.name).unlink()
    if t.get("frames_per_token") is not None:
        heads = dataclasses.replace(heads, frames_per_token=float(t["frames_per_token"]))
    cfg = LaciConfig(**t.get("cfg", {}))
    n_prefill = t.get("n_prefill", 0)
    d = t.get("duration_target")
    duration = (
        None
        if d is None
        else (
            DurationTarget(float(d))
            if isinstance(d, (int, float))
            else DurationTarget(float(d["frames"]), d.get("adjust_rate", True))
        )
    )
    r = t.get("rates")
    rates = None if r is None else RateBuckets(r["index"], tuple(r["edges"]), r["n_buckets"], r.get("first", 0))
    rd = t.get("rate_deltas")
    rate_deltas = None if rd is None else [None if e is None else torch.tensor(e, dtype=torch.float32) for e in rd]
    fj = t.get("follower")
    follower = None if fj is None else _follower(fj)
    s = LaciSession(
        heads=heads,
        cfg=cfg,
        prefill_rows=torch.full((n_prefill, 1), -1, dtype=torch.int32),
        seed=t["seed"] if "seed" in t else 0,
        max_tokens=t.get("max_tokens", 4000),
        n_codebooks=t.get("n_codebooks", 10),
        hold=t.get("hold", False),
        duration=duration,
        rates=rates,
        condition_values=t.get("condition_values"),
        bucket=t.get("bucket"),
        rate_deltas=rate_deltas,
        follower=follower,
    )
    attempts = t.get("attempts") or [{"steps": t["steps"], "eos_at": t.get("eos_at"), "eos_lag": t.get("eos_lag", 0)}]
    C = s.n_codebooks
    out_attempts = []
    for k in range(t.get("max_attempts", 64) + 1):
        assert k < t.get("max_attempts", 64), "more than max_attempts attempts"
        a = attempts[min(k, len(attempts) - 1)]
        tag = k + 1
        eos_at, eos_lag = a.get("eos_at"), a.get("eos_lag", 0)
        probe = SimpleNamespace(text_len=t["text_len"], mask=s.probe_mask(), stop=False)
        rec = {"mask": _mask(probe.mask)}
        changes, drained, fweights = [], [], []
        s.append(torch.tensor([tag], dtype=torch.int32), None)
        drained += _tags(s.drain())
        countdown, steps_run, stop_step, boost_step = None, 0, None, None
        for step, (pos, m) in enumerate(a["steps"], start=1):
            if follower is not None:
                fweights.append(list(follower.weights(s.attempt_start + step - s.n_prefill)))
            before = _mask(probe.mask)
            s.on_step(probe, step, pos, float(m))
            steps_run = step
            after = _mask(probe.mask)
            if after != before:
                changes.append([step, after])
            eos = None
            if eos_at is not None and step >= eos_at:
                eos = eos_at - eos_lag
                countdown = C + 1 if countdown is None else countdown
            s.append(torch.tensor([tag * 100000 + step], dtype=torch.int32), eos)
            if boost_step is None and s.stop_boost() > 0:
                boost_step = step  # the first step whose next row the sampler boosts
            if countdown is not None:
                countdown -= 1
            if probe.stop or countdown == 0:
                if probe.stop:
                    stop_step = step
                break
            drained += _tags(s.drain())
        rec.update(
            mask_changes=changes, steps_run=steps_run, stop_step=stop_step, drained=drained, boost_step=boost_step
        )
        if follower is not None:
            rec["follower_weights"] = fweights
        cont = s.finish_attempt()
        rec["continuation"] = (
            None
            if cont is None
            else {
                "n_rows": int(cont.rows.shape[0]),
                "history_rows": cont.history_rows,
                "seed": cont.seed,
                "max_tokens": cont.max_tokens,
                "condition_values": cont.condition_values,
                "swap_delta": cont.swap_delta,
                "steering_delta": None if cont.steering_delta is None else cont.steering_delta.tolist(),
                "probe_mask": _mask(s.probe_mask()),
            }
        )
        out_attempts.append(rec)
        if cont is None:
            break
    out = {
        "attempts": out_attempts,
        "final_drain": _tags(s.drain(final=True)),
        "eos_frame": s.eos_frame(),
        "end": list(s.end),
        "record": s.record(),
        "stream": _tags(s.stream[s.n_prefill :]),
    }
    return json.loads(json.dumps(out))


def _follower(j: dict) -> sf.SpanFollower:
    return sf.SpanFollower(
        [tuple(r) for r in j["ranges"]],
        torch.tensor(j["deltas"], dtype=torch.float32),
        sf.FollowConfig(**j.get("cfg", {})),
        j.get("origin", 0),
    )


def run_span_follow(t: dict) -> dict:
    f = _follower(t)
    results = []
    for op in t["ops"]:
        name = op[0]
        if name == "observe":
            f.observe(op[1], op[2])
            results.append(f.position)
        elif name == "on_step":
            f.on_step(None, op[1], op[2], float(op[2]))
            results.append(f.position)
        elif name == "weights":
            results.append(list(f.weights(op[1])))
        elif name == "delta":
            d = f.delta(op[1])
            results.append(None if d is None else d.tolist())
        elif name == "truncate":
            f.truncate(op[1])
            results.append(f.position)
        elif name == "position":
            results.append(f.position)
        else:
            raise ValueError(name)
    return json.loads(json.dumps({"results": results}))


class _WordBpe:
    """Test tokenizer for the SuperBPE path (laci-replay's word_offsets): tokens are the matches of \\s*\\S+|\\s+."""

    def encode(self, s, add_special_tokens=False):
        import re

        return SimpleNamespace(offsets=[m.span() for m in re.finditer(r"\s*\S+|\s+", s)])


def run_span_ranges(t: dict, monkeypatch) -> dict:
    import difflib

    from zonos3.tts import superbpe

    text, final = t["text"], t["final"]
    spec = None
    if t.get("bpe") == "words":
        monkeypatch.setattr(superbpe, "tokenizer", lambda *a: _WordBpe())
        spec = {"path": "-", "sha256": "-", "vocab_size": 0}
        if t.get("phonemes") is not None:
            spec["phonemes"] = t["phonemes"]
        if t.get("start"):
            spec["start"] = {"id": 0, "marker": "\ue002"}
        if t.get("pauses"):
            spec["pauses"] = {"ids": [0, 1], "markers": ["\ue003", "\ue004"], "seconds": [0.25, 0.5]}
    ops = difflib.SequenceMatcher(None, text, final, autojunk=False).get_opcodes()
    spans = [tuple(s) for s in t["spans"]]
    return json.loads(
        json.dumps(
            {
                "opcodes": [list(o) for o in ops],
                "map_starts": sf.map_offsets(text, final, t["offsets"]),
                "map_ends": sf.map_offsets(text, final, t["offsets"], end=True),
                "token_ranges": sf.token_char_ranges(final, spec),
                "span_ranges": sf.span_token_ranges(text, final, spans, spec),
            }
        )
    )


# ---------------------------------------------------------------------------------------------------------- traces


def reading(lo=0, hi=TEXT_LEN, fpt=FPT, jitter=0.0):
    """[pos, m] per step: each pos in [lo, hi) held for `fpt` steps; m = pos + a deterministic wobble."""
    out = []
    for pos in range(lo, hi):
        for _ in range(fpt):
            out.append([pos, pos + jitter * math.sin(len(out))])
    return out


def tail(n=60, jitter=0.0):
    return [[TEXT_LEN - 1, TEXT_LEN - 1 + jitter * math.sin(k)] for k in range(n)]


def base(**kw) -> dict:
    t = {
        "mode": "laci",
        "text_len": TEXT_LEN,
        "heads_json": HEADS,
        "n_prefill": 3,
        "n_codebooks": 10,
        "max_tokens": 4000,
        "seed": 0,
    }
    t.update(kw)
    return t


def laci_traces() -> dict[str, dict]:
    stall_follow = {
        "ranges": [[2, 9], [30, 45], [60, 60]],
        "deltas": [[1, 0, 0.5], [0, 2, -1], [3, 3, 3]],
        "cfg": {"lead": 1.0, "lag": 0.0, "smooth": 3, "ramp": 3},
    }
    traces = {
        "clean": base(steps=reading(jitter=0.4) + tail(jitter=0.2)),
        "skip": base(
            seed=7,
            attempts=[
                {"steps": reading(0, 40, jitter=0.3) + reading(56, 100, jitter=0.3) + tail()},
                {"steps": reading(38, 100, jitter=0.3) + tail()},
            ],
        ),
        "stall": base(
            seed=5,
            follower=stall_follow,
            attempts=[
                {"steps": reading(0, 40) + [[39, 39.2]] * 200},
                {"steps": reading(36, 100, jitter=0.5) + tail()},
            ],
        ),
        "early_eos": base(
            attempts=[
                {"steps": reading(0, 30), "eos_at": 120 - 12, "eos_lag": 4},
                {"steps": reading(18, 100) + tail()},
            ]
        ),
        # the release heads file (two heads, its own pace prior) through LaciHeads.load
        "release_heads": base(
            heads_json=str(REPO / "assets" / "laci_heads" / "zonos3_mini_mect60k_heads.json"),
            seed=11,
            attempts=[
                {"steps": reading(0, 50, fpt=3, jitter=0.8) + reading(70, 100, fpt=3) + tail()},
                {"steps": reading(45, 100, fpt=3, jitter=0.8) + tail()},
            ],
        ),
        # duration target 600 for a 400-frame reading: pace_fast retakes in slower buckets until the pace budget
        "pace_miss": base(
            duration_target={"frames": 600, "adjust_rate": True},
            rates={"index": 2, "edges": EDGES, "n_buckets": 10},
            condition_values=[10, 10, 4],
            bucket=4,
            steps=reading() + tail(),
        ),
        # extended scale: a slow take in a steered bucket comes back into the row with its delta swapped
        "pace_slow_extended": base(
            duration_target=200,
            rates={"index": 2, "edges": [8.4, 9.2, *EDGES, 23.3], "n_buckets": 10, "first": -2},
            condition_values=[10, 10, 10],
            bucket=-1,
            rate_deltas=[None if 0 <= b < 10 else [float(b)] * 4 for b in range(-2, 11)],
            steps=reading() + tail(),
        ),
        # a stall the target pace catches before the detector, then a truncation one bucket slower
        "pace_stall_then_eos": base(
            heads_json={**HEADS, "frames_per_token": 1.0},
            duration_target=100,
            rates={"index": 2, "edges": EDGES, "n_buckets": 10},
            condition_values=[10, 10, 4],
            attempts=[
                {"steps": reading(0, 30, fpt=1) + [[29, 29.0]] * 35 + reading(30, 100, fpt=1)},
                {"steps": reading(20, 40, fpt=1), "eos_at": 15, "eos_lag": 2},
                {"steps": reading(10, 100, fpt=1) + tail(30)},
            ],
        ),
        "capped_budget": base(
            cfg={"max_retries": 1},
            max_tokens=300,
            seed=None,
            attempts=[
                {"steps": reading(0, 40) + [[39, 39.0]] * 200},
                {"steps": reading(30, 50) + [[49, 49.0]] * 200},
            ],
        ),
        # two bins read slowly: the prior's share of the last bin fills long before the reading reaches EOS; the text
        # is done (and the tail starts) only once it does
        "slow_two_bins": base(
            text_len=14,
            heads_json={**HEADS, "frames_per_token": 3.5},
            steps=reading(0, 13, fpt=6, jitter=0.3) + [[13, 13.0]] * 30,
        ),
    }
    traces["stall_held"] = {**traces["stall"], "hold": True}
    rng = random.Random(1234)
    for k in range(4):
        traces[f"random_{k}"] = _random_trace(rng)
    return traces


def _random_trace(rng: random.Random) -> dict:
    """Random readings with skips, stalls and early EOS (test_laci.random_sessions), as a fixed trace."""
    attempts = []
    p0 = 0
    for _ in range(6):
        pos, p = [], p0
        while p < TEXT_LEN and len(pos) < 700:
            r = rng.random()
            if r < 0.01:
                p = min(TEXT_LEN - 1, p + rng.randint(17, 30))
            elif r < 0.02:
                pos += [p] * rng.randint(50, 120)
            pos += [min(p, TEXT_LEN - 1)] * rng.randint(2, 6)
            p += 1
        pos += [TEXT_LEN - 1] * 40
        steps = [[q, q + rng.uniform(-1.5, 1.5)] for q in pos]
        eos_at = rng.randint(20, len(steps) - 1) if rng.random() < 0.3 else None
        attempts.append({"steps": steps, "eos_at": eos_at, "eos_lag": rng.randint(0, 9)})
        p0 = max(0, p0 + rng.randint(-5, 25))
    t = base(seed=rng.randint(0, 2**32 - 1), cfg={"max_retries": rng.choice([1, 3, 8])}, attempts=attempts)
    if rng.random() < 0.5:
        t.update(
            duration_target={"frames": rng.choice([150, 300, 450, 700]), "adjust_rate": rng.random() < 0.8},
            rates={"index": 2, "edges": EDGES, "n_buckets": 10},
            condition_values=[10, 10, rng.randint(0, 9)],
        )
    return t


def span_follow_traces() -> dict[str, dict]:
    # scheduler-like: decide frame f, then the readout of frame f - 2 arrives (overlap scheduling); a rollback midway
    reading_ = [min(39, f // 3) for f in range(120)]
    ops = []
    for f in range(60):
        ops.append(["delta", f])
        if f >= 2:
            ops.append(["observe", f - 2, reading_[f - 2]])
    ops += [["truncate", 41], ["position"]]
    for f in range(41, 120):
        ops.append(["weights", f])
        ops.append(["observe", f, reading_[f] + (3 if f in (50, 51) else 0)])
    scheduler = {
        "mode": "span_follow",
        "ranges": [[1, 5], [10, 20], [30, 30], [25, 38]],
        "deltas": [[0.1, -0.2, 0.3], [1.0, 0.5, -0.25], [9, 9, 9], [-0.7, 0.05, 2.0]],
        "ops": ops,
    }
    # a span at the text start (on before any readout), on_step with an origin, wider smoothing and ramp
    ops2 = [["weights", 0], ["weights", 1]]
    pos = [0, 0, 1, 1, 3, 2, 2, 5, 6, 6, 9, 7, 8, 10, 12, 14, 14, 15, 15, 15, 16, 18, 19, 19, 20]
    for s, p in enumerate(pos, start=1):
        ops2 += [["on_step", s, p], ["delta", 7 + s], ["weights", 7 + s]]
    ops2 += [["truncate", 15], ["delta", 15], ["on_step", 30, 12], ["weights", 40], ["position"]]
    origin = {
        "mode": "span_follow",
        "ranges": [[0, 3], [6, 9], [12, 18]],
        "deltas": [[1.0, 2.0], [0.5, -0.5], [0.25, 4.0]],
        "cfg": {"lead": 2.5, "lag": 1.0, "smooth": 5, "ramp": 4},
        "origin": 7,
        "ops": ops2,
    }
    return {"span_follow_scheduler": scheduler, "span_follow_origin": origin}


def span_range_traces() -> dict[str, dict]:
    text = "Dr. Smith paid $5   <laugh/> today, honestly."
    final = "Doctor Smith paid five dollars <laugh/> today, honestly."
    words = {
        "mode": "span_ranges",
        "text": text,
        "final": final,
        "bpe": "words",
        "spans": [[0, 9], [10, 17], [15, 28], [20, 28], [29, 45], [17, 20], [44, 44]],
        "offsets": [0, 3, 4, 15, 16, 17, 18, 20, 45, 50],
    }
    plain = dict(
        words,
        bpe=None,
        text="Größe: 3 km — naïve café",
        final="Größe: drei Kilometer, naïve café",
        spans=[[0, 5], [7, 11], [14, 24], [5, 7]],
        offsets=[0, 6, 7, 8, 10, 11, 13, 24],
    )
    phon = dict(
        words,
        text="say \ue000tə\ue001 now",
        final="say \ue000tə\ue001 now",
        spans=[[4, 8], [0, 3], [9, 12]],
        offsets=[0, 4, 8, 12],
        phonemes={"open_marker": "\ue000", "close_marker": "\ue001"},
    )
    # the start marker (inserted by start_immediately) and a pause marker, each one range
    markers = dict(
        words,
        text="Say\ue004 \ue000tə\ue001 now",
        final="\ue002Say\ue004 \ue000tə\ue001 now",
        spans=[[0, 3], [5, 9], [10, 13]],
        offsets=[0, 3, 4, 9, 13],
        phonemes={"open_marker": "\ue000", "close_marker": "\ue001"},
        start=True,
        pauses=True,
    )
    literal = dict(
        words,
        bpe=None,
        text="<shouting>hi</shouting> ok",
        final="<shouting>hi</shouting> ok",
        spans=[[0, 23]],
        offsets=[0, 23],
    )
    return {
        "span_ranges_words": words,
        "span_ranges_bytes": plain,
        "span_ranges_phonemes": phon,
        "span_ranges_markers": markers,
        "span_ranges_literal": literal,
    }


# ------------------------------------------------------------------------------------------------------------ tests


def _close(a, b, path=""):
    """Equal structures; floats within 1e-6 (float32 span deltas)."""
    if isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)), path
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6), f"{path}: {a} != {b}"
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: {a} != {b}"
        for k, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{k}]")
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{path}: keys {a.keys()} != {b.keys()}"
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    else:
        assert a == b, f"{path}: {a} != {b}"


LACI = laci_traces()


@pytest.mark.parametrize("name", list(LACI))
def test_laci_session_matches_python(tool, name):
    trace = LACI[name]
    want = run_laci(trace)
    got = tool("laci-replay", trace)
    assert got == want  # rows, decisions, records and pace_tok exactly


def test_laci_traces_cover_every_failure_kind():
    kinds = {f["kind"] for t in LACI.values() for f in run_laci(t)["record"]["failures"]}
    assert {"skip", "hallucination", "truncation", "pace_fast", "pace_slow"} <= kinds
    ends = {run_laci(t)["record"]["end"] for t in LACI.values()}
    assert {"text_done", "eos"} <= ends


@pytest.mark.parametrize("name", list(span_follow_traces()))
def test_span_follower_matches_python(tool, name):
    trace = span_follow_traces()[name]
    _close(tool("laci-replay", trace), run_span_follow(trace))


@pytest.mark.parametrize("name", list(span_range_traces()))
def test_span_token_ranges_match_python(tool, monkeypatch, name):
    trace = span_range_traces()[name]
    want = run_span_ranges(trace, monkeypatch)
    if name == "span_ranges_literal":
        assert want["span_ranges"] == [[1, 24]]
    assert tool("laci-replay", trace) == want


CTEST_CASES = (
    "clean",
    "skip",
    "stall",
    "stall_held",
    "slow_two_bins",
    "early_eos",
    "pace_miss",
    "span_follow_scheduler",
)


def _ctest_cases() -> dict[str, tuple[dict, dict]]:
    sft = span_follow_traces()
    return {
        name: (LACI[name], run_laci(LACI[name])) if name in LACI else (sft[name], run_span_follow(sft[name]))
        for name in CTEST_CASES
    }


def test_ctest_fixtures_are_current():
    """The ctest fixtures equal the Python driver's traces and results (regenerate with --write-fixtures)."""
    for name, (trace, want) in _ctest_cases().items():
        assert json.loads((FIXTURES / f"{name}.trace.json").read_text()) == json.loads(json.dumps(trace)), name
        _close(json.loads((FIXTURES / f"{name}.expected.json").read_text()), want, name)


if __name__ == "__main__":
    if sys.argv[1:] != ["--write-fixtures"]:
        sys.exit(f"usage: {sys.argv[0]} --write-fixtures")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, (trace, want) in _ctest_cases().items():
        (FIXTURES / f"{name}.trace.json").write_text(json.dumps(trace) + "\n")
        (FIXTURES / f"{name}.expected.json").write_text(json.dumps(want) + "\n")
