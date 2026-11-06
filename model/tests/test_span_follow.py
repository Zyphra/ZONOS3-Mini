"""Span steering by reading position (zonos3.tts.span_follow): character spans -> text-span token positions, and the
per-request gate (ramps, smoothing, decisions per frame, LACI rollbacks)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("tokenizers")

from test_laci import TEXT_LEN, run_attempt, session  # noqa: E402
from tiny_lm import wordlevel_spec  # noqa: E402
from zonos3.tts.span_follow import (  # noqa: E402
    FollowConfig,
    SpanFollower,
    map_offsets,
    span_token_ranges,
    token_char_ranges,
)
from zonos3.tts.superbpe import load_spec, text_ids  # noqa: E402
from zonos3.utils.hashing import sha256_file  # noqa: E402

O, C = "\ue000", "\ue001"


@pytest.fixture
def spec(tmp_path):
    tokenizer, size = wordlevel_spec(["please", "call", "stella", "tomorrow", "morning", ".", "[UNK]"],
                                     whitespace=True)
    path = tmp_path / "superbpe.json"
    path.write_text(tokenizer)
    return dict(kind="superbpe", path=str(path), sha256=sha256_file(path),
                vocab_size=size, offset=4, bos=2, eos=3)


def phoneme_spec(spec):
    text_vocab = spec["vocab_size"] + 4
    return {**spec, "phonemes": dict(open_id=text_vocab + 1, close_id=text_vocab + 2, symbols=list("stɛlə"),
                                     open_marker=O, close_marker=C, inventory_sha256="0" * 64)}


# ------------------------------------------------------------------------------------------------- token mapping


def test_one_char_range_per_text_id(spec):
    """token_char_ranges lines up with text_ids for words, phoneme groups and bytes (BOS / EOS included)."""
    ps = phoneme_spec(spec)
    for text, sp in (("please call stella tomorrow morning .", spec), (f"please call {O}stɛlə{C} tomorrow .", ps),
                     (f"{O}stɛlə{C}", ps), ("please", ps)):
        assert len(token_char_ranges(text, sp)) == len(text_ids(text, sp)), text
    assert len(token_char_ranges("café ü", None)) == len("café ü".encode()) + 2  # byte ids: one per UTF-8 byte


def test_spans_map_to_the_tokens_spelling_them(spec):
    text = "please call stella tomorrow morning ."
    # BOS 0, please 1, call 2, stella 3, tomorrow 4, morning 5, . 6, EOS 7
    spans = [(0, 6), (7, 11), (12, 27), (28, 35), (11, 12)]
    assert span_token_ranges(text, text, spans, spec) == [(1, 2), (2, 3), (3, 5), (5, 6), (3, 3)]
    # outer whitespace and partial words: the span widens to whole tokens, never into the neighbours
    assert span_token_ranges(text, text, [(6, 12), (13, 15)], spec) == [(2, 3), (3, 4)]


def test_phoneme_group_spans(spec):
    ps = phoneme_spec(spec)
    text = f"please call {O}stɛlə{C} tomorrow ."
    # BOS 0, please 1, call 2, open 3, s t ɛ l ə 4..8, close 9, tomorrow 10, . 11, EOS 12
    group = (text.index(O), text.index(C) + 1)
    after = (text.index("tomorrow"), text.index("tomorrow") + 8)
    assert span_token_ranges(text, text, [group, after], ps) == [(3, 10), (10, 11)]
    assert text_ids(text, ps)[3] == ps["phonemes"]["open_id"] and text_ids(text, ps)[9] == ps["phonemes"]["close_id"]


def test_byte_tokens(spec):
    text = "a café b"
    # BOS 0, a 1, ' ' 2, c 3, a 4, f 5, é 6 7, ' ' 8, b 9, EOS 10
    assert span_token_ranges(text, text, [(2, 6), (7, 8)], None) == [(3, 8), (9, 10)]


def test_offsets_follow_text_normalization(spec):
    original, final = "Call Stella at 3 tomorrow.", "Call Stella at three tomorrow."
    assert map_offsets(original, final, [5, 11]) == [5, 11]  # before the rewrite: unchanged
    s, e = original.index("3"), original.index("3") + 1
    assert final[map_offsets(original, final, [s])[0]:map_offsets(original, final, [e], end=True)[0]] == "three"
    t = original.index("tomorrow")
    assert final[map_offsets(original, final, [t])[0]:].startswith("tomorrow")
    assert map_offsets(original, final, [len(original)]) == [len(final)]


REAL = Path(os.environ["ZONOS3_TEST_PHONEME_MODEL"]) if os.environ.get("ZONOS3_TEST_PHONEME_MODEL") else None


@pytest.mark.skipif(REAL is None or not (REAL / "params.json").exists(),
                    reason="set ZONOS3_TEST_PHONEME_MODEL to a phoneme-token checkpoint")
def test_release_superbpe_lines_up():
    real = load_spec(REAL)
    o, c = real["phonemes"]["open_marker"], real["phonemes"]["close_marker"]
    phone = real["phonemes"]["symbols"][0]
    for text in ("Hello world, of the people.", "Il était excité, et en même temps inquiet : peut-être.",
                 f"Please call {o}{phone * 3}{c} tomorrow morning.", "  spaced   out  "):
        assert len(token_char_ranges(text, real)) == len(text_ids(text, real)), text
    text = "Please call Stella. Ask her to bring these things with her from the store."
    (a, b), = span_token_ranges(text, text, [(text.index("Stella"), text.index("Stella") + 6)], real)
    words = "".join(text[x:y] for x, y in token_char_ranges(text, real)[a:b])
    assert "Stella" in words and "Please" not in words and "Ask" not in words


# ------------------------------------------------------------------------------------------------------- follower


def follower(ranges, lead=1.0, lag=0.0, smooth=3, d=4):
    deltas = torch.arange(1, len(ranges) + 1, dtype=torch.float32)[:, None].repeat(1, d)
    return SpanFollower(ranges, deltas, FollowConfig(lead=lead, lag=lag, smooth=smooth))


def drive(f, positions, lag=2, start=1):
    """Like the scheduler under overlap: frame t is decided with the readouts of frames <= t - lag; returns the
    per-frame weights from ``start``."""
    out = []
    for t in range(start, start + len(positions)):
        out.append(f.weights(t))
        if t - lag + 1 >= start:
            f.observe(t - lag + 1, positions[t - lag + 1 - start])
    return out


def test_span_at_the_start_is_on_before_any_readout():
    f = follower([(1, 3)])
    assert f.weights(1) == [1.0]
    assert torch.equal(f.delta(1), torch.ones(4))


def test_ramps_in_at_the_lead_and_out_after_the_span():
    f = follower([(10, 12)], lead=1)
    reading = [p for p in range(0, 20) for _ in range(4)]  # 4 frames per token
    w = [x[0] for x in drive(f, reading)]
    on = next(i for i, x in enumerate(w) if x > 0)
    # frame on + 1 is the first decided with the median of 3 readouts (of frames <= on - 1) at 10 - lead
    assert w[on:on + 3] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert sorted(reading[on - 4:on - 1])[1] == 9 and sorted(reading[on - 5:on - 2])[1] == 8
    off = next(i for i in range(on + 2, len(w)) if w[i] < 1)
    assert w[off:off + 3] == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert sorted(reading[off - 4:off - 1])[1] == 12  # the token after the span (lag 0)
    assert sorted(reading[off - 5:off - 2])[1] == 11
    assert all(x == 0 for x in w[off + 2:])


def test_a_lone_spike_does_not_open_a_span_and_jumps_are_capped():
    f = follower([(40, 42)], lead=0)
    reading = [5] * 10 + [60] + [5] * 10
    assert all(x == [0.0] for x in drive(f, reading))
    g = follower([(40, 42)], lead=0)
    for frame in range(1, 6):
        g.observe(frame, 60)  # a real jump: the smoothed position climbs one token per readout
    assert g.position == 5.0


def test_twin_and_repeated_calls_see_the_frame_decided_once():
    f = follower([(3, 5)], lead=0)
    f.observe(1, 4)
    f.observe(2, 4)
    w = f.weights(3)
    f.observe(3, 30)
    f.observe(4, 30)
    assert f.weights(3) is w  # the speaker-CFG twin asks again for the same frame


def test_truncate_replays_from_the_rollback_frame():
    f = follower([(5, 7), (20, 22)], lead=1)
    reading = [p for p in range(0, 30) for _ in range(3)]
    full = drive(f, reading)
    first = [w[0] for w in full]
    on = next(i for i, x in enumerate(first) if x > 0) + 1  # frames count from 1
    cut = on + 1
    f.truncate(cut)
    assert all(f._weights[t] == full[t - 1] for t in range(1, cut))
    assert all(t < cut for t in f._weights)
    assert f._enter[1] is None and f._enter[0] is not None  # the second span had entered after the cut
    # the regeneration reads slower: the first span finishes its ramp and stays on until its new exit
    slow = [p for p in range(reading[cut - 2], 30) for _ in range(6)]
    again = drive(f, slow, start=cut)
    assert [w[0] for w in again][:4] == pytest.approx([2 / 3, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        f.observe(cut - 5, 0)  # frames before the kept readouts are refused


def test_laci_session_feeds_and_rolls_back_the_follower():
    s = session(seed=5)
    f = follower([(30, 34)], lead=1)
    s.follower = f
    stall = [p for p in range(0, 40) for _ in range(4)] + [39] * 200
    run_attempt(s, stall)
    assert f._frames[0] == 1 and f._frames[-1] == len(f._frames)  # step s of the first attempt is frame s
    for t in range(1, len(f._frames) + 1):
        f.weights(t)
    cont = s.finish_attempt()
    assert cont is not None
    resume = cont.history_rows
    assert f._frames[-1] < resume and max(f._weights) < resume
    run_attempt(s, [39] * 3 + list(range(39, TEXT_LEN)) * 2)
    assert f._frames[len([x for x in f._frames if x < resume])] == resume + 1  # continuation step 1 = frame resume + 1


def test_follower_refuses_mismatched_deltas():
    with pytest.raises(ValueError):
        SpanFollower([(1, 2)], torch.zeros(2, 4))
