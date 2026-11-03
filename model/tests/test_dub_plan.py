from __future__ import annotations

import math

import numpy as np
import pytest

from zonos3.dub import plan as P

EDGES = [10.117, 12.123, 13.286, 14.25, 15.14, 16.079, 17.094, 18.489, 21.208]


def test_text_units_french_merges_dash_and_roundtrips():
    text = "Les requins — existent, bien."
    units = P.text_units(text, "fr")
    assert [u["text"] for u in units] == ["Les", "requins —", "existent,", "bien."]
    assert P.span_text(text, units, 0, len(units)) == text
    assert P.span_text(text, units, 2, 2) == ""


def test_text_units_leading_punctuation_joins_first_unit():
    units = P.text_units("— Bonjour monde", "fr")
    assert units[0]["text"] == "— Bonjour"


def test_text_units_chinese_one_per_han_char():
    text = "鲨鱼确实存在，但很少攻击人类。"
    units = P.text_units(text, "zh")
    assert len(units) == 13
    assert units[5]["text"] == "在，"
    ba = P.break_after(text, units)
    assert ba[5] and ba[-1] and not ba[4]
    assert P.span_text(text, units, 0, len(units)) == text


def test_text_units_unspaced_latin_run_is_one_unit():
    units = P.text_units("我用iPhone15手机", "zh")
    assert [u["text"] for u in units] == ["我", "用", "iPhone15", "手", "机"]


def test_unit_bytes_cover_text():
    text = "Les requins — existent, bien."
    units = P.text_units(text, "fr")
    b = P.unit_bytes(text, units)
    assert sum(b) == len(text.encode())
    assert b[0] == len("Les ".encode())


def test_ref_segments_split_on_pause_and_zero_width():
    w = [
        {"start": 0.0, "end": 0.3},
        {"start": 0.35, "end": 0.6},
        {"start": 1.2, "end": 1.2},  # zero width: stays with its predecessor
        {"start": 1.25, "end": 1.5},  # 0.65 s after the last wordy end (0.6) -> split
        {"start": 1.6, "end": 1.9},  # 0.1 s gap -> no split
    ]
    assert P.ref_segments(w, 0.3) == [(0, 3), (3, 5)]
    assert P.ref_segments(w, 5.0) == [(0, 5)]


def test_ref_segments_zero_width_never_starts_split():
    w = [{"start": 0.0, "end": 0.3}, {"start": 2.0, "end": 2.0}, {"start": 2.01, "end": 2.4}]
    # word 1 is zero-width -> stays in segment 0; word 2 follows a 1.7 s gap from wordy end 0.3 -> split before it
    assert P.ref_segments(w, 0.3) == [(0, 2), (2, 3)]


def test_project_segments_crossing_links_inside_segment():
    links = [(0, 0), (1, 2), (2, 1), (3, 3), (4, 4), (5, 5)]
    got = P.project_segments([(0, 3), (3, 6)], [(0, 1), (1.5, 2.5)], links, [5] * 6, [False, False, True, False, False, True])
    assert got == [(0, 3), (3, 6)]


def test_project_segments_unlinked_unit_goes_to_break_side():
    # 5 units; unit 2 is unlinked. break_after[1] true -> boundary 2 is free, boundary 3 costs 0.5.
    links = [(0, 0), (1, 1), (2, 3), (3, 4)]
    kw = dict(unit_bytes=[1] * 5)
    got = P.project_segments([(0, 2), (2, 4)], [(0, 1), (1, 2)], links, break_after=[False, True, False, False, True], **kw)
    assert got[0] == (0, 2)
    got = P.project_segments([(0, 2), (2, 4)], [(0, 1), (1, 2)], links, break_after=[False, False, True, False, True], **kw)
    assert got[0] == (0, 3)


def test_project_segments_unlinked_middle_segment_is_empty():
    links = [(0, 0), (1, 1), (4, 2), (5, 3)]
    got = P.project_segments(
        [(0, 2), (2, 4), (4, 6)], [(0, 1), (1.5, 1.6), (2, 3)], links, [1] * 4, [True] * 4
    )
    assert got == [(0, 2), (2, 2), (2, 4)]


def test_project_segments_no_links_is_proportional():
    got = P.project_segments([(0, 1), (1, 2)], [(0, 1), (1, 4)], [], [1] * 8, [True] * 8)
    assert got == [(0, 2), (2, 8)]


def test_rate_bucket_release_edges():
    row = P.RateScale(EDGES)
    assert (row.first, row.last) == (0, 9)
    assert row.bucket(14.25) == 4
    assert row.bucket(10.0) == 0
    assert row.bucket(30.0) == 9


BELOW = (9.2, 8.4, 7.7)  # nominal lower edges of buckets 0, -1, -2
ABOVE = (23.3, 25.5)  # upper edges of buckets 9, 10


def test_extended_scale_continues_the_row_with_the_steered_buckets():
    s = P.RateScale.extended(EDGES, BELOW, ABOVE, {-2: -1.0, -1: -0.9, 10: 1.4})
    assert (s.first, s.last) == (-2, 10)
    assert s.edges == (8.4, 9.2, *EDGES, 23.3)
    assert [s.bucket(r) for r in (5.0, 8.4, 9.0, 9.5, 10.117, 21.0, 22.0, 40.0)] == [-2, -1, -1, 0, 1, 8, 9, 10]
    assert s.rate(0) == pytest.approx((9.2 + 10.117) / 2)  # bucket 0 is no longer an open end
    one_sided = P.RateScale.extended(EDGES, BELOW, ABOVE, {10: 1.4})
    assert (one_sided.first, one_sided.last, one_sided.edges[0]) == (0, 10, EDGES[0])
    assert P.RateScale.extended(EDGES, BELOW, ABOVE, {}) == P.RateScale(tuple(EDGES))
    with pytest.raises(ValueError, match="contiguous"):
        P.RateScale.extended(EDGES, BELOW, ABOVE, {-2: -1.0})
    with pytest.raises(ValueError, match="no nominal edges"):
        P.RateScale.extended(EDGES, BELOW, ABOVE, {10: 1.0, 11: 1.5, 12: 2.0})


def test_candidate_plan_bucket_zero():
    plan = P.candidate_plan(0, P.RateScale(EDGES), 7, 3)
    assert [c["bucket"] for c in plan] == [0, 0, 1, 1, 2, 2]
    assert [c["offset"] for c in plan] == [0, 0, 1, 1, 2, 2]
    seeds = [c["seed"] for c in plan]
    assert len(set(seeds)) == len(seeds)
    assert seeds[0] == 7 + 300
    # with steered buckets below the row the plan spreads into them, seeds unchanged per offset
    steered = P.candidate_plan(0, P.RateScale.extended(EDGES, BELOW, ABOVE, {-1: -0.9}), 7, 3)
    assert [c["bucket"] for c in steered] == [0, 0, -1, -1, 1, 1, 2, 2]
    assert [c["seed"] for c in steered if c["bucket"] >= 0] == seeds


def test_candidate_plan_middle_has_all_offsets():
    plan = P.candidate_plan(5, P.RateScale(EDGES), 0, 0)
    assert [c["offset"] for c in plan][::2] == [0, -1, 1, -2, 2]


def test_pinned_candidate_plan_keeps_the_bucket_and_the_candidate_count():
    plan = P.pinned_candidate_plan(3, 7, 2)
    assert len(plan) == len(P.CANDIDATE_BUCKET_OFFSETS) * P.CANDIDATE_SEEDS
    assert {c["bucket"] for c in plan} == {3} and {c["offset"] for c in plan} == {0}
    assert len({c["seed"] for c in plan}) == len(plan)


def cand(speech_s, capped=False, end="eos"):
    return {"speech_s": speech_s, "capped": capped, "end": end}


def test_choose_candidate():
    assert P.choose_candidate([cand(3.0), cand(2.1), cand(1.0)], 2.0, math.inf) == 1
    # room penalty flips the choice
    assert P.choose_candidate([cand(2.4), cand(1.9)], 2.0, math.inf) == 1
    assert P.choose_candidate([cand(2.05), cand(1.7)], 2.0, 2.0) == 1
    # capped loses to a slightly worse uncapped take
    assert P.choose_candidate([cand(2.0, capped=True), cand(2.3)], 2.0, math.inf) == 1
    assert P.choose_candidate([cand(2.0, end="max_tokens"), cand(2.3)], 2.0, math.inf) == 1
    assert P.choose_candidate([None, cand(0.0), None], 2.0, math.inf) is None
    assert P.choose_candidate([None, cand(1.0)], 2.0, math.inf) == 1


def test_place_segments_overrun_shifts_next():
    items = [
        {"ref_start": 0.0, "ref_end": 1.0, "take_len": 4.0, "s0": 0.3, "s1": 2.3},  # 2 s speech for 1 s slot
        {"ref_start": 1.5, "ref_end": 2.5, "take_len": 3.0, "s0": 0.2, "s1": 1.2},
    ]
    placed, tl = P.place_segments(items, 3.0)
    assert placed[0]["shift"] == 0.0
    assert placed[0]["speech_end"] == pytest.approx(2.0)
    assert placed[1]["shift"] == pytest.approx(2.0 + 0.05 - 1.5)
    assert placed[1]["start_error"] == placed[1]["shift"]
    assert all(p["place_start"] >= 0 for p in placed)
    assert tl >= 3.0
    assert placed[0]["duration_error"] == pytest.approx(1.0)
    # no overrun -> no shift beyond the reference start
    ok, _ = P.place_segments([items[1]], 3.0)
    assert ok[0]["shift"] == 0.0


def test_place_segments_negative_place_start_is_cropped():
    items = [{"ref_start": 0.0, "ref_end": 1.0, "take_len": 2.0, "s0": 0.3, "s1": 1.0}]
    placed, _ = P.place_segments(items, 1.0)
    p = placed[0]
    assert p["place_start"] == 0.0
    # speech lands on its start time inside the placed crop
    assert p["speech_start"] - p["place_start"] == pytest.approx(0.3 - p["crop_start"])


def test_timeline_extends_past_reference():
    items = [{"ref_start": 0.5, "ref_end": 1.0, "take_len": 5.0, "s0": 0.0, "s1": 4.0}]
    placed, tl = P.place_segments(items, 1.0)
    assert tl == pytest.approx(placed[0]["place_start"] + placed[0]["crop_end"] - placed[0]["crop_start"])
    assert tl > 4.5


def test_voice_prompt_span():
    segs = [P.RefSegment(1.0, 5.0), P.RefSegment(6.0, 15.0), P.RefSegment(16.0, 30.0)]
    a, b = P.voice_prompt_span(segs, 60.0)
    assert a == pytest.approx(0.9)
    assert b == pytest.approx(15.1)  # the 30 s end would exceed 20 s of prompt
    one = [P.RefSegment(0.0, 30.0)]
    a, b = P.voice_prompt_span(one, 60.0)
    assert (a, b) == (0.0, 20.0)
    a, b = P.voice_prompt_span(one, 12.0)
    assert b == 12.0


def test_speech_span_aligner_and_vad_fallback():
    assert P.speech_span([{"start": 0.2, "end": 0.3}, {"start": 0.5, "end": 0.7}, {"start": 0.9, "end": 0.9}]) == (0.2, 0.7)
    regions = [[0.3, 0.45], [0.5, 0.6]]
    assert P.speech_span([], regions) == (0.3, 0.6)
    unplaced = {"start": 0.0, "end": 0.95, "placed": False}  # FuseAlign's fallback times: not speech evidence
    assert P.speech_span([unplaced, {"start": 0.4, "end": 0.5, "placed": True}], regions) == (0.4, 0.5)
    assert P.speech_span([unplaced], regions) == (0.3, 0.6)
    assert P.speech_span([unplaced]) is None
    assert P.speech_span([], []) is None


def test_word_windows_cover_overlap_and_cap():
    counts = [3] * 400
    wins = P.word_windows(counts)
    assert wins[0][0] == 0 and wins[-1][1] == 400
    for s, e in wins:
        assert sum(counts[s:e]) <= P.WINDOW_MAX_WP
    for (s0, e0), (s1, e1) in zip(wins, wins[1:]):
        assert s1 < e0 and s1 > s0
    covered = set()
    for s, e in wins:
        covered.update(range(s, e))
    assert covered == set(range(400))
    with pytest.raises(ValueError, match="510 wordpieces"):
        P.word_windows([2, 600])
    assert P.word_windows([]) == []
    assert P.word_windows([0, 0]) == [(0, 2)]


def test_window_of_words_picks_most_central():
    wins = [(0, 10), (7, 17)]
    owner = P.window_of_words(wins, 17)
    assert owner[0] == 0 and owner[16] == 1
    assert owner[7] == 0 and owner[9] == 1  # 7: margin 7 vs 0; 9: margin 1 vs 2
    assert owner[8] == 0  # margin 1 vs 1 -> tie goes to the earlier window


def test_band_mask():
    assert P.band_mask(30, 40).all()  # max |p*w2/w1 - q| = 39 <= BAND_MIN_WP
    m = P.band_mask(1000, 1000)
    assert not m[0, 999] and m[0, 0] and m[500, 500]


def test_lip_map_inverts_and_extends_with_the_end_offsets():
    # a span from 2 s at 62.5 frames / s that starts 0.1 s late (dub played earlier) and ends 0.2 s early
    rate = 62.5
    n = 250  # 4 s
    src = np.linspace(-0.1 * rate, n - 1 + 0.2 * rate, n)
    m = P.LipMap.from_frames(2.0, rate, src)
    assert m.out_t[0] == 2.0 and m.out_t[-1] == pytest.approx(2.0 + (n - 1) / rate)
    t = np.array([0.0, 1.0, 2.5, 4.0, 5.9, 8.0])
    assert np.allclose(m.out_time(m.src_time(t)), t, atol=1e-6)
    assert m.src_time(0.0) == pytest.approx(0.0 - 0.1)  # before the span: the start offset
    assert m.src_time(8.0) == pytest.approx(8.0 + 0.2)  # after it: the end offset
    assert m.shifts()[0] == pytest.approx(0.1, abs=1e-5) and m.shifts()[-1] == pytest.approx(-0.2, abs=1e-5)


def test_lip_map_makes_a_dp_plateau_strictly_increasing():
    m = P.LipMap.from_frames(0.0, 62.5, [0, 1, 3, 2, 2, 5])  # a dip and a plateau
    assert np.all(np.diff(m.src_t) > 0)
    assert m.out_time(m.src_time(0.05)) == pytest.approx(0.05, abs=1e-6)


SR = 44100


def harmonic(seconds, f0=150.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6)) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t))


def dominant_hz(y, sr=SR):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.argmax(spec) * sr / len(y)


def test_wsola_slope_one_reproduces_the_shifted_input():
    x = harmonic(1.0)
    assert np.allclose(P.wsola(x, lambda t: t, len(x)), x, atol=1e-9)
    y = P.wsola(x, lambda t: t + 300, len(x) - 300)
    assert np.allclose(y, x[300:], atol=1e-9)


@pytest.mark.parametrize("rate", [0.8, 1.25])
def test_wsola_rate_change_keeps_pitch_and_level(rate):
    x = harmonic(2.0)
    n_out = int(len(x) / rate)
    y = P.wsola(x, lambda t: t * rate, n_out)
    assert len(y) == n_out
    core = y[2000 : n_out - 2000]
    assert abs(dominant_hz(core) - 150.0) < 2.0  # a resampling warp would move it to 150 * rate
    level = np.sqrt((core**2).mean()) / np.sqrt((x[2000:-2000] ** 2).mean())
    assert 0.95 < level < 1.05


def test_wsola_follows_a_time_varying_map():
    # tone bursts at known source times; slope 0.8 for the first second, 1.25 after
    t = np.arange(int(2.0 * SR)) / SR
    centres = [0.3, 0.7, 1.3, 1.7]
    x = sum(np.exp(-0.5 * ((t - c) / 0.03) ** 2) * np.sin(2 * np.pi * 220 * t) for c in centres)
    out_knots = np.array([0.0, 1.0 / 0.8, 1.0 / 0.8 + 1.0 / 1.25])  # output seconds
    src_knots = np.array([0.0, 1.0, 2.0])
    y = P.wsola(x, lambda n: np.interp(n / SR, out_knots, src_knots) * SR, int(out_knots[-1] * SR))
    env = np.abs(y)
    for c in centres:
        expected = np.interp(c, src_knots, out_knots)
        lo, hi = int((expected - 0.12) * SR), int((expected + 0.12) * SR)
        found = (np.arange(lo, hi) * env[lo:hi]).sum() / env[lo:hi].sum() / SR
        assert abs(found - expected) < 0.003, (c, found, expected)  # no drift: well inside the 11.6 ms tolerance


def test_lip_warp_scores_stretch_and_squeeze_per_span():
    # face track over output 1-3.5 s: the first 1.25 s of output play 1 s of dub (stretched 25 %), the next 1.25 s
    # play 1.5625 s of dub (squeezed 20 %); outside it the offsets of the ends hold (rate 1)
    m = P.LipMap(out_t=np.array([1.0, 2.25, 3.5]), src_t=np.array([1.0, 2.0, 3.5625]))
    first = P.lip_warp(m, 1.2, 1.8)
    assert first == pytest.approx({"lip_scale": 1.25, "lip_stretch": 1.25, "lip_squeeze": 1.0, "lip_warp": 1.25})
    second = P.lip_warp(m, 2.3, 3.4)  # 0.8x the length: played 1.25x faster
    assert second == pytest.approx({"lip_scale": 0.8, "lip_stretch": 1.0, "lip_squeeze": 1.25, "lip_warp": 1.25})
    both = P.lip_warp(m, 1.2, 3.4)
    assert both["lip_stretch"] == pytest.approx(1.25) and both["lip_squeeze"] == pytest.approx(1.25)
    assert 0.8 < both["lip_scale"] < 1.25
    outside = P.lip_warp(m, 4.0, 5.0)  # after the track: shifted, not scaled
    assert outside == pytest.approx({"lip_scale": 1.0, "lip_stretch": 1.0, "lip_squeeze": 1.0, "lip_warp": 1.0})


def test_lip_warp_caps_a_warp_that_stands_the_dub_still():
    m = P.LipMap.from_frames(0.0, 62.5, [0] * 40 + list(range(1, 100)))  # the dub held for 0.64 s, then played
    assert P.lip_warp(m, 0.0, 1.5)["lip_stretch"] == pytest.approx(P.LIP_SCALE_MAX)


def test_guard_lip_frames_bounds_shift_rate_and_start():
    rate = 62.5
    n = 400
    # an AV-sync path that skips the first 1.6 s, then stands still for 1 s, then runs on
    src = np.concatenate([np.full(1, 100.0), np.linspace(100, 160, 150), np.full(62, 160.0), np.linspace(160, 400, n - 213)])
    out = P.guard_lip_frames(src, rate, max_rate=1.5, max_shift_s=0.4)
    steps = np.diff(out)
    assert out[0] <= 0.0  # no advance at the track start: the dub's first words stay on the timeline
    assert steps.min() >= 1 / 1.5 - 1e-9 and steps.max() <= 1.5 + 1e-9
    assert np.abs(out - np.arange(n)).max() <= 0.4 * rate + 1.5  # within the slack of the rate bound
    m = P.LipMap.from_frames(0.0, rate, out)
    w = P.lip_warp(m, 0.0, n / rate)
    assert w["lip_stretch"] <= 1.5 + 1e-6 and w["lip_squeeze"] <= 1.5 + 1e-6
