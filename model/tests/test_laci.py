"""LACI (zonos3.tts.laci): detector, ACI centre and the per-request session the scheduler runs.

The detector is exercised on synthetic (row, pos) streams: a text of 100 positions (13 bins of 8, the last holding 4)
read at 4 frames per token, with the generated rows starting after 3 prefill rows.
"""

from __future__ import annotations

import math
import random
import unittest
from types import SimpleNamespace

import torch
from zonos3.tts.laci import (
    DURATION_MIN_STALL_FRAMES,
    PACE_MAX_RETRIES,
    PACE_SLACK_FRAMES,
    PACE_SMOOTH_STEPS,
    PACE_TOLERANCE,
    AciCenter,
    Detector,
    DurationTarget,
    Failure,
    LaciConfig,
    LaciHeads,
    LaciSession,
    PaceMonitor,
    RateBuckets,
)

TEXT_LEN = 100
FPT = 4
FIRST_ROW = 3
CFG = LaciConfig()


def reading(lo=0, hi=TEXT_LEN, start_row=FIRST_ROW, fpt=FPT):
    """Monotone stream: each pos in [lo, hi) held for `fpt` rows."""
    row = start_row
    for pos in range(lo, hi):
        for _ in range(fpt):
            yield row, pos
            row += 1


def detector(**kw):
    return Detector(TEXT_LEN, float(FPT), LaciConfig(**kw), min_rollback_row=FIRST_ROW)


def feed(det, stream):
    """Observe until the first failure; return (failure, last row fed)."""
    row = None
    for row, pos in stream:
        f = det.observe(row, pos)
        if f is not None:
            return f, row
    return None, row


class DetectorTest(unittest.TestCase):
    def test_geometry(self):
        det = detector()
        self.assertEqual(det.nb, 13)
        self.assertEqual(det.frontier, 0)
        self.assertEqual(det.counts, [0] * 13)
        self.assertEqual(det.pace_tok(), FPT)  # prior until warmed up
        self.assertIsNone(det.stop_row())
        self.assertEqual(det.last_pos_before(100), 0)

    def test_clean_reading(self):
        det = detector()
        last = None
        for row, pos in reading():
            self.assertIsNone(det.observe(row, pos), f"row {row} pos {pos}")
            last = row
        self.assertEqual(det.frontier, det.nb - 1)
        self.assertEqual(det.counts, [8 * FPT] * 12 + [4 * FPT])
        self.assertEqual(det.pace_tok(), FPT)
        # The last bin (4 tokens) is full at its final row: stop 15 frames later.
        self.assertEqual(det.stop_row(), last + CFG.tail_frames)
        self.assertIsNone(det.on_eos(last))

    def test_stop_row_unset_before_last_bin_full(self):
        det = detector()
        stream = list(reading())
        for row, pos in stream[:-1]:
            det.observe(row, pos)
        self.assertIsNone(det.stop_row())
        det.observe(*stream[-1])
        self.assertEqual(det.stop_row(), stream[-1][0] + CFG.tail_frames)

    def test_skip(self):
        det = detector()
        before = list(reading(0, 40))  # bins 0-4 fully read, frontier 4
        for row, pos in before:
            self.assertIsNone(det.observe(row, pos))
        self.assertEqual(det.frontier, 4)
        jump_row = before[-1][0] + 1
        # jump from bin 4 into bin 7 (pos 56), skipping bins 5 and 6
        f, detect_row = feed(det, reading(56, 100, start_row=jump_row))
        self.assertEqual(f.kind, "skip")
        self.assertEqual(f.bin, 5)
        self.assertEqual(f.frontier, 7)
        self.assertEqual(f.onset_row, jump_row)
        self.assertEqual(f.rollback_row, jump_row - CFG.rollback_margin_frames)
        self.assertEqual(f.detect_row, detect_row)
        share = det.pace_tok() * CFG.bin_tokens
        self.assertLessEqual(f.detect_row - jump_row, CFG.cover_frac * share)
        self.assertEqual(det.frontier, 4)  # the frontier did not advance

    def test_skip_rollback_clamped(self):
        det = detector()
        # jump straight from the start: onset is the first row, margin is clamped
        f, _ = feed(det, [(FIRST_ROW, 0)] + list(reading(40, 100, start_row=FIRST_ROW + 1)))
        self.assertEqual(f.kind, "skip")
        self.assertEqual(f.rollback_row, FIRST_ROW)
        self.assertEqual(f.bin, 1)

    def test_short_dip_is_not_a_skip(self):
        # bin 5 read for 5 frames (> skip_frac * share = 4.8) then bin 6 normally
        det = detector()
        for row, pos in reading(0, 40):
            det.observe(row, pos)
        row = 3 + 40 * FPT
        for pos in [40] * 5 + [48] * 40:
            self.assertIsNone(det.observe(row, pos))
            row += 1
        self.assertEqual(det.frontier, 6)

    def test_hallucination(self):
        det = detector()
        last_advance = None
        frontier = det.frontier
        for row, pos in reading(0, 32):  # frontier reaches bin 3
            self.assertIsNone(det.observe(row, pos))
            if det.frontier != frontier:
                frontier, last_advance = det.frontier, row
        self.assertEqual(frontier, 3)
        limit = max(CFG.stall_factor * FPT * CFG.bin_tokens, CFG.min_stall_frames)
        self.assertEqual(limit, 96)
        row = 3 + 32 * FPT
        f = None
        while f is None:
            f = det.observe(row, 31)  # parked in bin 3
            if f is None:
                self.assertLessEqual(row - last_advance, limit)
            row += 1
        self.assertEqual(f.kind, "hallucination")
        self.assertEqual(f.detect_row, last_advance + 97)
        self.assertEqual(f.onset_row, last_advance)
        self.assertEqual(f.rollback_row, last_advance - CFG.rollback_margin_frames)
        self.assertEqual(f.bin, 4)
        self.assertEqual(f.frontier, 3)

    def test_no_stall_after_text_done(self):
        det = detector()
        last = None
        for row, pos in reading():
            det.observe(row, pos)
            last = row
        for row in range(last + 1, last + 400):
            self.assertIsNone(det.observe(row, TEXT_LEN - 1))

    def test_truncate_matches_fresh(self):
        stream = list(reading())
        k = stream[150][0]
        a = detector()
        for row, pos in stream[:230]:
            a.observe(row, pos)
        a.truncate(k)
        b = detector()
        for row, pos in stream:
            if row < k:
                b.observe(row, pos)
        for det in (a, b):
            self.assertEqual(det.counts, b.counts)
        self.assertEqual(
            (a.counts, a.frontier, a.pace_tok(), a.stop_row()), (b.counts, b.frontier, b.pace_tok(), b.stop_row())
        )
        self.assertEqual(a.last_pos_before(k), stream[149][1])
        # re-observing the same rows leaves them identical, through to the end
        for row, pos in stream[150:]:
            self.assertIsNone(a.observe(row, pos))
            self.assertIsNone(b.observe(row, pos))
        self.assertEqual(
            (a.counts, a.frontier, a.pace_tok(), a.stop_row()), (b.counts, b.frontier, b.pace_tok(), b.stop_row())
        )

    def test_truncate_clears_done(self):
        det = detector()
        stream = list(reading())
        for row, pos in stream:
            det.observe(row, pos)
        self.assertIsNotNone(det.stop_row())
        det.truncate(stream[-20][0])
        self.assertIsNone(det.stop_row())

    def test_truncate_restores_stall_budget(self):
        det = detector()
        stream = list(reading(0, 32))
        for row, pos in stream:
            det.observe(row, pos)
        row = stream[-1][0] + 1
        f = None
        while f is None:
            f = det.observe(row, 31)
            row += 1
        det.truncate(f.rollback_row)
        # a regeneration from the rollback row gets a full stall budget from there
        self.assertGreaterEqual(det.last_advance_row, f.rollback_row)
        r = f.rollback_row
        for _ in range(96):
            self.assertIsNone(det.observe(r, 31))
            r += 1

    def test_repeated_truncate_replays_earlier_stall_budgets(self):
        # The live run after truncate(15) had its stall clock moved up to row 15; a later truncate must replay the
        # kept rows with that same bump, or row 25 (gap 11 from the old clock at row 14) would read as a stall.
        det = Detector(40, 1.0, LaciConfig(bin_tokens=2, min_stall_frames=10), min_rollback_row=FIRST_ROW)
        self.assertIsNone(det.observe(10, 0))
        self.assertIsNone(det.observe(14, 2))
        det.truncate(15)
        self.assertEqual(det.last_advance_row, 15)
        self.assertIsNone(det.observe(25, 2))  # gap 10 from the bump: not a stall
        det.truncate(30)
        self.assertEqual(det.last_advance_row, 30)
        self.assertEqual(det.counts[:2], [1, 2])

    def test_recover_after_skip(self):
        det = detector()
        for row, pos in reading(0, 40):
            det.observe(row, pos)
        jump_row = 3 + 40 * FPT
        f, _ = feed(det, reading(56, 100, start_row=jump_row))
        self.assertEqual(f.kind, "skip")
        self.assertEqual(det.last_pos_before(f.rollback_row), 37)
        det.truncate(f.rollback_row)
        self.assertEqual(det.frontier, 4)
        for row, pos in reading(37, 100, start_row=f.rollback_row):
            self.assertIsNone(det.observe(row, pos))
        self.assertEqual(det.frontier, det.nb - 1)
        self.assertIsNotNone(det.stop_row())

    def test_rows_must_increase(self):
        det = detector()
        det.observe(5, 0)
        with self.assertRaises(ValueError):
            det.observe(5, 0)
        with self.assertRaises(ValueError):
            det.observe(6, TEXT_LEN)

    def test_eos(self):
        det = detector()
        stream = list(reading(0, 40))
        for row, pos in stream:
            det.observe(row, pos)
        row = stream[-1][0] + 1
        f = det.on_eos(row)
        self.assertEqual(f.kind, "truncation")
        self.assertEqual((f.detect_row, f.onset_row), (row, row))
        self.assertEqual(f.rollback_row, row - CFG.rollback_margin_frames)
        self.assertEqual(f.bin, det.frontier + 1)
        self.assertIsInstance(f, Failure)
        # one unread bin is tolerated: EOS may fire while the tail is being read
        det = detector()
        for row, pos in reading(0, 96):  # frontier = nb-2 = 11
            det.observe(row, pos)
        self.assertEqual(det.frontier, det.nb - 2)
        self.assertIsNone(det.on_eos(row + 1))

    def test_short_text_single_bin(self):
        det = Detector(5, 4.0, CFG, min_rollback_row=3)
        self.assertEqual(det.nb, 1)
        self.assertIsNone(det.on_eos(10))
        for row, pos in reading(0, 5, fpt=1):
            det.observe(row, pos)


class AciCenterTest(unittest.TestCase):
    def test_follows_unit_slope(self):
        aci = AciCenter(5, 100)
        for t in range(1, 40):
            self.assertEqual(aci.update(5 + t), 5 + t)

    def test_monotone_and_bounded_step(self):
        aci = AciCenter(0, 100)
        prev = 0
        for t in range(1, 200):
            m = 0.5 * t if t < 120 else 60 - 0.3 * (t - 120)  # reading then retreat
            c = aci.update(m)
            self.assertGreaterEqual(c, prev)
            self.assertLessEqual(c - prev, 1)
            prev = c

    def test_follows_linear_m(self):
        aci = AciCenter(10, 200)
        for t in range(1, 100):
            m = 10 + 0.5 * t
            c = aci.update(m)
            self.assertLessEqual(abs(c - m), 1.5, f"t={t} m={m} c={c}")

    def test_holds_when_attention_stalls(self):
        aci = AciCenter(20, 100)
        for _ in range(30):
            c = aci.update(20.0)
        self.assertEqual(c, 20)

    def test_capped_at_text_end(self):
        aci = AciCenter(8, 10)
        for _ in range(30):
            c = aci.update(1000.0)
        self.assertEqual(c, 9)

    def test_c0_clamped(self):
        self.assertEqual(AciCenter(50, 10).update(9.0), 9)


# ------------------------------------------------------------------------------------------------------ session

C = 10  # codebooks
P = 3  # prefill rows
HEADS = LaciHeads(heads=((7, 4),), radius={(7, 4): 3}, frames_per_token=float(FPT))


def session(cfg=CFG, max_tokens=4000, seed=0, hold=False):
    return LaciSession(
        heads=HEADS,
        cfg=cfg,
        prefill_rows=torch.full((P, C + 1), -1, dtype=torch.int32),
        seed=seed,
        max_tokens=max_tokens,
        n_codebooks=C,
        hold=hold,
    )


def frame(tag):
    return torch.full((C + 1,), tag, dtype=torch.int32)


def run_attempt(s, positions, eos_at=None, eos_lag=0, tag=0):
    """Drive one attempt like the scheduler: the prefill-sampled row (unobserved), then per decode step the observer,
    the row, EOS bookkeeping (finish n_codebooks + 1 rows after the EOA row), a stop. Returns the rows drained."""
    probe = SimpleNamespace(text_len=TEXT_LEN, mask=s.probe_mask(), stop=False)
    drained = []
    s.append(frame(tag), None)
    drained += s.drain()
    countdown = None
    for step, pos in enumerate(positions, start=1):
        s.on_step(probe, step, pos, float(pos))
        eos = None
        if eos_at is not None and step >= eos_at:
            eos = eos_at - eos_lag
            countdown = C + 1 if countdown is None else countdown
        s.append(frame(tag * 100000 + step), eos)
        if countdown is not None:
            countdown -= 1
        if probe.stop or countdown == 0:
            break
        drained += s.drain()
    return drained


class SessionTest(unittest.TestCase):
    def test_clean_reading_ends_at_text_done_and_streams_rows_early(self):
        s = session()
        pos = [p for _, p in reading(start_row=0)] + [TEXT_LEN - 1] * 60
        drained = run_attempt(s, pos)
        self.assertIsNone(s.finish_attempt())
        self.assertEqual(s.end[0], "text_done")
        self.assertGreater(len(drained), 300)  # most rows went out while generating
        final = s.drain(final=True)
        self.assertTrue(final)  # the last row is always held for the final message
        self.assertEqual(s.eos_frame(), s.end[1] - P)
        rec = s.record()
        self.assertEqual((rec["attempts"], rec["end"], rec["failures"]), (0, "text_done", []))

    def test_rollback_builds_the_continuation(self):
        s = session(seed=5)
        stall = [p for _, p in reading(0, 40, start_row=0)] + [39] * 200
        run_attempt(s, stall)
        cont = s.finish_attempt()
        self.assertIsNotNone(cont)
        f = s.failures[-1]
        self.assertEqual(f["kind"], "hallucination")
        self.assertEqual(cont.rows.shape[0], f["rollback_row"])
        self.assertTrue(torch.equal(cont.rows[:P], torch.full((P, C + 1), -1, dtype=torch.int32)))
        self.assertEqual(cont.history_rows, f["rollback_row"] - P)
        self.assertEqual(cont.seed, 5 + 1000)
        self.assertEqual(cont.max_tokens, 4000 - (f["rollback_row"] - P))
        self.assertEqual(s.probe_mask() is not None, True)  # the regeneration runs under the ACI mask

    def test_truncation_on_early_eos(self):
        s = session()
        pos = [p for _, p in reading(0, 30, start_row=0)]
        run_attempt(s, pos, eos_at=len(pos) - 12, eos_lag=4)
        cont = s.finish_attempt()
        self.assertIsNotNone(cont)
        self.assertEqual(s.failures[-1]["kind"], "truncation")

    def test_emitted_rows_are_never_rolled_back(self):
        """Random readings with skips, stalls and early EOS: every row handed downstream is in the final stream."""
        self.assertGreater(random_sessions(self, lambda rng, cfg: session(cfg)), 20)

    def test_held_session_sends_nothing_before_the_result_is_final_and_never_clamps(self):
        self.assertGreater(random_sessions(self, lambda rng, cfg: session(cfg, hold=True)), 20)


def random_sessions(test, make_session, trials=120) -> int:
    """Drive sessions through random readings (skips, stalls, early EOS); assert that every row handed downstream
    is in the final stream. Returns the number of rollbacks."""
    rng = random.Random(0)
    rollbacks = 0
    for trial in range(trials):
        cfg = LaciConfig(max_retries=rng.choice([1, 3, 8]))
        s = make_session(rng, cfg)
        out = []
        attempt = 0
        while True:
            pos, p, n = [], s.detector.last_pos_before(s.attempt_start) if s.detector else 0, 0
            while p < TEXT_LEN and n < 700:
                r = rng.random()
                if r < 0.004:
                    p = min(TEXT_LEN - 1, p + rng.randint(17, 30))  # skip two or three bins
                elif r < 0.008:
                    pos += [p] * rng.randint(50, 120)  # stall
                pos += [min(p, TEXT_LEN - 1)] * rng.randint(2, 6)
                p += 1
                n = len(pos)
            pos += [TEXT_LEN - 1] * 40
            eos_at = rng.randint(20, len(pos) - 1) if rng.random() < 0.15 else None
            drained = run_attempt(s, pos, eos_at=eos_at, eos_lag=rng.randint(0, C - 1), tag=attempt + 1)
            if s.hold:
                test.assertEqual(drained, [], trial)
            out += drained
            if s.finish_attempt() is None:
                break
            rollbacks += 1
            attempt += 1
        out += s.drain(final=True)
        if s.hold:
            test.assertFalse(any(f["clamped"] for f in s.failures), trial)
        final = s.stream[P:]
        test.assertEqual(len(out), len(final), trial)
        test.assertTrue(all(torch.equal(a, b) for a, b in zip(out, final)), trial)
    return rollbacks


class CommitRowTest(unittest.TestCase):
    def test_bound_below_pending_onsets(self):
        det = detector()
        for row, pos in reading(0, 40):
            det.observe(row, pos)
        bound = det.commit_row(3 + 40 * FPT)
        self.assertLessEqual(bound, det.last_advance_row - CFG.rollback_margin_frames)
        self.assertGreaterEqual(bound, FIRST_ROW)

    def test_unbounded_once_text_done(self):
        det = detector()
        for row, pos in reading():
            det.observe(row, pos)
        row = 3 + TEXT_LEN * FPT
        while det.done_row is None:
            det.observe(row, TEXT_LEN - 1)
            row += 1
        self.assertEqual(det.commit_row(row), math.inf)


# ------------------------------------------------------------------------------------------------ duration targets

EDGES = (10.117, 12.123, 13.286, 14.25, 15.14, 16.079, 17.094, 18.489, 21.208)  # ZONOS3 Mini release speaking rate
RATES = RateBuckets(index=2, edges=EDGES, n_buckets=10)


def pace_run(monitor, positions, start_row=3, text_len=21, text_done_from=None):
    """Feed one pos per row from start_row; (row, failure) of the first failure, or None."""
    for k, pos in enumerate(positions):
        row = start_row + k
        done = text_done_from is not None and row >= text_done_from
        f = monitor.observe(row, pos, text_len, text_done=done)
        if f is not None:
            return row, f
    return None


def linear(frames, text_len=21, p0=3):
    """pos per frame of a reading that moves from p0 to EOS (text_len - 1) over `frames` frames."""
    return [p0 + int((text_len - 1 - p0) * k / frames) for k in range(frames)] + [text_len - 1]


class PaceMonitorTest(unittest.TestCase):
    def monitor(self, target=60):
        return PaceMonitor(target_frames=target, origin_row=3, min_rollback_row=3)

    def test_on_time_reading_and_its_tail_pass(self):
        m = self.monitor()
        self.assertIsNone(pace_run(m, linear(60) + [20] * 30))
        self.assertIsNotNone(m.read_row)

    def test_fast_when_the_text_is_read_too_early(self):
        m = self.monitor()
        row, f = pace_run(m, linear(36) + [20] * 30)
        self.assertEqual((f.kind, f.rollback_row, f.detect_row), ("pace_fast", 3, m.read_row))
        self.assertAlmostEqual(f.ratio, (row - 3) / 60)
        self.assertIsNone(pace_run(self.monitor(), linear(56) + [20] * 30))  # inside the tolerance

    def test_slow_at_the_deadline_extrapolates_the_ratio(self):
        row, f = pace_run(self.monitor(), linear(90))  # would take 1.5x
        self.assertEqual((f.kind, f.rollback_row), ("pace_slow", 3))
        self.assertEqual(row - 3, math.floor(60 + PACE_TOLERANCE * 60 + PACE_SLACK_FRAMES) + 1)  # first row past it
        self.assertAlmostEqual(f.ratio, 1.5, delta=0.1)

    def test_uneven_reading_that_ends_on_time_passes(self):
        # a long hold early (a slow word or a pause), then a catch-up: only the moment the text is read counts
        self.assertIsNone(pace_run(self.monitor(), [3] * 25 + linear(35, p0=4) + [20] * 10))

    def test_text_read_by_the_detector_or_never_reaching_the_last_token(self):
        hold = linear(60)[:55] + [18] * 60
        self.assertIsNone(pace_run(self.monitor(), hold, text_done_from=3 + 56))
        self.assertEqual(pace_run(self.monitor(), hold)[1].kind, "pace_slow")

    def test_single_argmax_jump_to_eos_is_not_reading_the_text(self):
        positions = linear(60)
        positions[10] = 20
        self.assertIsNone(pace_run(self.monitor(), positions))

    def test_stall_when_the_reading_stops_before_the_text_is_read(self):
        m = PaceMonitor(target_frames=100, origin_row=3, min_rollback_row=3)  # text_len 101: 1 frame / token
        cap = m.stall_frames(101)
        self.assertEqual(cap, CFG.stall_factor * CFG.bin_tokens * 100 / 101)
        reading_ = linear(100, text_len=101, p0=0)
        row, f = pace_run(m, reading_[:40] + [39] * 60, text_len=101)
        self.assertEqual(f.kind, "hallucination")
        self.assertIsInstance(f, Failure)  # a content failure: rolled back to its onset, not retaken
        self.assertLessEqual(abs(f.onset_row - (3 + 40)), PACE_SMOOTH_STEPS)
        self.assertEqual(f.rollback_row, f.onset_row - CFG.rollback_margin_frames)
        self.assertGreater(row - f.onset_row, cap)
        # a shorter hold (still inside the duration tolerance), or one after the text is read, is no stall
        m = PaceMonitor(target_frames=100, origin_row=3, min_rollback_row=3)
        self.assertIsNone(pace_run(m, reading_[:40] + [39] * 10 + reading_[40:] + [100] * 60, text_len=101))
        # a short target never goes under DURATION_MIN_STALL_FRAMES
        self.assertEqual(PaceMonitor(10, 3, 3).stall_frames(101), DURATION_MIN_STALL_FRAMES)

    def test_truncate_restarts_the_stall_clock(self):
        m = PaceMonitor(target_frames=100, origin_row=3, min_rollback_row=3)
        reading_ = linear(100, text_len=101, p0=0)
        row, f = pace_run(m, reading_[:40] + [39] * 60, text_len=101)
        m.truncate(f.rollback_row)
        # the regeneration may hold as long as a fresh attempt from the rollback row
        resumed = [39] * int(m.stall_frames(101))
        self.assertIsNone(pace_run(m, resumed, start_row=f.rollback_row, text_len=101))

    def test_truncate_to_the_first_row_starts_a_fresh_take(self):
        m = self.monitor()
        _, f = pace_run(m, linear(30) + [20] * 5)
        m.truncate(f.rollback_row)
        self.assertIsNone(m.read_row)
        self.assertIsNone(pace_run(m, linear(60)))


class RateBucketsTest(unittest.TestCase):
    def test_retry_bucket_follows_the_measured_ratio(self):
        self.assertAlmostEqual(RATES.rate(4), (14.25 + 15.14) / 2)
        self.assertEqual(RATES.retry_bucket(4, 1.3), 8)  # 14.7 B/s * 1.3 = 19.1 B/s
        self.assertEqual(RATES.retry_bucket(4, 0.77), 1)
        self.assertEqual(RATES.retry_bucket(4, 1.02), 5)  # at least one bucket in the asked direction
        self.assertEqual(RATES.retry_bucket(4, 0.99), 3)
        self.assertIsNone(RATES.retry_bucket(9, 1.4))
        self.assertIsNone(RATES.retry_bucket(0, 0.7))
        self.assertEqual((RATES.first, RATES.last, RATES.row(9)), (0, 9, 9))

    def test_extended_scale_retries_into_steered_buckets(self):
        self.assertAlmostEqual(EXT.rate(0), (9.2 + 10.117) / 2)  # bucket 0 is no longer an open end
        self.assertEqual(EXT.retry_bucket(0, 0.7), -2)  # 6.8 B/s
        self.assertEqual(EXT.retry_bucket(9, 1.4), 10)
        self.assertEqual(EXT.retry_bucket(4, 1.3), 8)  # inside the row as on the row's own scale
        self.assertIsNone(EXT.retry_bucket(10, 1.2))
        self.assertEqual([EXT.row(b) for b in (-2, -1, 0, 9, 10)], [10, 10, 0, 9, 10])  # steered: row unknown


EXT = RateBuckets(index=2, edges=(8.4, 9.2, *EDGES, 23.3), n_buckets=10, first=-2)  # buckets -2..10


def paced_session(target, adjust_rate=True, bucket=4, cfg=CFG, rates=RATES, rate_deltas=None):
    return LaciSession(
        heads=HEADS,
        cfg=cfg,
        prefill_rows=torch.full((P, C + 1), -1, dtype=torch.int32),
        seed=0,
        max_tokens=4000,
        n_codebooks=C,
        duration=DurationTarget(target, adjust_rate),
        rates=rates,
        condition_values=[10, 10, rates.row(bucket)],
        bucket=bucket,
        rate_deltas=rate_deltas,
    )


class DurationSessionTest(unittest.TestCase):
    READING = [p for _, p in reading(start_row=0)]  # 400 frames, then the last token

    def test_fast_take_is_retaken_from_the_first_row_in_a_slower_bucket(self):
        s = paced_session(target=600)  # the clean reading takes 400 frames: ratio ~0.67
        drained = run_attempt(s, self.READING + [TEXT_LEN - 1] * 60)
        self.assertEqual(drained, [])  # held: the whole take may still be redone
        cont = s.finish_attempt()
        self.assertIsNotNone(cont)
        f = s.failures[-1]
        self.assertEqual((f["kind"], f["rollback_row"], f["ignored"]), ("pace_fast", P, False))
        self.assertEqual(cont.rows.shape[0], P)  # a fresh take
        self.assertLess(cont.condition_values[2], 4)
        self.assertEqual(cont.condition_values[:2], [10, 10])
        self.assertIsNone(s.probe_mask())  # no ACI mask: the reading was fine
        self.assertEqual(cont.seed, 1000)

    def test_on_time_take_streams_after_the_text_is_read(self):
        s = paced_session(target=400)
        drained = run_attempt(s, self.READING + [TEXT_LEN - 1] * 60)
        self.assertGreater(len(drained), 0)
        self.assertIsNone(s.finish_attempt())
        rec = s.record()
        self.assertEqual((rec["pace_retries"], rec["rate_bucket"], rec["end"]), (0, 4, "text_done"))

    def test_emitted_rows_are_never_rolled_back_with_a_duration_target(self):
        def make(rng, cfg):
            return paced_session(
                rng.choice([150, 300, 450, 700]), adjust_rate=rng.random() < 0.8, bucket=rng.randint(0, 9), cfg=cfg
            )

        self.assertGreater(random_sessions(self, make), 20)

    def test_early_eos_retries_one_bucket_slower(self):
        early = [p for _, p in reading(0, 30, start_row=0)]
        s = paced_session(target=400, bucket=9)
        run_attempt(s, early, eos_at=len(early) - 12, eos_lag=4)
        cont = s.finish_attempt()
        f = s.failures[-1]
        self.assertEqual((f["kind"], f["retry_bucket"]), ("truncation", 8))
        self.assertEqual(cont.condition_values, [10, 10, 8])
        self.assertIsNotNone(s.probe_mask())  # still a content rollback: under the ACI mask
        self.assertEqual(s.pace_retries, 0)
        for bucket, adjust in ((0, True), (9, False)):  # nothing slower left / a set bucket: reseed in place
            s = paced_session(target=400, bucket=bucket, adjust_rate=adjust)
            run_attempt(s, early, eos_at=len(early) - 12, eos_lag=4)
            cont = s.finish_attempt()
            self.assertEqual((s.failures[-1]["kind"], s.failures[-1]["retry_bucket"]), ("truncation", None))
            self.assertEqual(cont.condition_values[2], bucket)
            self.assertFalse(cont.swap_delta)  # the row's own scale keeps the request's steering delta

    def test_extended_scale_retakes_switch_the_row_and_the_steering_delta(self):
        deltas = [None if 0 <= b < 10 else torch.full((4,), float(b)) for b in range(EXT.first, EXT.last + 1)]
        s = paced_session(target=600, bucket=0, rates=EXT, rate_deltas=deltas)  # read in 400 frames: ratio ~0.67
        run_attempt(s, self.READING + [TEXT_LEN - 1] * 60)
        cont = s.finish_attempt()
        self.assertEqual((s.failures[-1]["kind"], s.failures[-1]["retry_bucket"]), ("pace_fast", -2))
        self.assertEqual(cont.condition_values, [10, 10, 10])  # steered: the rate row unknown
        self.assertTrue(cont.swap_delta)
        self.assertTrue(torch.equal(cont.steering_delta, torch.full((4,), -2.0)))
        run_attempt(s, self.READING + [TEXT_LEN - 1] * 60)
        cont = s.finish_attempt()  # still fast, nothing slower left on the scale: only recorded
        self.assertIsNone(cont)
        self.assertEqual(s.record()["rate_bucket"], -2)
        # a slow take in a steered bucket comes back into the row: its condition row and no tempo delta
        s = paced_session(target=200, bucket=-1, rates=EXT, rate_deltas=deltas)
        run_attempt(s, self.READING + [TEXT_LEN - 1] * 60)
        cont = s.finish_attempt()
        self.assertEqual(s.failures[-1]["kind"], "pace_slow")
        self.assertGreaterEqual(s.rate_bucket, 0)
        self.assertEqual(cont.condition_values[2], s.rate_bucket)
        self.assertTrue(cont.swap_delta)
        self.assertIsNone(cont.steering_delta)
        # truncation at the row's slowest bucket goes one slower: into the steered bucket below
        early = [p for _, p in reading(0, 30, start_row=0)]
        s = paced_session(target=400, bucket=0, rates=EXT, rate_deltas=deltas)
        run_attempt(s, early, eos_at=len(early) - 12, eos_lag=4)
        cont = s.finish_attempt()
        self.assertEqual((s.failures[-1]["kind"], s.failures[-1]["retry_bucket"]), ("truncation", -1))
        self.assertEqual(cont.condition_values[2], 10)
        self.assertTrue(torch.equal(cont.steering_delta, torch.full((4,), -1.0)))
        with self.assertRaises(ValueError):
            paced_session(target=400, bucket=0, rates=EXT, rate_deltas=deltas[:-1])

    def test_stall_limit_follows_the_target_pace(self):
        # a text read at 1 frame / token with a 35-frame hold early on: under the detector's 45-frame floor, but over
        # three bins at the 100-frame target's pace (24 frames)
        heads = LaciHeads(heads=((7, 4),), radius={(7, 4): 3}, frames_per_token=1.0)
        clean = [p for _, p in reading(0, TEXT_LEN, start_row=0, fpt=1)] + [TEXT_LEN - 1] * 30
        held = clean[:30] + [29] * 35 + clean[30:]
        plain = LaciSession(
            heads=heads,
            cfg=CFG,
            prefill_rows=torch.full((P, C + 1), -1, dtype=torch.int32),
            seed=0,
            max_tokens=4000,
            n_codebooks=C,
        )
        run_attempt(plain, held)
        self.assertIsNone(plain.finish_attempt())
        self.assertEqual(plain.failures, [])

        def timed():
            return LaciSession(
                heads=heads,
                cfg=CFG,
                prefill_rows=torch.full((P, C + 1), -1, dtype=torch.int32),
                seed=0,
                max_tokens=4000,
                n_codebooks=C,
                duration=DurationTarget(100),
                rates=RATES,
                condition_values=[10, 10, 4],
            )

        s = timed()
        run_attempt(s, held)
        cont = s.finish_attempt()
        f = s.failures[-1]
        self.assertEqual((f["kind"], f["retry_bucket"]), ("hallucination", None))
        self.assertLess(f["detect_row"], P + 30 + 35)  # caught inside the hold
        self.assertLessEqual(f["rollback_row"], P + 30)  # back to before it
        self.assertEqual(cont.condition_values[2], 4)  # a stall is no rate miss: same bucket
        self.assertIsNotNone(s.probe_mask())  # regenerated under the ACI mask
        on_time = timed()  # the same reading without the hold passes
        run_attempt(on_time, clean)
        self.assertIsNone(on_time.finish_attempt())
        self.assertEqual((on_time.failures, on_time.end[0]), ([], "text_done"))

    def test_a_miss_with_no_bucket_left_is_only_recorded(self):
        s = paced_session(target=600, bucket=0)  # too fast, and already in the slowest bucket
        run_attempt(s, self.READING + [TEXT_LEN - 1] * 60)
        self.assertIsNone(s.finish_attempt())
        self.assertEqual([(f["kind"], f["ignored"]) for f in s.record()["failures"]], [("pace_fast", True)])

    def test_set_bucket_only_records_the_miss(self):
        s = paced_session(target=600, adjust_rate=False)
        drained = run_attempt(s, self.READING + [TEXT_LEN - 1] * 60)
        self.assertGreater(len(drained), 0)  # nothing to hold: the take is never redone for its length
        self.assertIsNone(s.finish_attempt())
        rec = s.record()
        self.assertEqual([(f["kind"], f["ignored"]) for f in rec["failures"]], [("pace_fast", True)])
        self.assertEqual((rec["pace_retries"], rec["rate_bucket"]), (0, 4))

    def test_retries_stop_after_the_pace_budget(self):
        s = paced_session(target=480)  # ratio ~0.83: each retry a bucket or two slower, never off the bottom
        buckets = []
        for _ in range(PACE_MAX_RETRIES):
            run_attempt(s, self.READING + [TEXT_LEN - 1] * 60)
            cont = s.finish_attempt()
            self.assertIsNotNone(cont)
            buckets.append(cont.condition_values[2])
        self.assertEqual(buckets, sorted(buckets, reverse=True))
        self.assertLess(buckets[-1], buckets[0])
        out = run_attempt(s, self.READING + [TEXT_LEN - 1] * 60)
        self.assertIsNone(s.finish_attempt())
        rec = s.record()
        self.assertEqual(rec["pace_retries"], PACE_MAX_RETRIES)
        self.assertEqual([f["ignored"] for f in rec["failures"]], [False] * PACE_MAX_RETRIES + [True])
        out += s.drain(final=True)
        self.assertGreater(len(out), 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(out, s.stream[P:])))  # the last take is the result
        self.assertEqual(len(out), len(s.stream) - P)
