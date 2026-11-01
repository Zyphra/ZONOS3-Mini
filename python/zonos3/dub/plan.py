"""Pure planning logic of the LACI dubbing tool (stdlib + numpy only).

Imported by the server process and by the alignment worker (which runs in a separate venv).
"""

from __future__ import annotations

import bisect
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

REF_LANGUAGES = {  # code -> Qwen3-ASR / ForcedAligner language name
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ru": "Russian",
    "yue": "Cantonese",
}
TARGET_LANGUAGES = {  # code -> (label, zonos3 TTS language for text normalization)
    "en": ("English", "en_us"),
    "fr": ("French", "fr_fr"),
    "de": ("German", "de"),
    "es": ("Spanish", "es"),
    "it": ("Italian", "it"),
    "pt": ("Portuguese", "pt_br"),
    "ja": ("Japanese", "ja"),
    "zh": ("Chinese", "cmn"),
    "ko": ("Korean", "ko"),
}
UNSPACED = frozenset({"ja", "zh", "yue"})
MAX_REF_SECONDS = 180.0  # qwen_asr MAX_FORCE_ALIGN_INPUT_SECONDS (one aligner forward)
MIN_REF_SECONDS = 0.5  # qwen_asr MIN_ASR_INPUT_SECONDS
MAX_VIDEO_BYTES = 1 << 30
DEFAULT_MIN_PAUSE_S = 0.30
MIN_PAUSE_RANGE = (0.10, 2.00)
MAX_SEGMENTS = 120
CANDIDATE_BUCKET_OFFSETS = (0, -1, 1, -2, 2)
CANDIDATE_SEEDS = 2
PRE_PAD_S, POST_PAD_S, MIN_GAP_S, FADE_S = 0.04, 0.06, 0.05, 0.01
VOICE_PROMPT_MAX_S = 20.0
FRAME_RATE = 15.0
SAMPLE_RATE = 44100
BREAK_PUNCT = ",.;:!?…、。，；：！？"
WINDOW_MAX_WP = 510  # mBERT positions minus [CLS]/[SEP]
WINDOW_STRIDE_FRACTION = 0.75
BAND_MIN_WP, BAND_FRACTION = 40, 0.2
_CJK_BLOCKS = (
    (0x3040, 0x30FF),
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF66, 0xFF9F),
)
_TOKEN = re.compile(r"\S+")


@dataclass
class RefSegment:
    ref_start: float
    ref_end: float
    text: str = ""
    bucket: int | None = None  # speaking-rate bucket set by the user; None = from the text's rate


def _wordy(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LN" or ch == "'"


def _cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_BLOCKS)


def text_units(text: str, lang: str) -> list[dict]:
    """Split the target text into word units ``{"text", "start", "end"}`` (char offsets); trailing punctuation sticks to its unit."""
    spans: list[list[int]] = []  # [start, end)
    lead: int | None = None  # start of non-wordy chars seen before the first unit

    def attach(start: int, end: int) -> None:
        nonlocal lead
        if spans:
            spans[-1][1] = end
        elif lead is None:
            lead = start

    if lang in UNSPACED:
        run_end = -1  # end of the open run of non-CJK wordy chars
        for i, ch in enumerate(text):
            if ch.isspace():
                continue
            if _wordy(ch):
                if _cjk(ch):
                    spans.append([i, i + 1])
                    run_end = -1
                elif run_end == i:
                    spans[-1][1] = i + 1
                    run_end = i + 1
                else:
                    spans.append([i, i + 1])
                    run_end = i + 1
            else:
                attach(i, i + 1)
    else:
        for m in _TOKEN.finditer(text):
            if any(_wordy(c) for c in m.group()):
                spans.append([m.start(), m.end()])
            else:
                attach(m.start(), m.end())
    if spans and lead is not None:
        spans[0][0] = lead
    return [{"text": text[a:b], "start": a, "end": b} for a, b in spans]


def unit_bytes(text: str, units: list[dict]) -> list[int]:
    out = []
    for k, u in enumerate(units):
        stop = units[k + 1]["start"] if k + 1 < len(units) else u["end"]
        out.append(len(text[u["start"] : stop].encode()))
    return out


def break_after(text: str, units: list[dict]) -> list[bool]:
    return [text[u["end"] - 1] in BREAK_PUNCT for u in units]


def span_text(text: str, units: list[dict], a: int, b: int) -> str:
    if a == b:
        return ""
    return text[units[a]["start"] : units[b - 1]["end"]]


def join_words(words: list[str], lang: str) -> str:
    return ("" if lang in UNSPACED else " ").join(words)


def ref_segments(words: list[dict], min_pause: float) -> list[tuple[int, int]]:
    """Half-open word ranges split at pauses >= ``min_pause``; zero-width words never start a segment."""
    if not words:
        return []
    starts = [0]
    prev_end: float | None = None
    for j, w in enumerate(words):
        if w["end"] - w["start"] < 0.001:
            continue
        if prev_end is not None and j > 0 and w["start"] - prev_end >= min_pause:
            starts.append(j)
        prev_end = w["end"]
    return [(a, b) for a, b in zip(starts, starts[1:] + [len(words)])]


def project_segments(
    ref_ranges: list[tuple[int, int]],
    ref_spans: list[tuple[float, float]],
    links,
    unit_bytes: list[int],
    break_after: list[bool],
) -> list[tuple[int, int]]:
    """One contiguous target unit range per reference segment (monotonic, covering ``[0, m)``, may be empty)."""
    n_seg, m = len(ref_ranges), len(unit_bytes)
    if n_seg == 0:
        return []
    if n_seg == 1:
        return [(0, m)]
    seg_of: dict[int, int] = {}
    for k, (a, b) in enumerate(ref_ranges):
        for i in range(a, b):
            seg_of[i] = k
    # cross[k, j]: links into target unit j whose reference word lies outside segment k
    per_unit = np.zeros((n_seg, m), dtype=np.float64)
    for i, j in links:
        if 0 <= j < m and i in seg_of:
            per_unit[:, j] += 1.0
            per_unit[seg_of[i], j] -= 1.0
    prefix = np.concatenate([np.zeros((n_seg, 1)), np.cumsum(per_unit, axis=1)], axis=1)
    cum_bytes = np.concatenate([[0.0], np.cumsum(np.asarray(unit_bytes, dtype=np.float64))])
    total_bytes = float(cum_bytes[-1])
    durations = [max(0.0, e - s) for s, e in ref_spans]
    total_dur = sum(durations)
    pen = np.zeros(m + 1)
    for b in range(1, m):
        if not break_after[b - 1]:
            pen[b] = 0.5

    idx = np.arange(m + 1)
    upper = idx[:, None] <= idx[None, :]  # a <= b
    inf = np.inf
    f = np.full(m + 1, inf)
    f[0] = 0.0
    back: list[np.ndarray] = []
    for k in range(n_seg):
        cost = prefix[k][None, :] - prefix[k][:, None]
        if total_bytes > 0 and total_dur > 0:
            share = (cum_bytes[None, :] - cum_bytes[:, None]) / total_bytes
            cost = cost + np.abs(share - durations[k] / total_dur)
        total = f[:, None] + cost
        total = np.where(upper, total, inf)
        arg = np.argmin(total, axis=0)  # first minimum: smallest a
        g = total[arg, idx]
        if k < n_seg - 1:
            g = g + pen
        back.append(arg)
        f = g
    bounds = [m]
    b = m
    for k in range(n_seg - 1, -1, -1):
        b = int(back[k][b])
        bounds.append(b)
    bounds.reverse()  # b_0 .. b_K
    return [(bounds[k], bounds[k + 1]) for k in range(n_seg)]


@dataclass(frozen=True)
class RateScale:
    """The speaking-rate buckets a dub language plans in: the condition row's buckets 0..n-1 and, where measured safe
    for the language, tempo-steered buckets below 0 / from n up (README "Extended speaking-rate buckets"). Bucket i,
    ``first`` <= i <= ``last``, spans [edges[i - first - 1], edges[i - first]) in UTF-8 bytes / s; ``doses`` holds the
    tempo dose of every steered bucket (the rate row unknown)."""

    edges: tuple[float, ...]
    first: int = 0
    doses: Mapping[int, float] = field(default_factory=dict)

    @classmethod
    def extended(cls, row_edges, below, above, doses: Mapping[int, float]) -> "RateScale":
        """The row's scale plus the steered buckets of ``doses`` (contiguous from the row): ``below[k]`` is the
        nominal lower edge of bucket -k, ``above[k]`` the upper edge of bucket n - 1 + k."""
        n = len(row_edges) + 1
        lo = min((b for b in doses if b < 0), default=0)
        hi = max((b for b in doses if b >= n), default=n - 1)
        if set(doses) != set(range(lo, 0)) | set(range(n, hi + 1)):
            raise ValueError(f"steered buckets {sorted(doses)} are not contiguous from the row's 0..{n - 1}")
        if -lo > len(below) or hi - n + 1 > len(above):
            raise ValueError(f"steered buckets {lo}..{hi} have no nominal edges")
        edges = (*reversed(below[:-lo]), *row_edges, *above[: hi - n + 1])
        return cls(tuple(float(e) for e in edges), lo, {int(b): float(d) for b, d in doses.items()})

    @property
    def last(self) -> int:
        return self.first + len(self.edges)

    def bucket(self, rate: float) -> int:
        """The bucket of a rate."""
        return self.first + bisect.bisect_right(self.edges, rate)

    def rate(self, bucket: int) -> float:
        """Nominal rate of a bucket: its midpoint; the open end buckets extend by half a neighbour (laci.RateBuckets)."""
        e, i = self.edges, bucket - self.first
        if i <= 0:
            return e[0] - (e[1] - e[0]) / 2
        if i >= len(e):
            return e[-1] + (e[-1] - e[-2]) / 2
        return (e[i - 1] + e[i]) / 2


def candidate_plan(bucket: int, scale: RateScale, seed: int, segment_index: int) -> list[dict]:
    out, used = [], set()
    for oi, offset in enumerate(CANDIDATE_BUCKET_OFFSETS):
        b = bucket + offset
        if not scale.first <= b <= scale.last or b in used:
            continue
        used.add(b)
        for r in range(CANDIDATE_SEEDS):
            out.append({"bucket": b, "offset": offset, "seed": seed + 100 * segment_index + 10 * oi + r})
    return out


def pinned_candidate_plan(bucket: int, seed: int, segment_index: int) -> list[dict]:
    """Candidates of a segment whose bucket the user set: the full count of an automatic plan, all in ``bucket``,
    told apart by seed only."""
    n = len(CANDIDATE_BUCKET_OFFSETS) * CANDIDATE_SEEDS
    return [{"bucket": bucket, "offset": 0, "seed": seed + 100 * segment_index + r} for r in range(n)]


def max_tokens_for(duration_s: float) -> int:
    return min(9000, math.ceil(FRAME_RATE * (2.0 * duration_s + 3.0)))


def voice_prompt_span(segments, duration: float) -> tuple[float, float]:
    start = max(0.0, segments[0].ref_start - 0.1)
    end = None
    for seg in segments:
        cand = seg.ref_end + 0.1
        if cand - start <= VOICE_PROMPT_MAX_S and (end is None or cand > end):
            end = cand
    if end is None:
        end = start + VOICE_PROMPT_MAX_S
    return min(start, duration), min(end, duration)


def speech_span(items: list[dict], audio: np.ndarray, sr: int) -> tuple[float, float] | None:
    spans = [(it["start"], it["end"]) for it in items if it["end"] > it["start"]]
    if spans:
        return min(s for s, _ in spans), max(e for _, e in spans)
    frame = max(1, int(0.02 * sr))
    n = len(audio) // frame
    if n == 0:
        return None
    rms = np.sqrt((audio[: n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(axis=1))
    peak = rms.max()
    if peak <= 0:
        return None
    above = np.nonzero(rms > 0.1 * peak)[0]
    return float(above[0] * frame / sr), float((above[-1] + 1) * frame / sr)


def candidate_score(c: dict | None, ref_dur: float, room: float) -> float | None:
    if c is None or c["speech_s"] <= 0:
        return None
    score = abs(math.log(c["speech_s"] / ref_dur))
    if c["speech_s"] > room:
        score += 0.5
    if c["capped"]:
        score += 1.0
    if c["end"] == "max_tokens":
        score += 1.0
    return score


def choose_candidate(cands: list[dict | None], ref_dur: float, room: float) -> int | None:
    best, best_score = None, None
    for i, c in enumerate(cands):
        s = candidate_score(c, ref_dur, room)
        if s is not None and (best_score is None or s < best_score):
            best, best_score = i, s
    return best


def place_segments(items: list[dict], ref_duration: float) -> tuple[list[dict], float]:
    placed, prev_end, timeline = [], -math.inf, ref_duration
    for it in items:
        s0, s1 = it["s0"], it["s1"]
        crop_start = max(0.0, s0 - PRE_PAD_S)
        crop_end = min(it["take_len"], s1 + POST_PAD_S)
        speech_start = max(it["ref_start"], prev_end + MIN_GAP_S)
        speech_end = speech_start + (s1 - s0)
        place_start = speech_start - (s0 - crop_start)
        if place_start < 0:
            crop_start += -place_start
            place_start = 0.0
        prev_end = speech_end
        timeline = max(timeline, place_start + crop_end - crop_start)
        placed.append(
            {
                "crop_start": crop_start,
                "crop_end": crop_end,
                "place_start": place_start,
                "speech_start": speech_start,
                "speech_end": speech_end,
                "shift": speech_start - it["ref_start"],
                "start_error": speech_start - it["ref_start"],
                "end_error": speech_end - it["ref_end"],
                "duration_error": (s1 - s0) - (it["ref_end"] - it["ref_start"]),
            }
        )
    return placed, timeline


def word_windows(counts: list[int]) -> list[tuple[int, int]]:
    n = len(counts)
    if any(c > WINDOW_MAX_WP for c in counts):
        raise ValueError(f"a word exceeds {WINDOW_MAX_WP} wordpieces")
    windows, s = [], 0
    while n:
        e, total = s, 0
        while e < n and total + counts[e] <= WINDOW_MAX_WP:
            total += counts[e]
            e += 1
        windows.append((s, e))
        if e == n:
            break
        s += max(1, int((e - s) * WINDOW_STRIDE_FRACTION))
    return windows


def window_of_words(windows: list[tuple[int, int]], n: int) -> list[int]:
    owner, best = [0] * n, [-1] * n
    for wi, (s, e) in enumerate(windows):
        for w in range(s, e):
            margin = min(w - s, e - 1 - w)
            if margin > best[w]:
                best[w], owner[w] = margin, wi
    return owner


def band_mask(w1: int, w2: int) -> np.ndarray:
    p = np.arange(w1, dtype=np.float64)[:, None]
    q = np.arange(w2, dtype=np.float64)[None, :]
    return np.abs(p * w2 / w1 - q) <= max(BAND_MIN_WP, BAND_FRACTION * w2)


@dataclass
class LipMap:
    """AV-sync's time warp of a dub: at output time ``out_t[i]`` the warped dub plays the dub's time ``src_t[i]``
    (seconds, both increasing). Outside ``[out_t[0], out_t[-1]]`` (the face track) the offset of the nearer end holds,
    so the warp is continuous over the whole dub."""

    out_t: np.ndarray
    src_t: np.ndarray

    @classmethod
    def from_frames(cls, start: float, rate: float, src_frames) -> "LipMap":
        """AV-sync's warp: the source frame of each output frame (``rate`` frames / s) of a span starting at ``start``.
        The source times are made strictly increasing (the inverse needs it)."""
        src = np.maximum.accumulate(np.asarray(src_frames, dtype=np.float64)) / rate
        out = np.arange(len(src), dtype=np.float64) / rate
        src = src + np.arange(len(src)) * 1e-9
        return cls(start + out, start + src)

    @staticmethod
    def _map(t, xp, fp):
        t = np.asarray(t, dtype=np.float64)
        inside = np.interp(t, xp, fp)
        return np.where(t < xp[0], t + (fp[0] - xp[0]), np.where(t > xp[-1], t + (fp[-1] - xp[-1]), inside))

    def src_time(self, t):
        """Dub time played at output time ``t``."""
        return self._map(t, self.out_t, self.src_t)

    def out_time(self, t):
        """Output time at which the dub's time ``t`` plays."""
        return self._map(t, self.src_t, self.out_t)

    def shifts(self) -> np.ndarray:
        """How far the warp moves the dub over the face track (seconds, + = later)."""
        return self.out_t - self.src_t


# Local time scale of the lip-sync warp is measured over this span (about a phone); it is capped at LIP_SCALE_MAX
# either way (a warp that stands the dub still, or skips over it, reads as the cap).
LIP_SCALE_WINDOW_S = 0.05
LIP_SCALE_MAX = 10.0

# Bounds of the lip-sync warp the server applies (guard_lip_frames): local playback rate and shift.
LIP_GUARD_MAX_RATE = 1.5
LIP_GUARD_MAX_SHIFT_S = 0.4
# The warp is kept only when it lowers the dub's mean SyncNet distance to the lips by more than this.
LIP_MIN_SYNC_GAIN = 0.0


def guard_lip_frames(src_frames, rate: float, max_rate: float, max_shift_s: float) -> np.ndarray:
    """Bound AV-sync's warp (``src_frames``: the dub frame, from the face track's start, each output frame of the track
    plays; ``rate`` frames / s) so it cannot wreck the dub: every shift within ``max_shift_s``, no advance at the track's
    first frame (before the track the warp holds that offset, so an advance there would play the dub's first words
    before t = 0 and drop them), and the local playback rate within [1 / max_rate, max_rate]. A cross-language dub's
    mouth shapes do not match the video's, and the unbounded DP path then stands the dub still (10x stretch) or skips
    whole words. Shift bounds and rate bounds alternate; the last pass enforces the rate, the shift bound holds to
    within the slack the rate bound needs."""
    src = np.maximum.accumulate(np.asarray(src_frames, dtype=np.float64))
    idx = np.arange(len(src), dtype=np.float64)
    s = max_shift_s * rate
    lo, hi = idx - s, idx + s
    hi[0] = min(hi[0], 0.0)
    lo_step, hi_step = 1.0 / max_rate, float(max_rate)
    for _ in range(4):
        src = np.clip(src, lo, hi)
        for i in range(1, len(src)):
            src[i] = min(max(src[i], src[i - 1] + lo_step), src[i - 1] + hi_step)
    return src


def lip_warp(m: LipMap, src_start: float, src_end: float) -> dict:
    """How much the lip-sync warp stretches and squeezes the dub's ``[src_start, src_end]`` (unwarped seconds).

    The local time scale s(t) = d out / d src over LIP_SCALE_WINDOW_S around output time t is > 1 where the dub is
    stretched (played slower, longer) and < 1 where it is squeezed (played faster). Returns factors, 1 = untouched:
    ``lip_scale`` (warped / unwarped length of the span), ``lip_stretch`` (largest s, >= 1), ``lip_squeeze``
    (largest 1 / s, >= 1: 1.25 plays 25 % faster) and ``lip_warp`` (geometric mean of max(s, 1 / s)).
    """
    a, b = float(m.out_time(src_start)), float(m.out_time(src_end))
    if src_end - src_start < 1e-6 or b - a < 1e-6:
        return {"lip_scale": 1.0, "lip_stretch": 1.0, "lip_squeeze": 1.0, "lip_warp": 1.0}
    t = np.linspace(a, b, max(2, int((b - a) / 0.01) + 1))
    h = LIP_SCALE_WINDOW_S / 2
    scale = 2 * h / np.maximum(m.src_time(t + h) - m.src_time(t - h), 1e-9)
    log_scale = np.log(np.clip(scale, 1 / LIP_SCALE_MAX, LIP_SCALE_MAX))
    return {
        "lip_scale": (b - a) / (src_end - src_start),
        "lip_stretch": float(np.exp(max(0.0, log_scale.max()))),
        "lip_squeeze": float(np.exp(max(0.0, -log_scale.min()))),
        "lip_warp": float(np.exp(np.abs(log_scale).mean())),
    }


# WSOLA (waveform-similarity overlap-add, Verhelst & Roelands 1993) at 44.1 kHz: 23 ms Hann frames at 50 % overlap,
# each taken from within ±11.6 ms (one pitch period down to 86 Hz) of its nominal source position.
WSOLA_WINDOW = 1024
WSOLA_TOLERANCE = 512
# Correlation given up per tolerance of distance from the nominal position: of near-equal matches (a periodic voiced
# frame matches one pitch period apart too) the nearest wins, so the output does not drift from the map.
WSOLA_DRIFT_PENALTY = 0.05


def wsola(x: np.ndarray, src_of, n_out: int, window: int = WSOLA_WINDOW, tolerance: int = WSOLA_TOLERANCE) -> np.ndarray:
    """Time-warp ``x`` (1-D) without changing its pitch: output sample ``t`` plays about source sample ``src_of(t)``.

    ``src_of`` maps an array of output sample positions to source sample positions (any increasing map). Output frame
    k is centred at ``k * hop`` (hop = window / 2); its source frame is the one within ``tolerance`` of
    ``src_of(k * hop)`` that best continues the previous source frame (normalized cross-correlation with that frame's
    natural continuation, less WSOLA_DRIFT_PENALTY per tolerance of distance), so the overlap-add stays phase-coherent.
    A map of slope 1 reproduces the shifted input.
    """
    x = np.asarray(x, dtype=np.float64)
    hop, half = window // 2, window // 2
    win = np.hanning(window + 1)[:window]  # periodic Hann: copies at 50 % overlap sum to 1
    pad = half + tolerance + hop
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = n_out // hop + 2
    nominal = np.rint(np.clip(np.asarray(src_of(np.arange(n_frames) * hop), dtype=np.float64), 0, len(x))).astype(int) + pad
    size = 1 << int(math.ceil(math.log2(2 * window + 2 * tolerance)))
    y = np.zeros(n_frames * hop + window)
    penalty = WSOLA_DRIFT_PENALTY * np.abs(np.arange(2 * tolerance + 1) - tolerance) / max(tolerance, 1)
    prev = None
    for k, c in enumerate(nominal):
        if prev is not None:
            natural = xp[prev + hop - half : prev + hop + half]
            region = xp[c - tolerance - half : c + tolerance + half]
            corr = np.fft.irfft(np.fft.rfft(region, size) * np.conj(np.fft.rfft(natural, size)), size)[: 2 * tolerance + 1]
            energy = np.cumsum(np.concatenate([[0.0], region * region]))
            energy = energy[window:] - energy[:-window]
            score = corr / np.sqrt(energy * float(natural @ natural) + 1e-12)
            c = c - tolerance + int(np.argmax(score - penalty))
        y[k * hop : k * hop + window] += win * xp[c - half : c + half]
        prev = c
    return y[half : half + n_out]
