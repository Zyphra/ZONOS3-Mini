"""Parity of src/dub-plan.cpp (via the dub-plan-replay binary) with model/src/zonos3/dub/plan.py, and of
src/align-text.cpp with zonos3.tts.align_text.

Every case runs the same call through ``dub-plan-replay`` ({"fn", "args"} JSON on stdin) and through ``zonos3.dub.plan``
(``align_tokens``: ``zonos3.tts.align_text``); results must be equal: exactly for ints / bools / strings, floats within
1e-9 relative, wsola within 1e-5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from zonos3.dub import plan as P
from zonos3.tts.align_text import align_tokens

EDGES = [10.117, 12.123, 13.286, 14.25, 15.14, 16.079, 17.094, 18.489, 21.208]
BELOW = [9.2, 8.4, 7.7]
ABOVE = [23.3, 25.5]
SR = 44100


# ---------------------------------------------------------------------------------------------------- python side


def _jsonable(x):
    if isinstance(x, P.RateScale):
        return {"edges": list(x.edges), "first": x.first, "last": x.last, "doses": [[b, d] for b, d in sorted(x.doses.items())]}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


def _num(v):
    return float(v) if isinstance(v, str) else v


def _scale(j):
    return P.RateScale(tuple(j["edges"]), j.get("first", 0), {int(b): float(d) for b, d in j.get("doses", [])})


def _lipmap(j):
    return P.LipMap(out_t=np.asarray(j["out_t"], dtype=np.float64), src_t=np.asarray(j["src_t"], dtype=np.float64))


def _src_of(j):
    if "affine" in j:
        scale, offset = j["affine"]
        return lambda t: t * scale + offset
    if "interp" in j:
        xp, fp, sr = j["interp"]["xp"], j["interp"]["fp"], j["interp"]["sr"]
        return lambda t: np.interp(t / sr, xp, fp) * sr
    m, sr = _lipmap(j["lipmap"]), j["lipmap"]["sr"]
    return lambda t: m.src_time(t / sr) * sr


_CONSTANTS = [
    "REF_LANGUAGES", "TARGET_LANGUAGES", "UNSPACED", "MAX_REF_SECONDS", "MIN_REF_SECONDS", "MAX_VIDEO_BYTES",
    "DEFAULT_MIN_PAUSE_S", "MIN_PAUSE_RANGE", "MAX_SEGMENTS", "CANDIDATE_BUCKET_OFFSETS", "CANDIDATE_SEEDS",
    "PRE_PAD_S", "POST_PAD_S", "MIN_GAP_S", "FADE_S", "VOICE_PROMPT_MAX_S", "FRAME_RATE", "SAMPLE_RATE", "BREAK_PUNCT",
    "WINDOW_MAX_WP", "WINDOW_STRIDE_FRACTION", "BAND_MIN_WP", "BAND_FRACTION", "LIP_SCALE_WINDOW_S", "LIP_SCALE_MAX",
    "LIP_GUARD_MAX_RATE", "LIP_GUARD_MAX_SHIFT_S", "LIP_MIN_SYNC_GAIN", "WSOLA_WINDOW", "WSOLA_TOLERANCE",
    "WSOLA_DRIFT_PENALTY",
]


def py_call(fn: str, a: dict):
    if fn == "align_tokens":
        return align_tokens(**a)
    if fn == "constants":
        out = {k: _jsonable(getattr(P, k)) for k in _CONSTANTS}
        out["UNSPACED"] = sorted(P.UNSPACED)
        return out
    if fn in ("text_units", "join_words", "word_windows", "window_of_words", "band_mask", "max_tokens_for",
              "pinned_candidate_plan"):
        return _jsonable(getattr(P, fn)(**a))
    if fn in ("unit_bytes", "break_after", "span_text", "ref_segments"):
        return _jsonable(getattr(P, fn)(**a))
    if fn == "project_segments":
        return _jsonable(P.project_segments(
            [tuple(r) for r in a["ref_ranges"]], [tuple(s) for s in a["ref_spans"]], [tuple(x) for x in a["links"]],
            a["unit_bytes"], a["break_after"]))
    if fn == "RateScale":
        return _jsonable(_scale(a))
    if fn == "RateScale.extended":
        return _jsonable(P.RateScale.extended(a["row_edges"], a["below"], a["above"], {int(b): d for b, d in a["doses"]}))
    if fn == "RateScale.bucket":
        return _scale(a["scale"]).bucket(a["rate"])
    if fn == "RateScale.rate":
        return _scale(a["scale"]).rate(a["bucket"])
    if fn == "candidate_plan":
        return P.candidate_plan(a["bucket"], _scale(a["scale"]), a["seed"], a["segment_index"])
    if fn == "voice_prompt_span":
        segs = [P.RefSegment(s["ref_start"], s["ref_end"]) for s in a["segments"]]
        return _jsonable(P.voice_prompt_span(segs, a["duration"]))
    if fn == "speech_span":
        return _jsonable(P.speech_span(a["items"], np.asarray(a["audio"], dtype=np.float32), a["sr"]))
    if fn == "candidate_score":
        return P.candidate_score(a["c"], a["ref_dur"], _num(a["room"]))
    if fn == "choose_candidate":
        return P.choose_candidate(a["cands"], a["ref_dur"], _num(a["room"]))
    if fn == "place_segments":
        return _jsonable(P.place_segments(a["items"], a["ref_duration"]))
    if fn == "interp":
        return _jsonable(np.interp(np.asarray(a["x"], dtype=np.float64), a["xp"], a["fp"]))
    if fn == "LipMap.from_frames":
        m = P.LipMap.from_frames(a["start"], a["rate"], a["src_frames"])
        return {"out_t": m.out_t.tolist(), "src_t": m.src_t.tolist()}
    if fn == "LipMap.src_time":
        return _jsonable(_lipmap(a["map"]).src_time(a["t"]))
    if fn == "LipMap.out_time":
        return _jsonable(_lipmap(a["map"]).out_time(a["t"]))
    if fn == "LipMap.shifts":
        return _jsonable(_lipmap(a["map"]).shifts())
    if fn == "guard_lip_frames":
        return _jsonable(P.guard_lip_frames(a["src_frames"], a["rate"], a["max_rate"], a["max_shift_s"]))
    if fn == "lip_warp":
        return _jsonable(P.lip_warp(_lipmap(a["map"]), a["src_start"], a["src_end"]))
    if fn == "wsola":
        kw = {k: a[k] for k in ("window", "tolerance") if k in a}
        return _jsonable(P.wsola(np.asarray(a["x"], dtype=np.float64), _src_of(a["src_of"]), a["n_out"], **kw))
    raise AssertionError(fn)


def py_result(fn, a):
    try:
        return py_call(fn, a)
    except ValueError as e:
        return {"error": "ValueError", "message": str(e)}
    except IndexError:
        return {"error": "IndexError"}


def assert_same(got, want, tol, path="$"):
    if isinstance(want, bool) or want is None or isinstance(want, str):
        assert got == want and type(got) is type(want), f"{path}: {got!r} != {want!r}"
    elif isinstance(want, (int, float)):
        assert isinstance(got, (int, float)) and not isinstance(got, bool), f"{path}: {got!r} != {want!r}"
        if isinstance(want, int) and isinstance(got, int):
            assert got == want, f"{path}: {got} != {want}"
        elif tol is None:
            assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12), f"{path}: {got!r} != {want!r}"
        else:
            assert abs(got - want) <= tol, f"{path}: {got!r} != {want!r}"
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            assert_same(g, w, tol, f"{path}[{i}]")
    elif isinstance(want, dict):
        if want.get("error") == "IndexError":
            assert isinstance(got, dict) and got.get("error") == "IndexError", f"{path}: {got!r} is not an IndexError"
            return
        assert isinstance(got, dict) and set(got) == set(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for k in want:
            assert_same(got[k], want[k], tol, f"{path}.{k}")
    else:
        raise AssertionError(f"{path}: unexpected {type(want)}")


# ---------------------------------------------------------------------------------------------------- cases

TEXTS = [
    ("Les requins — existent, bien.", "fr"),
    ("— Bonjour monde", "fr"),
    ("鲨鱼确实存在，但很少攻击人类。", "zh"),
    ("我用iPhone15手机", "zh"),
    ("", "en"),
    ("   \t\n ", "en"),
    ("— —", "fr"),
    ("... ¿Qué? ¡Sí! fin", "es"),
    ("l'homme d'affaires est là.", "fr"),
    ("こんにちは、世界！ABC def。", "ja"),
    ("「東京」へ行く", "ja"),
    ("a\x1cb\x1fc d", "en"),
    ("hi 👋 there 😀!", "en"),
    ("123 456, 7.", "de"),
    ("안녕하세요 세계. 반갑습니다!", "ko"),
    ("  leading and trailing  ", "en"),
    ("你好 world，テスト ok", "yue"),
    ("!!! ???", "zh"),
    ("x", "en"),
    ("Straße über Œuvre… done", "de"),
]


def text_cases():
    cases = []
    for text, lang in TEXTS:
        for lg in (lang, "zh", "en"):
            cases.append(("text_units", {"text": text, "lang": lg}))
            cases.append(("align_tokens", {"text": text, "lang": lg}))
            units = P.text_units(text, lg)
            cases.append(("unit_bytes", {"text": text, "units": units}))
            if units:
                cases.append(("break_after", {"text": text, "units": units}))
            n = len(units)
            for a, b in {(0, n), (0, 0), (n // 2, n), (0, max(0, n - 1)), (min(1, n), min(1, n))}:
                if a <= b:
                    cases.append(("span_text", {"text": text, "units": units, "a": a, "b": b}))
    for words, lang in ((["a", "b", "c"], "en"), (["鲨", "鱼"], "zh"), ([], "fr"), (["x"], "ja"), (["Hi", "you"], "yue")):
        cases.append(("join_words", {"words": words, "lang": lang}))
    return cases


def segment_cases(rng):
    w = [{"start": 0.0, "end": 0.3}, {"start": 0.35, "end": 0.6}, {"start": 1.2, "end": 1.2},
         {"start": 1.25, "end": 1.5}, {"start": 1.6, "end": 1.9}]
    cases = [("ref_segments", {"words": w, "min_pause": p}) for p in (0.3, 5.0, 0.05, 0.65)]
    cases.append(("ref_segments", {"words": [{"start": 0.0, "end": 0.3}, {"start": 2.0, "end": 2.0},
                                             {"start": 2.01, "end": 2.4}], "min_pause": 0.3}))
    cases.append(("ref_segments", {"words": [], "min_pause": 0.3}))
    cases.append(("ref_segments", {"words": [{"start": 1.0, "end": 1.0}, {"start": 3.0, "end": 3.0005}], "min_pause": 0.1}))
    for _ in range(10):
        t, words = 0.0, []
        for _ in range(int(rng.integers(1, 40))):
            t += float(rng.choice([0.0, 0.05, 0.3, 0.6, 1.2]))
            d = float(rng.choice([0.0, 0.0005, 0.2, 0.4]))
            words.append({"start": t, "end": t + d})
            t += d
        cases.append(("ref_segments", {"words": words, "min_pause": float(rng.choice([0.1, 0.3, 0.5]))}))

    cases += [
        ("project_segments", {"ref_ranges": [[0, 3], [3, 6]], "ref_spans": [[0, 1], [1.5, 2.5]],
                              "links": [[0, 0], [1, 2], [2, 1], [3, 3], [4, 4], [5, 5]], "unit_bytes": [5] * 6,
                              "break_after": [False, False, True, False, False, True]}),
        ("project_segments", {"ref_ranges": [[0, 2], [2, 4]], "ref_spans": [[0, 1], [1, 2]],
                              "links": [[0, 0], [1, 1], [2, 3], [3, 4]], "unit_bytes": [1] * 5,
                              "break_after": [False, True, False, False, True]}),
        ("project_segments", {"ref_ranges": [[0, 2], [2, 4]], "ref_spans": [[0, 1], [1, 2]],
                              "links": [[0, 0], [1, 1], [2, 3], [3, 4]], "unit_bytes": [1] * 5,
                              "break_after": [False, False, True, False, True]}),
        ("project_segments", {"ref_ranges": [[0, 2], [2, 4], [4, 6]], "ref_spans": [[0, 1], [1.5, 1.6], [2, 3]],
                              "links": [[0, 0], [1, 1], [4, 2], [5, 3]], "unit_bytes": [1] * 4, "break_after": [True] * 4}),
        ("project_segments", {"ref_ranges": [[0, 1], [1, 2]], "ref_spans": [[0, 1], [1, 4]], "links": [],
                              "unit_bytes": [1] * 8, "break_after": [True] * 8}),
        ("project_segments", {"ref_ranges": [], "ref_spans": [], "links": [], "unit_bytes": [1, 2], "break_after": [True] * 2}),
        ("project_segments", {"ref_ranges": [[0, 3]], "ref_spans": [[0, 1]], "links": [], "unit_bytes": [1, 2],
                              "break_after": [True] * 2}),
        # zero bytes / zero durations: link costs only, ties broken towards the smallest start
        ("project_segments", {"ref_ranges": [[0, 1], [1, 2], [2, 3]], "ref_spans": [[0, 0], [1, 1], [2, 2]],
                              "links": [[0, 0], [2, 3], [9, 1], [1, -1], [1, 7]], "unit_bytes": [0] * 4,
                              "break_after": [False] * 4}),
        ("project_segments", {"ref_ranges": [[0, 2], [2, 3]], "ref_spans": [[0, 1], [1, 2]], "links": [],
                              "unit_bytes": [], "break_after": []}),
    ]
    for _ in range(25):
        n_words = int(rng.integers(2, 30))
        cuts = sorted(set(int(c) for c in rng.integers(1, n_words, size=int(rng.integers(1, 6)))))
        bounds = [0, *cuts, n_words]
        ranges = [[a, b] for a, b in zip(bounds, bounds[1:])]
        t, spans = 0.0, []
        for _ in ranges:
            t += float(rng.uniform(0, 1))
            d = float(rng.uniform(0, 3))
            spans.append([t, t + d])
            t += d
        m = int(rng.integers(1, 40))
        links = [[int(rng.integers(0, n_words + 2)), int(rng.integers(-1, m + 1))] for _ in range(int(rng.integers(0, 3 * m)))]
        cases.append(("project_segments", {"ref_ranges": ranges, "ref_spans": spans, "links": links,
                                           "unit_bytes": [int(b) for b in rng.integers(1, 12, size=m)],
                                           "break_after": [bool(b) for b in rng.integers(0, 2, size=m)]}))
    return cases


def rate_cases():
    row = {"edges": EDGES}
    ext = P.RateScale.extended(EDGES, BELOW, ABOVE, {-2: -1.0, -1: -0.9, 10: 1.4})
    ext_j = _jsonable(ext)
    cases = [("RateScale", row), ("RateScale", ext_j)]
    for doses in ([[-2, -1.0], [-1, -0.9], [10, 1.4]], [[10, 1.4]], [], [[-1, -0.9]], [[-3, -1.2], [-2, -1], [-1, -0.9]],
                  [[-2, -1.0]], [[10, 1.0], [11, 1.5], [12, 2.0]], [[-4, 1], [-3, 1], [-2, 1], [-1, 1]], [[11, 1.0]],
                  [[0, 1.0]], [[10, 1.0], [11, 1.5]]):
        cases.append(("RateScale.extended", {"row_edges": EDGES, "below": BELOW, "above": ABOVE, "doses": doses}))
    rates = [0.0, 5.0, 8.4, 9.0, 9.2, 9.5, 10.117, 14.25, 14.2500001, 21.0, 21.208, 22.0, 23.3, 40.0, -3.0]
    for scale in (row, ext_j):
        for r in rates:
            cases.append(("RateScale.bucket", {"scale": scale, "rate": r}))
        for b in range(-4, 14):
            cases.append(("RateScale.rate", {"scale": scale, "bucket": b}))
            cases.append(("candidate_plan", {"bucket": b, "scale": scale, "seed": 7, "segment_index": 3}))
    cases.append(("RateScale.rate", {"scale": {"edges": [12.0]}, "bucket": 0}))
    one = _jsonable(P.RateScale.extended(EDGES, BELOW, ABOVE, {-1: -0.9}))
    cases.append(("candidate_plan", {"bucket": 0, "scale": one, "seed": 7, "segment_index": 3}))
    cases.append(("candidate_plan", {"bucket": 5, "scale": row, "seed": 0, "segment_index": 0}))
    cases.append(("candidate_plan", {"bucket": 9, "scale": row, "seed": 2**31 - 1, "segment_index": 119}))
    for b, seed, k in ((3, 7, 2), (0, 0, 0), (-2, 123456789, 50)):
        cases.append(("pinned_candidate_plan", {"bucket": b, "seed": seed, "segment_index": k}))
    for d in (0.0, 1.234, 2.0, 3.3333333, 298.5, 299.0, 1000.0, -1.0, 0.1):
        cases.append(("max_tokens_for", {"duration_s": d}))
    return cases


def take(speech_s, capped=False, end="eos"):
    return {"speech_s": speech_s, "capped": capped, "end": end}


def choice_cases(rng):
    inf = "inf"
    cases = [
        ("choose_candidate", {"cands": [take(3.0), take(2.1), take(1.0)], "ref_dur": 2.0, "room": inf}),
        ("choose_candidate", {"cands": [take(2.4), take(1.9)], "ref_dur": 2.0, "room": inf}),
        ("choose_candidate", {"cands": [take(2.05), take(1.7)], "ref_dur": 2.0, "room": 2.0}),
        ("choose_candidate", {"cands": [take(2.0, capped=True), take(2.3)], "ref_dur": 2.0, "room": inf}),
        ("choose_candidate", {"cands": [take(2.0, end="max_tokens"), take(2.3)], "ref_dur": 2.0, "room": inf}),
        ("choose_candidate", {"cands": [None, take(0.0), None], "ref_dur": 2.0, "room": inf}),
        ("choose_candidate", {"cands": [None, take(1.0)], "ref_dur": 2.0, "room": inf}),
        ("choose_candidate", {"cands": [take(1.0), take(4.0)], "ref_dur": 2.0, "room": inf}),  # exact tie: first
        ("choose_candidate", {"cands": [], "ref_dur": 2.0, "room": inf}),
        ("candidate_score", {"c": None, "ref_dur": 2.0, "room": inf}),
        ("candidate_score", {"c": take(-1.0), "ref_dur": 2.0, "room": inf}),
        ("candidate_score", {"c": take(2.5, True, "max_tokens"), "ref_dur": 2.0, "room": 2.2}),
    ]
    for _ in range(10):
        cands = [None if rng.random() < 0.2 else take(float(rng.uniform(0, 5)), bool(rng.random() < 0.3),
                                                      str(rng.choice(["eos", "max_tokens", "laci"])))
                 for _ in range(int(rng.integers(1, 10)))]
        cases.append(("choose_candidate", {"cands": cands, "ref_dur": float(rng.uniform(0.5, 4)),
                                           "room": float(rng.uniform(0.5, 5))}))
        for c in cands:
            cases.append(("candidate_score", {"c": c, "ref_dur": 1.7, "room": 2.5}))
    return cases


def placement_cases(rng):
    items = [{"ref_start": 0.0, "ref_end": 1.0, "take_len": 4.0, "s0": 0.3, "s1": 2.3},
             {"ref_start": 1.5, "ref_end": 2.5, "take_len": 3.0, "s0": 0.2, "s1": 1.2}]
    cases = [
        ("place_segments", {"items": items, "ref_duration": 3.0}),
        ("place_segments", {"items": items[1:], "ref_duration": 3.0}),
        ("place_segments", {"items": [{"ref_start": 0.0, "ref_end": 1.0, "take_len": 2.0, "s0": 0.3, "s1": 1.0}],
                            "ref_duration": 1.0}),
        ("place_segments", {"items": [{"ref_start": 0.5, "ref_end": 1.0, "take_len": 5.0, "s0": 0.0, "s1": 4.0}],
                            "ref_duration": 1.0}),
        ("place_segments", {"items": [], "ref_duration": 2.0}),
    ]
    for _ in range(10):
        t, its = float(rng.uniform(0, 1)), []
        for _ in range(int(rng.integers(1, 12))):
            d = float(rng.uniform(0.2, 3))
            take_len = float(rng.uniform(0.5, 5))
            s0 = float(rng.uniform(0, take_len / 2))
            its.append({"ref_start": t, "ref_end": t + d, "take_len": take_len, "s0": s0,
                        "s1": float(rng.uniform(s0, take_len))})
            t += d + float(rng.uniform(0, 1))
        cases.append(("place_segments", {"items": its, "ref_duration": t}))

    segs = [{"ref_start": 1.0, "ref_end": 5.0}, {"ref_start": 6.0, "ref_end": 15.0}, {"ref_start": 16.0, "ref_end": 30.0}]
    cases += [
        ("voice_prompt_span", {"segments": segs, "duration": 60.0}),
        ("voice_prompt_span", {"segments": [{"ref_start": 0.0, "ref_end": 30.0}], "duration": 60.0}),
        ("voice_prompt_span", {"segments": [{"ref_start": 0.0, "ref_end": 30.0}], "duration": 12.0}),
        ("voice_prompt_span", {"segments": [{"ref_start": 0.05, "ref_end": 19.85}], "duration": 40.0}),
        ("voice_prompt_span", {"segments": [{"ref_start": 50.0, "ref_end": 55.0}], "duration": 30.0}),
        ("voice_prompt_span", {"segments": [], "duration": 30.0}),
    ]
    for _ in range(8):
        t, ss = float(rng.uniform(0, 3)), []
        for _ in range(int(rng.integers(1, 8))):
            d = float(rng.uniform(0.5, 9))
            ss.append({"ref_start": t, "ref_end": t + d})
            t += d + float(rng.uniform(0, 2))
        cases.append(("voice_prompt_span", {"segments": ss, "duration": float(rng.uniform(5, t + 5))}))
    return cases


def speech_cases(rng):
    zeros = np.zeros(SR, dtype=np.float32)
    burst = zeros.copy()
    burst[int(0.3 * SR) : int(0.6 * SR)] = 0.5
    noise = (rng.standard_normal(int(0.7 * SR)) * np.linspace(0, 1, int(0.7 * SR)) ** 3).astype(np.float32)
    items = [{"start": 0.2, "end": 0.3}, {"start": 0.5, "end": 0.7}, {"start": 0.9, "end": 0.9}]
    return [
        ("speech_span", {"items": items, "audio": zeros.tolist(), "sr": SR}),
        ("speech_span", {"items": [], "audio": burst.tolist(), "sr": SR}),
        ("speech_span", {"items": [], "audio": zeros.tolist(), "sr": SR}),
        ("speech_span", {"items": [{"start": 0.4, "end": 0.4}], "audio": noise.tolist(), "sr": SR}),
        ("speech_span", {"items": [{"start": 0.2, "end": 0.3, "placed": True}, {"start": 0.1, "end": 0.9, "placed": False}],
                         "audio": zeros.tolist(), "sr": SR}),
        ("speech_span", {"items": [{"start": 0.1, "end": 0.9, "placed": False}], "audio": burst.tolist(), "sr": SR}),
        ("speech_span", {"items": [], "audio": noise.tolist(), "sr": 16000}),
        ("speech_span", {"items": [], "audio": noise[:500].tolist(), "sr": SR}),
        ("speech_span", {"items": [], "audio": noise[:40].tolist(), "sr": 10}),
    ]


def window_cases(rng):
    cases = [
        ("word_windows", {"counts": [3] * 400}),
        ("word_windows", {"counts": [2, 600]}),
        ("word_windows", {"counts": []}),
        ("word_windows", {"counts": [0, 0]}),
        ("word_windows", {"counts": [510, 510, 1]}),
        ("word_windows", {"counts": [255, 256, 1, 509]}),
        ("window_of_words", {"windows": [[0, 10], [7, 17]], "n": 17}),
        ("window_of_words", {"windows": [], "n": 4}),
        ("window_of_words", {"windows": [[0, 3]], "n": 2}),
    ]
    for _ in range(8):
        counts = [int(c) for c in rng.integers(0, 60, size=int(rng.integers(1, 200)))]
        cases.append(("word_windows", {"counts": counts}))
        wins = P.word_windows(counts)
        cases.append(("window_of_words", {"windows": [list(w) for w in wins], "n": len(counts)}))
    for w1, w2 in ((30, 40), (1000, 1000), (7, 300), (300, 7), (0, 5), (1, 1), (213, 77)):
        cases.append(("band_mask", {"w1": w1, "w2": w2}))
    return cases


def lip_cases(rng):
    rate = 62.5
    n = 250
    src = np.linspace(-0.1 * rate, n - 1 + 0.2 * rate, n)
    m = P.LipMap.from_frames(2.0, rate, src)
    mj = {"out_t": m.out_t.tolist(), "src_t": m.src_t.tolist()}
    plateau = P.LipMap.from_frames(0.0, 62.5, [0, 1, 3, 2, 2, 5])
    pj = {"out_t": plateau.out_t.tolist(), "src_t": plateau.src_t.tolist()}
    hand = {"out_t": [1.0, 2.25, 3.5], "src_t": [1.0, 2.0, 3.5625]}
    still = P.LipMap.from_frames(0.0, 62.5, [0] * 40 + list(range(1, 100)))
    sj = {"out_t": still.out_t.tolist(), "src_t": still.src_t.tolist()}
    ts = [0.0, 1.0, 2.0, 2.5, 4.0, 5.9, 8.0, 5.984, -1.0]
    cases = [
        ("LipMap.from_frames", {"start": 2.0, "rate": rate, "src_frames": src.tolist()}),
        ("LipMap.from_frames", {"start": 0.0, "rate": 62.5, "src_frames": [0, 1, 3, 2, 2, 5]}),
        ("LipMap.from_frames", {"start": 1.5, "rate": 25.0, "src_frames": []}),
        ("LipMap.shifts", {"map": mj}),
        ("LipMap.shifts", {"map": pj}),
    ]
    for mm in (mj, pj, hand, sj):
        cases.append(("LipMap.src_time", {"map": mm, "t": ts}))
        cases.append(("LipMap.out_time", {"map": mm, "t": ts}))
        cases.append(("LipMap.src_time", {"map": mm, "t": mm["out_t"]}))
        cases.append(("LipMap.out_time", {"map": mm, "t": mm["src_t"]}))
    cases.append(("LipMap.src_time", {"map": hand, "t": 1.7}))
    for a, b in ((1.2, 1.8), (2.3, 3.4), (1.2, 3.4), (4.0, 5.0), (1.0, 1.0000001), (3.0, 2.0)):
        cases.append(("lip_warp", {"map": hand, "src_start": a, "src_end": b}))
    cases.append(("lip_warp", {"map": sj, "src_start": 0.0, "src_end": 1.5}))
    cases.append(("lip_warp", {"map": mj, "src_start": 1.9, "src_end": 6.3}))
    cases.append(("lip_warp", {"map": mj, "src_start": float(m.src_t[0]), "src_end": float(m.src_t[-1])}))

    n = 400
    gsrc = np.concatenate([np.full(1, 100.0), np.linspace(100, 160, 150), np.full(62, 160.0), np.linspace(160, 400, n - 213)])
    cases.append(("guard_lip_frames", {"src_frames": gsrc.tolist(), "rate": rate, "max_rate": 1.5, "max_shift_s": 0.4}))
    cases.append(("guard_lip_frames", {"src_frames": [5.0], "rate": rate, "max_rate": 1.5, "max_shift_s": 0.4}))
    cases.append(("guard_lip_frames", {"src_frames": [], "rate": rate, "max_rate": 1.5, "max_shift_s": 0.4}))
    for _ in range(6):
        k = int(rng.integers(2, 300))
        walk = np.cumsum(rng.choice([0.0, 0.5, 1.0, 1.0, 2.0, 3.0], size=k)) + float(rng.uniform(-20, 20))
        guarded = P.guard_lip_frames(walk, rate, 1.5, 0.4)
        cases.append(("guard_lip_frames", {"src_frames": walk.tolist(), "rate": rate,
                                           "max_rate": float(rng.uniform(1.1, 3)), "max_shift_s": float(rng.uniform(0.1, 1))}))
        gm = P.LipMap.from_frames(float(rng.uniform(0, 3)), rate, guarded)
        gj = {"out_t": gm.out_t.tolist(), "src_t": gm.src_t.tolist()}
        cases.append(("lip_warp", {"map": gj, "src_start": float(gm.src_t[0]), "src_end": float(gm.src_t[-1])}))
        cases.append(("lip_warp", {"map": gj, "src_start": float(gm.src_t[0]) - 0.3,
                                   "src_end": float(gm.src_t[len(gm.src_t) // 2])}))

    xp = np.sort(rng.uniform(-2, 5, size=12))
    cases.append(("interp", {"x": [float(v) for v in rng.uniform(-3, 6, size=50)] + xp.tolist(), "xp": xp.tolist(),
                             "fp": rng.standard_normal(12).tolist()}))
    cases.append(("interp", {"x": [0.0, 1.0, 2.0], "xp": [1.0], "fp": [3.0]}))
    return cases


def harmonic(seconds, f0=150.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6)) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t))


def wsola_cases(rng):
    x1 = harmonic(1.0).tolist()
    x2 = harmonic(2.0)
    cases = [
        ("wsola", {"x": x1, "src_of": {"affine": [1.0, 0.0]}, "n_out": len(x1)}),
        ("wsola", {"x": x1, "src_of": {"affine": [1.0, 300.0]}, "n_out": len(x1) - 300}),
    ]
    for rate in (0.8, 1.25):
        cases.append(("wsola", {"x": x2.tolist(), "src_of": {"affine": [rate, 0.0]}, "n_out": int(len(x2) / rate)}))
    t = np.arange(int(2.0 * SR)) / SR
    bursts = sum(np.exp(-0.5 * ((t - c) / 0.03) ** 2) * np.sin(2 * np.pi * 220 * t) for c in (0.3, 0.7, 1.3, 1.7))
    out_knots = [0.0, 1.0 / 0.8, 1.0 / 0.8 + 1.0 / 1.25]
    cases.append(("wsola", {"x": bursts.tolist(), "src_of": {"interp": {"xp": out_knots, "fp": [0.0, 1.0, 2.0], "sr": SR}},
                            "n_out": int(out_knots[-1] * SR)}))
    # the dub_avsync use: a guarded lip map
    rate = 62.5
    walk = np.cumsum(rng.choice([0.5, 1.0, 1.0, 1.5], size=60)) - 3.0
    m = P.LipMap.from_frames(0.2, rate, P.guard_lip_frames(walk, rate, 1.5, 0.4))
    noise = (rng.standard_normal(int(1.5 * SR)) * 0.3).tolist()
    n_out = int(round(m.out_time(len(noise) / SR) * SR))
    cases.append(("wsola", {"x": noise, "src_of": {"lipmap": {"out_t": m.out_t.tolist(), "src_t": m.src_t.tolist(), "sr": SR}},
                            "n_out": n_out}))
    short = rng.standard_normal(3000).tolist()
    cases += [
        ("wsola", {"x": short, "src_of": {"affine": [0.9, 17.0]}, "n_out": 3500, "window": 256, "tolerance": 64}),
        ("wsola", {"x": short, "src_of": {"affine": [1.3, -40.0]}, "n_out": 2000, "window": 128, "tolerance": 0}),
        ("wsola", {"x": short, "src_of": {"affine": [1.0, 0.0]}, "n_out": 0}),
        ("wsola", {"x": [], "src_of": {"affine": [1.0, 0.0]}, "n_out": 100}),
    ]
    return cases


def _ids(cases):
    return [f"{fn}-{i}" for i, (fn, _) in enumerate(cases)]


_RNG = np.random.default_rng(1234)
EXACT_CASES = (
    [("constants", {})]
    + text_cases()
    + segment_cases(_RNG)
    + rate_cases()
    + choice_cases(_RNG)
    + placement_cases(_RNG)
    + speech_cases(_RNG)
    + window_cases(_RNG)
    + lip_cases(_RNG)
)
WSOLA_CASES = wsola_cases(np.random.default_rng(99))


@pytest.mark.parametrize(("fn", "args"), EXACT_CASES, ids=_ids(EXACT_CASES))
def test_ggml_dub_plan_matches_python(tool, fn, args):
    want = py_result(fn, args)
    got = tool("dub-plan-replay", {"fn": fn, "args": args})
    if fn == "constants":
        got["UNSPACED"] = sorted(got["UNSPACED"])  # a frozenset in Python
    assert_same(got, want, None)


@pytest.mark.parametrize(("fn", "args"), WSOLA_CASES, ids=_ids(WSOLA_CASES))
def test_ggml_dub_plan_wsola_matches_python(tool, fn, args):
    want = py_result(fn, args)
    got = tool("dub-plan-replay", {"fn": fn, "args": args})
    assert_same(got, want, 1e-5)


def test_ggml_dub_plan_covers_every_function():
    ported = {
        "text_units", "unit_bytes", "break_after", "span_text", "join_words", "ref_segments", "project_segments",
        "candidate_plan", "pinned_candidate_plan", "max_tokens_for", "voice_prompt_span", "speech_span",
        "candidate_score", "choose_candidate", "place_segments", "word_windows", "window_of_words", "band_mask",
        "guard_lip_frames", "lip_warp", "wsola",
    }
    public = {n for n, v in vars(P).items() if callable(v) and not n.startswith("_") and getattr(v, "__module__", "") == P.__name__
              and not isinstance(v, type)}
    assert public == ported
    called = {fn for fn, _ in EXACT_CASES + WSOLA_CASES}
    assert ported <= called
    assert {"RateScale", "RateScale.extended", "RateScale.bucket", "RateScale.rate", "LipMap.from_frames",
            "LipMap.src_time", "LipMap.out_time", "LipMap.shifts", "constants"} <= called
