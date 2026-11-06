"""Goldens of the speech-edit logic the in-browser engine ports (servers/web/src/edit): zonos3.tts.textmatch,
align_text, edit_align, edit and edit_runtime.region_doses run on fixed and seeded random inputs.

Run from the repository root with the workspace interpreter:
    .venv/bin/python servers/web/tests/edit/make_goldens.py
It writes servers/web/tests/edit/goldens.json (read by tests/edit/*.test.ts).
"""
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from zonos3.tts import textmatch
from zonos3.tts.align_text import align_tokens
from zonos3.tts.edit import edit_prompt_tokens, extract_fill, sheared_block, splice_edit, validate_edit_spans
from zonos3.tts.edit_align import (
    WordDiff,
    aligned_words,
    canonical_markup,
    diff_spans,
    edit_words,
    expected_fill_seconds,
    plan_edit_windows,
    seconds_to_spans,
    target_tokens,
    token_chars,
)
from zonos3.tts.edit_runtime import region_doses
from zonos3.tts.prompt import TTSPromptConfig

OUT = Path(__file__).with_name("goldens.json")
CONFIG = TTSPromptConfig(n_codebooks=10, audio_pad_id=4097, text_vocab=8196)
K = 10
rng = random.Random(20261106)


def attempt(f):
    try:
        return {"out": f()}
    except ValueError as exc:
        return {"error": str(exc)}


def words_out(words):
    return [{"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence, "tokens": w.tokens}
            for w in words]


def rows(a):
    return np.asarray(a).tolist()


TEXTS = [
    "Hello, world!", "don\u2019t stop", "snake_case_word", "Stra\u00dfe", "na\u00efve caf\u00e9", "e\u0301cole",
    "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8", "Mr. Smith's 2nd   car", "x\u00b2 \u2460", "  ", "",
    "rock'n'roll", "\u0149", "\u00dcn\u00efc\u00f6d\u00e9\u2014dash", "\ufb01ne", "U.S.A.",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35", "\u02bcokina", "tab\tnew\nline\u3000wide", "50% off -- now!",
]
ALIGN_TEXTS = [
    "Hello world.", "  ... leading punctuation here", "\u4eca\u65e5\u306f\u3001\u3044\u3044\u5929\u6c17\u3067\u3059\u306d\u3002",
    "\u6211\u7231ABC123\u548c cats!", "\u300c\u5f15\u7528\u300d\u30c6\u30b9\u30c8", "", "\u2014 ", "abc-def ghi",
    "Rock 'n' roll", "\u4f60 \u597d \u4e16\u754c", "A B. C,", "\uff76\uff80\uff76\uff85 text",
]
MARKUP_TEXTS = [
    "Hello <laugh /> there", "<shouting>Stop it</shouting> now", "Say \ue000\u02c8sjo\u028a\ue001 please",
    "He said\ue000ni\u02d0v\ue001, <sigh/>okay.", "plain text, here.", "<laugh/>", "word<breath  />word",
    "a \ue000b c\ue001 d", "<giggle/> not a tag", "</crying>", "",
]


def gen_tokens():
    return [{"text": t, "out": textmatch.tokens(t)} for t in TEXTS + ALIGN_TEXTS + MARKUP_TEXTS]


def gen_align_tokens():
    return [{"text": t, "lang": lang, "out": align_tokens(t, lang)}
            for t in ALIGN_TEXTS + TEXTS for lang in ("en", "ja", "zh", "yue", "ko")]


def gen_markup():
    canon = [{"text": t, "out": canonical_markup(t)} for t in MARKUP_TEXTS + TEXTS]
    targets = [{"text": t, "out": target_tokens(canonical_markup(t))} for t in MARKUP_TEXTS + TEXTS]
    chars = [{"token": tok, "out": token_chars(tok)} for case in targets for tok in case["out"][0]]
    return canon, targets, chars


def synthetic_alignment(text, lang, start=0.3, overlap=True):
    """FuseAlign-like words of align_tokens(text, lang) with seeded timings."""
    words, t = [], start
    for w in align_tokens(text, lang):
        d = 0.15 + 0.045 * len(w) + rng.random() * 0.1
        words.append({"word": w, "start": round(t, 3), "end": round(t + d, 3), "presence": round(rng.random(), 4)})
        gap = rng.choice([0.0, 0.03, 0.12, 0.4] + ([-0.02] if overlap else []))
        t += d + gap
    return words


def gen_aligned_words():
    out = []
    for text in ["The quick brown fox, jumps over -- the lazy dog.", "One... two? Three!", "\u2014 dash only \u2014",
                 "solo"]:
        fw = synthetic_alignment(text, "en")
        toks = [textmatch.tokens(w) for w in align_tokens(text, "en")]
        out.append({"words": fw, "wordTokens": toks, **attempt(lambda: words_out(aligned_words(fw, toks)))})
    out.append({"words": [{"word": "a", "start": 0, "end": 1, "presence": 1}], "wordTokens": [],
                **attempt(lambda: aligned_words([{"word": "a", "start": 0, "end": 1, "presence": 1}], []))})
    return out


def gen_edit_words():
    base = [{"word": "Don't", "start": 0.1, "end": 0.4, "confidence": 0.9},
            {"word": "stop", "start": 0.4, "end": 0.7, "confidence": None},
            {"word": "now.", "start": 0.8, "end": 1.2}]
    toks = [textmatch.tokens(w["word"]) for w in base]
    orig = textmatch.tokens("Don't stop now.")
    cases = [
        (base, toks, orig, 1.5),
        ([], [], orig, 1.5),
        ([{**base[0], "start": -0.1}] + base[1:], toks, orig, 1.5),
        ([base[0], {**base[1], "end": 0.3}, base[2]], toks, orig, 1.5),
        ([base[0], {**base[1], "start": 0.3}, base[2]], toks, orig, 1.5),
        (base, toks, orig, 1.0),
        (base, toks, textmatch.tokens("Don't stop it now."), 1.5),
        (base, toks, textmatch.tokens("Do not stop now."), 1.5),
        (base, toks, [], 1.5),
        ([{**base[0], "start": 0.0625, "end": 0.0625}] + base[1:], toks, orig, 1.5),
        ([base[0], {**base[1], "start": 0.4 - 5e-7}, base[2]], toks, orig, 1.5),
        (base, toks, orig, 1.2 - 5e-4),
    ]
    out = []
    for words, wt, ot, dur in cases:
        out.append({"words": words, "wordTokens": wt, "originalTokens": ot, "duration": dur,
                    **attempt(lambda: words_out(edit_words(words, wt, ot, duration=dur)))})
    return out


def gen_seconds_to_spans():
    out = []
    for _ in range(40):
        total = rng.randint(1, 400)
        pairs = []
        for _ in range(rng.randint(1, 4)):
            a = round(rng.uniform(-1, total / 15 + 1), rng.choice([1, 2, 3, 6]))
            pairs.append([a, round(a + rng.uniform(-0.2, 3), rng.choice([1, 2, 3, 6]))])
        fr = rng.choice([15.0, 15.0, 12.5, 21.533])
        out.append({"pairs": pairs, "totalFrames": total, "frameRate": fr,
                    "out": [list(s) for s in seconds_to_spans(pairs, total, fr)]})
    out.append({"pairs": [[1 / 15, 2 / 15], [0.2, 0.4]], "totalFrames": 10, "frameRate": 15.0,
                "out": [list(s) for s in seconds_to_spans([[1 / 15, 2 / 15], [0.2, 0.4]], 10, 15.0)]})
    return out


def gen_opcodes():
    import difflib
    out = []
    for _ in range(60):
        alpha = "abcdef"[:rng.randint(1, 6)]
        a = [rng.choice(alpha) for _ in range(rng.randint(0, 14))]
        b = [rng.choice(alpha) for _ in range(rng.randint(0, 14))]
        out.append({"a": a, "b": b, "out": [list(o) for o in
                                            difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()]})
    return out


ORIGINALS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "I think we should meet at noon, not at three. Let's talk later!",
    ("It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
     "foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it "
     "was the season of Darkness, it was the spring of hope, it was the winter of despair."),
    ("Speech editing lets you change a few words in a recording without re-recording the whole thing; the "
     "model fills each edited region in the speaker's voice, keeping every other frame exactly as it was. "
     "Longer clips are cut into windows around the regions so the prompt stays short."),
]


def mutate(words, n_edits):
    words = list(words)
    for _ in range(n_edits):
        op = rng.choice(["replace", "insert", "delete", "group", "marker"])
        k = rng.randrange(len(words))
        if op == "replace":
            words[k] = rng.choice(["purple", "seven", "quickly", "Edinburgh,", "ain't", "tomorrow."])
        elif op == "insert":
            words.insert(k, rng.choice(["really", "very big", "not", "Siobhan"]))
        elif op == "delete" and len(words) > 3:
            del words[k]
        elif op == "group":
            words[k] = rng.choice(["\ue000\u02c8sjo\u028a\ue001", "\ue000ni\u02d0v\ue001,", "a\ue000b\ue001c"])
        elif op == "marker":
            words.insert(k, rng.choice(["<laugh/>", "<sigh />", "<shouting>loud</shouting>"]))
    return " ".join(words)


def gen_scenarios():
    out = []
    for original in ORIGINALS:
        for n_edits in (0, 1, 1, 2, 3, 5):
            fw = synthetic_alignment(original, "en", overlap=False)
            words = aligned_words(fw, [textmatch.tokens(w) for w in align_tokens(original, "en")])
            duration = words[-1].end + 0.5
            total = math.ceil(duration * 15)
            target = canonical_markup(mutate(original.split(), n_edits))
            tt = target_tokens(target)
            target_words = target.split()
            margin = rng.choice([0.0, 0.08, 0.25])
            case = {"words": words_out(words), "targetText": target, "targetWords": target_words,
                    "targetTokens": tt, "totalFrames": total, "margin": margin}
            diff = attempt(lambda: (lambda d: {"spans": [list(s) for s in d.spans],
                                               "newTokens": d.new_target_tokens})(
                diff_spans(words, tt[0], total_frames=total, margin_seconds=margin, max_spans=3)))
            case["diff"] = diff
            span_sets = []
            if "out" in diff:
                span_sets.append(diff["out"]["spans"])
            # explicit spans (seconds_to_spans of random regions)
            for _ in range(2):
                n = rng.randint(1, 3)
                cuts = sorted(rng.sample(range(total), min(total, 2 * n)))
                spans = [[cuts[2 * i], cuts[2 * i + 1] - cuts[2 * i]] for i in range(len(cuts) // 2)]
                spans = [s for s in spans if s[1] > 0]
                if spans:
                    span_sets.append(spans)
            case["windows"], case["expected"] = [], []
            for spans in span_sets:
                for context in (0.0, 2.0, 4.0):
                    res = attempt(lambda: [{"start": w.start, "end": w.end, "regions": w.regions, "text": w.text}
                                           for w in plan_edit_windows(spans, words, target_words, tt[1:],
                                                                      total_frames=total, context_seconds=context,
                                                                      target_text=target)])
                    case["windows"].append({"spans": spans, "context": context, **res})
                case["expected"].append({"spans": spans, **attempt(
                    lambda: expected_fill_seconds(spans, words, target_words, tt[1:]))})
            diffw = WordDiff(words, target_words, tt[1:])
            case["wordDiff"] = [{"k": k, "left": diffw.target_word(k, True), "right": diffw.target_word(k, False)}
                                for k in range(len(words) + 1)]
            out.append(case)
    return out


def gen_region_doses():
    cases = [
        (None, None, None), ({"a.x": 1.0}, None, None), ({"a.x": 0.0}, None, None),
        ({"a.x": 1.0}, [None, {"a.x": -1.0, "b.y": 0.5}], [[0, 1], [2, 3]]),
        (None, [{"b.y": 0.0}, None], [[0, 1], [2, 3]]), (None, [{"b.y": 1}], None),
        (None, [{"b.y": 1}], [[0, 1], [2, 3]]), ({"a.x": 0.5, "b.y": 0.25}, [{"b.y": -0.25}], [[0, 1]]),
    ]
    out = []
    for controls, region, spans in cases:
        req = SimpleNamespace(controls=controls, region_controls=region, spans=spans)
        out.append({"controls": controls, "regionControls": region, "spans": spans,
                    **attempt(lambda: region_doses(req, None))})
    return out


def gen_validate_spans():
    cases = [([[0, 1]], 10, 3), ([], 10, 3), ([[0, 1]] * 4, 100, 3), ([[2, 3], [5, 1]], 10, 3),
             ([[2, 3], [6, 1]], 10, 3), ([[-1, 2]], 10, 3), ([[0, 0]], 10, 3), ([[8, 3]], 10, 3),
             ([[8, 2]], 10, 3), ([[3, 1], [1, 1]], 10, 3)]
    out = []
    for spans, total, mx in cases:
        try:
            validate_edit_spans(spans, total, mx)
            err = None
        except ValueError as exc:
            err = str(exc)
        out.append({"spans": spans, "totalFrames": total, "maxSpans": mx, "error": err})
    return out


def codes(n):
    return [[rng.randrange(4096) for _ in range(K)] for _ in range(n)]


def gen_frames():
    sheared = []
    for n in (0, 1, 3, 12):
        for end in (None, 4096, 4098):
            fr = codes(n)
            sheared.append({"frames": fr, "endId": end,
                            "out": rows(sheared_block(np.asarray(fr, dtype=np.int32).reshape(n, K), CONFIG,
                                                      end_id=end))})
    prompts = []
    for _ in range(14):
        total = rng.randint(4, 24)
        n_spans = rng.randint(1, 3)
        cuts = sorted(rng.sample(range(total), min(total, 2 * n_spans)))
        spans = [[cuts[2 * i], max(1, cuts[2 * i + 1] - cuts[2 * i])] for i in range(len(cuts) // 2)]
        try:
            validate_edit_spans(spans, total, 3)
        except ValueError:
            spans = [[1, 2]]
        n_fills = rng.randint(0, len(spans))
        fills = [codes(rng.randint(0, 4)) for _ in range(n_fills)]
        text = [[4097] * K + [rng.randrange(8196)] for _ in range(rng.randint(1, 5))]
        cs = codes(total)
        span_end = rng.choice([4098, 4098, 4098, 4099])
        res = attempt(lambda: rows(edit_prompt_tokens(
            np.asarray(text, dtype=np.int32), np.asarray(cs), spans,
            [np.asarray(f, dtype=np.int32).reshape(-1, K) for f in fills], CONFIG, span_end_id=span_end)))
        prompts.append({"textRows": text, "codes": cs, "spans": spans, "fills": fills, "spanEndId": span_end, **res})
    fills = []
    for _ in range(30):
        n = rng.randint(0, 24)
        r = [[rng.choice([rng.randrange(4096)] * 30 + [4096, 4097, 4098]) for _ in range(K)] for _ in range(n)]
        eos = rng.choice([None, None, rng.randint(0, n + 3)])
        fill, ended = extract_fill(np.asarray(r, dtype=np.int64).reshape(n, K), eos, audio_pad_id=4097,
                                   codebook_sizes=[4096] * K)
        fills.append({"rows": r, "eosFrame": eos, "fill": rows(fill), "ended": ended})
    splices = []
    for _ in range(12):
        total = rng.randint(3, 20)
        n_spans = rng.randint(1, 3)
        cuts = sorted(rng.sample(range(total), min(total, 2 * n_spans)))
        spans = [[cuts[2 * i], max(1, cuts[2 * i + 1] - cuts[2 * i])] for i in range(len(cuts) // 2)]
        fl = [codes(rng.randint(0, 5)) for _ in spans]
        if rng.random() < 0.15:
            fl = fl[:-1]
        cs = codes(total)
        res = attempt(lambda: rows(splice_edit(np.asarray(cs), spans,
                                               [np.asarray(f, dtype=np.int64).reshape(-1, K) for f in fl])))
        splices.append({"codes": cs, "spans": spans, "fills": fl, **res})
    return sheared, prompts, fills, splices


def main():
    canon, targets, chars = gen_markup()
    sheared, prompts, fills, splices = gen_frames()
    goldens = {
        "tokens": gen_tokens(), "alignTokens": gen_align_tokens(), "canonicalMarkup": canon,
        "targetTokens": targets, "tokenChars": chars, "alignedWords": gen_aligned_words(),
        "editWords": gen_edit_words(), "secondsToSpans": gen_seconds_to_spans(), "opcodes": gen_opcodes(),
        "scenarios": gen_scenarios(), "regionDoses": gen_region_doses(), "validateSpans": gen_validate_spans(),
        "shearedBlock": sheared, "editPrompt": prompts, "extractFill": fills, "splice": splices,
    }
    OUT.write_text(json.dumps(goldens, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
