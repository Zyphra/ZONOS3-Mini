"""Parity of src/zonos3-edit.cpp and src/zonos3-textmatch.cpp (via the edit-replay binary) with
model/src/zonos3/tts/{edit,edit_align,textmatch}.py and Python's str.upper()/str.lower().

Every case runs the same call through ``edit-replay`` ({"fn", "args"} JSON on stdin) and through the Python module;
results must be equal (floats within 1e-9 relative). ``test_fixture_file_is_current`` keeps
tests/fixtures/edit-replay/cases.json (the ctest replay) equal to the Python answers; regenerate it with
``ZONOS3_WRITE_EDIT_FIXTURES=1``.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import random
from pathlib import Path

import pytest
import torch

from zonos3.tts import edit as E
from zonos3.tts import edit_align as A
from zonos3.tts import textmatch as T
from zonos3.tts.prompt import TTSPromptConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "edit-replay" / "cases.json"
K, PAD, TEXT_VOCAB, SPAN_END = 4, 13, 500, 14   # a small layout: codebook 12, EOA 12, pad 13, span-end 14
G = "\ue000{}\ue001".format


def _words(spec):
    """[(word, start, end, tokens)] -> AlignedWord dicts."""
    return [{"word": w, "start": s, "end": e, "confidence": 0.9, "tokens": t} for w, s, e, t in spec]


WORDS = _words([
    ("The", 0.10, 0.30, ["THE"]), ("quick", 0.35, 0.70, ["QUICK"]), ("brown", 0.80, 1.10, ["BROWN"]),
    ("fox", 1.20, 1.50, ["FOX"]), ("jumps", 1.70, 2.10, ["JUMPS"]), ("over", 2.20, 2.40, ["OVER"]),
    ("the", 2.45, 2.55, ["THE"]), ("lazy", 2.60, 2.95, ["LAZY"]), ("dog's", 3.00, 3.40, ["DOG'S"]),
    ("back", 3.50, 3.90, ["BACK"]), ("twenty-one", 4.10, 4.90, ["TWENTY", "ONE"]), ("times.", 5.00, 5.50, ["TIMES"]),
])


def _target(words: list[str]):
    """Target words and their (fake, upper-cased) edit tokens; a group is a token of its own."""
    tokens = [[w] if A.is_group_token(w) else [t for t in w.upper().replace("-", " ").replace(".", "").split()] for w in words]
    return words, tokens


def _cases() -> list[dict]:
    rng = random.Random(1234)
    cases = []

    def add(name, fn, **args):
        cases.append({"name": name, "fn": fn, "args": args})

    def codes(n):
        return [[rng.randrange(12) for _ in range(K)] for _ in range(n)]

    layout = {"n_codebooks": K, "audio_pad_id": PAD, "text_vocab": TEXT_VOCAB, "span_end_id": SPAN_END}
    text_rows = [[PAD] * K + [t] for t in (2, 70, 71, 3)]
    c20 = codes(20)
    add("prompt-one-span", "edit_prompt_tokens", text_rows=text_rows, codes=c20, spans=[[5, 4]], fills=[], **layout)
    add("prompt-span-at-0", "edit_prompt_tokens", text_rows=text_rows, codes=c20, spans=[[0, 3], [10, 2]], fills=[], **layout)
    add("prompt-second-fill", "edit_prompt_tokens", text_rows=text_rows, codes=c20, spans=[[2, 3], [9, 4], [15, 5]],
        fills=[codes(6), codes(2)], **layout)
    add("prompt-span-to-end", "edit_prompt_tokens", text_rows=text_rows, codes=c20, spans=[[14, 6]], fills=[], **layout)
    add("prompt-no-ungenerated", "edit_prompt_tokens", text_rows=text_rows, codes=c20, spans=[[2, 3]], fills=[codes(1)], **layout)
    add("prompt-bad-span-end", "edit_prompt_tokens", text_rows=text_rows, codes=c20, spans=[[2, 3]], fills=[],
        **{**layout, "span_end_id": 15})
    add("prompt-overlap", "edit_prompt_tokens", text_rows=text_rows, codes=c20, spans=[[2, 3], [5, 2]], fills=[], **layout)

    def delayed(frames, end):
        """Generated delayed rows of `frames` followed by an end row (cut like the server: no finishing row)."""
        audio = frames + ([[end] * K] if end is not None else []) + [[PAD] * K] * (K - 1)
        rows = [[audio[r - c][c] if r >= c else PAD for c in range(K)] for r in range(len(audio))]
        return rows
    f7 = codes(7)
    add("fill-eos", "extract_fill", delayed_rows=delayed(f7, SPAN_END)[:-1], eos_frame=7, audio_pad_id=PAD, codebook_sizes=[12] * K)
    add("fill-no-eos", "extract_fill", delayed_rows=delayed(f7, None)[:9], eos_frame=None, audio_pad_id=PAD, codebook_sizes=[12] * K)
    bad = [row[:] for row in f7]
    bad[4][2] = SPAN_END
    add("fill-truncated", "extract_fill", delayed_rows=delayed(bad, SPAN_END)[:-1], eos_frame=7, audio_pad_id=PAD,
        codebook_sizes=[12] * K)
    add("fill-limits", "extract_fill", delayed_rows=delayed(f7, SPAN_END)[:-1], eos_frame=7, audio_pad_id=PAD,
        codebook_sizes=[12, 12, 5, 12])
    add("fill-empty", "extract_fill", delayed_rows=delayed([], SPAN_END)[:-1], eos_frame=0, audio_pad_id=PAD, codebook_sizes=[12] * K)
    add("splice", "splice_edit", codes=c20, spans=[[2, 3], [9, 4], [15, 5]], fills=[codes(6), codes(0), codes(1)])
    add("splice-count", "splice_edit", codes=c20, spans=[[2, 3], [9, 4]], fills=[codes(6)])
    add("validate-ok", "validate_edit_spans", spans=[[0, 1], [2, 3]], total_frames=5, max_spans=3)
    add("validate-adjacent", "validate_edit_spans", spans=[[0, 2], [2, 3]], total_frames=5, max_spans=3)
    add("validate-too-many", "validate_edit_spans", spans=[[0, 1], [3, 1]], total_frames=5, max_spans=1)
    add("seconds", "seconds_to_spans", pairs=[[0.0, 0.2], [1.0 / 15, 2.0 / 15], [3.33, 3.9999], [9.5, 12.0]],
        total_frames=150, frame_rate=15.0)

    original = [t for w in WORDS for t in w["tokens"]]
    for name, b in [("same", original), ("replace", original[:3] + ["RED"] + original[4:]),
                    ("insert", original[:5] + ["VERY", "HIGH"] + original[5:]), ("delete", original[:2] + original[4:]),
                    ("mixed", ["A"] + original[1:7] + ["CRAZY"] + original[8:11] + ["TWO", "TIMES", "TODAY"]),
                    ("dups", ["THE", "THE", "QUICK", "THE", "FOX", "THE"])]:
        add(f"opcodes-{name}", "sequence_opcodes", a=original, b=b)
    for name, target, margin, max_spans in [
        ("one-word", original[:3] + ["RED"] + original[4:], 0.08, 3),
        ("insert-start", ["WELL"] + original, 0.08, 3),
        ("append-end", original + ["AGAIN"], 0.08, 3),
        ("merge-gap", original[:3] + ["RED", "FAT"] + original[5:], 0.3, 3),
        ("two", original[:1] + ["SLOW"] + original[2:8] + ["SLEEPY"] + original[9:], 0.08, 3),
        ("too-many", ["A"] + original[1:4] + ["B"] + original[5:9] + ["C"] + original[10:], 0.0, 2),
        ("equal", original, 0.08, 3),
    ]:
        add(f"diff-{name}", "diff_spans", words=WORDS, target_tokens=target, total_frames=90, frame_rate=15.0,
            margin_seconds=margin, min_gap_frames=5, max_spans=max_spans)

    plain = [w["word"] for w in WORDS]
    for name, words, spans, ctx in [
        ("one", plain[:3] + ["red"] + plain[4:], [[16, 8]], 1.0),
        ("merge", plain[:1] + ["slow"] + plain[2:8] + ["sleepy"] + plain[9:], [[4, 8], [38, 7]], 1.0),
        ("apart", plain[:1] + ["slow"] + plain[2:10] + ["twenty-two"] + plain[11:], [[4, 8], [60, 14]], 0.5),
        ("whole", plain[:3] + ["red"] + plain[4:], [[16, 8]], 10.0),
        ("insert", plain[:5] + ["very", "high"] + plain[5:], [[30, 4]], 0.5),
        ("group", plain[:3] + [G("ɹɛd")] + plain[4:], [[16, 8]], 1.0),
        ("join-word", plain[:7] + ["lazydog's"] + plain[9:], [[38, 14]], 0.3),
    ]:
        tw, tt = _target(words)
        add(f"windows-{name}", "plan_edit_windows", spans=spans, words=WORDS, target_words=tw, target_word_tokens=tt,
            total_frames=90, frame_rate=15.0, context_seconds=ctx, target_text=" ".join(tw))
        add(f"expected-{name}", "expected_fill_seconds", spans=spans, words=WORDS, target_words=tw,
            target_word_tokens=tt, frame_rate=15.0)
    tw, tt = _target(plain)
    add("windows-no-text", "plan_edit_windows", spans=[[16, 8]], words=WORDS, target_words=tw, target_word_tokens=tt,
        total_frames=90, frame_rate=15.0, context_seconds=0.2, target_text=None)
    add("windows-no-words", "plan_edit_windows", spans=[[16, 8]], words=[], target_words=tw, target_word_tokens=tt,
        total_frames=90, frame_rate=15.0, context_seconds=0.2, target_text=None)
    add("expected-pure-insert", "expected_fill_seconds", spans=[[85, 5]], words=WORDS,
        target_words=tw + ["again"], target_word_tokens=tt + [["AGAIN"]], frame_rate=15.0)

    for name, ref, hyp in [
        ("exact", ["HELLO", G("ˈwɝld")], ["HELLO", "WORLD"]),
        ("missing", ["SAY", G("ʃɪv"), "NOW"], ["SAY", "NOW"]),
        ("extra", ["SAY", G("ʃɪv"), "NOW"], ["SAY", "SHIV", "RIGHT", "NOW"]),
        ("errors", ["THE", "CAT", "SAT"], ["A", "CAT", "SAD", "DOWN"]),
        ("empty-hyp", ["THE", "CAT"], []),
        ("unicode", ["CAFÉ", G("a")], ["CAFE", "OUI"]),
    ]:
        add(f"verify-{name}", "wildcard_verify", reference_tokens=ref, hypothesis_tokens=hyp)
    add("fill-groups", "fill_groups", text=f"say {G('ʃɪv')} now {G('x')}  ok", heard=["SHIV", None])
    add("fill-groups-latin", "fill_groups", text=f"{G('ka')} é", heard=["ÉCOLE"])
    add("canonical", "canonical_markup", text="Hi <laugh /> there < sigh/> <shouting >Out</shouting > <nope />")
    add("canonical-plain", "canonical_markup", text="no markup here")

    for name, text in [("plain", "Hello, big world"), ("group", f"Hi {G('ʃɪv')}, how{G('ɑɹ')} you"),
                       ("markup", "Oh <laugh/> really<sigh/>,  yes"), ("both", f"{G('a')}<laugh/>b c"),
                       ("cjk", "我们今天 去公园。 日本語、です"), ("quotes", "don’t ‘stop’ snake_case 42nd")]:
        add(f"target-{name}", "target_tokens", target_text=text)

    good = [{"word": "Hello", "start": 0.1, "end": 0.5, "confidence": 0.9}, {"word": "there", "start": 0.6, "end": 1.0}]
    add("words-ok", "edit_words", words=good, word_tokens=[["HELLO"], ["THERE"]], original_tokens=["HELLO", "THERE"], duration=1.2)
    add("words-spell", "edit_words", words=good, word_tokens=[["HELLO"], ["THEIR"]], original_tokens=["HELLO", "THERE", "NOW"],
        duration=1.2)
    add("words-short-spell", "edit_words", words=good, word_tokens=[["HELLO"], []], original_tokens=["HELLO", "THERE"], duration=1.2)
    add("words-order", "edit_words", words=[good[1], good[0]], word_tokens=[["THERE"], ["HELLO"]],
        original_tokens=["THERE", "HELLO"], duration=1.2)
    add("words-late", "edit_words", words=good, word_tokens=[["HELLO"], ["THERE"]], original_tokens=["HELLO", "THERE"], duration=0.9)
    add("words-backwards", "edit_words", words=[{"word": "it's", "start": 0.5, "end": 0.5}], word_tokens=[["IT'S"]],
        original_tokens=["IT'S"], duration=1.0)
    add("words-negative", "edit_words", words=[{"word": "a", "start": -0.5, "end": 0.5}], word_tokens=[["A"]],
        original_tokens=["A"], duration=1.0)
    add("words-empty", "edit_words", words=[], word_tokens=[], original_tokens=[], duration=1.0)
    placed = [{"word": "Hello,", "start": 0.1, "end": 0.5, "presence": 0.97, "placed": True},
              {"word": "—", "start": 0.5, "end": 0.5, "presence": 0.0, "placed": False},
              {"word": "there", "start": 0.6, "end": 1.0, "presence": 0.4, "placed": True}]
    add("aligned-ok", "aligned_words", words=placed, word_tokens=[["HELLO"], [], ["THERE"]])
    add("aligned-empty", "aligned_words", words=[], word_tokens=[])
    add("aligned-mismatch", "aligned_words", words=placed, word_tokens=[["HELLO"]])
    overlapping = [{"word": "a", "start": 0.10, "end": 0.502, "presence": 0.9, "placed": True},
                   {"word": ",", "start": 0.45, "end": 0.46, "presence": 0.0, "placed": False},
                   {"word": "b", "start": 0.50, "end": 0.801, "presence": 0.8, "placed": True},
                   {"word": "c", "start": 0.70, "end": 0.75, "presence": 0.7, "placed": True},
                   {"word": "d", "start": 0.74, "end": 1.20, "presence": 0.6, "placed": True}]
    add("aligned-overlap", "aligned_words", words=overlapping, word_tokens=[["A"], [], ["B"], ["C"], ["D"]])
    asr_languages = ["Chinese", "English", "Cantonese", "French", "German", "Japanese", "Korean", "Portuguese",
                     "Spanish", "Italian", "Russian"]
    for lang in ("en_us", "en_gb", "cmn", "pt_br", "ko", "xx"):
        add(f"lang-{lang}", "edit_language", language=lang, supported=asr_languages)
    add("lang-unsupported-name", "edit_language", language="ko", supported=["English", "French"])
    add("lang-none", "edit_language", language="en_us", supported=[])
    add("split", "split_words", text=" a\u3000b\tc\u00a0 d\u200be ")
    add("chars-group", "token_chars", token=G("ʃɪv"))
    add("chars-word", "token_chars", token="ÉCOLE")

    # zonos3.tts.textmatch: Python's \w (letters, numbers, _), str.upper() (full mappings) and curly quotes
    texts = [
        "Hello, world!", "don’t ‘stop’ ʼtil it‛s_done", "snake_case __x__ a_1", "Straße ﬁne ŉ ΐ ǅemal İstanbul",
        "我们今天去公园。我们 今天", "日本語の文章です、はい。カタカナ ｶﾀｶﾅ", "한국어 텍스트입니다!", "Café 123 ٣٤٥ Ⅻ ½ ² ①",
        "ὈΔΥΣΣΕΎΣ ΣΑΣ σοφΣ. Σ ΑΣ' ΑΣ.Α", "हिन्दी भाषा", "  ", "", "tab\tnew\nline\u3000x\u00a0y", "'quoted' — dash… 'tis",
        "Ⓐⓑ ﬀ ﬃ ʰᵃ", "Привет, МИР ёЁ", "x\u0301y\u0308 e\u20ddz", "\u2160\u2161 \U0001d400\U0001d7ce", "🙂 emoji 🙂ok",
    ]
    rng2 = random.Random(77)
    pools = [(0x20, 0x250), (0x370, 0x3FF), (0x400, 0x4FF), (0x900, 0x97F), (0x3040, 0x30FF), (0x4E00, 0x4F00),
             (0x1E00, 0x1FFF), (0xAC00, 0xAC80), (0x2000, 0x2070), (0xFB00, 0xFB06)]
    for _ in range(40):
        texts.append("".join(chr(rng2.randrange(*rng2.choice(pools))) if rng2.random() < 0.8
                             else rng2.choice(["_", "\u2019", " ", "'", "\u03a3", "\u02bc"]) for _ in range(rng2.randrange(1, 24))))
    for k, text in enumerate(texts):
        add(f"normalize-{k}", "normalize", text=text)
        add(f"tokens-{k}", "tokens", text=text)
        add(f"upper-{k}", "upper", text=text)
        add(f"lower-{k}", "lower", text=text)
    for k in range(0, len(texts) - 1):
        add(f"cer-{k}", "char_error_rate", reference=texts[k], hypothesis=texts[k + 1])
        add(f"verify-text-{k}", "verify", reference=texts[k], hypothesis=texts[(k * 7 + 3) % len(texts)])
    add("cer-same", "char_error_rate", reference="Hello world", hypothesis="hello, WORLD!")
    add("cer-empty-ref", "char_error_rate", reference="—", hypothesis="abc")
    add("cer-empty-hyp", "char_error_rate", reference="abc", hypothesis="...")
    add("cer-cjk", "char_error_rate", reference="我们今天去公园", hypothesis="我们明天去公园。")
    for name, ref, hyp in [("same", ["A", "B"], ["A", "B"]), ("sub", ["A", "B", "C"], ["A", "X", "C"]),
                           ("ins-del", ["A", "B", "C"], ["B", "C", "D", "E"]), ("empty-ref", [], ["A"]),
                           ("empty-hyp", ["A", "B"], []), ("both-empty", [], [])]:
        add(f"word-errors-{name}", "word_errors", reference=ref, hypothesis=hyp)
    add("verify-text-cjk", "verify", reference="我们 今天 去 公园", hypothesis="我们今天去公园")
    return cases


# ---------------------------------------------------------------------------------------------------- python side


def _t(x):
    return torch.tensor(x, dtype=torch.int64).reshape(-1, K) if not x else torch.tensor(x, dtype=torch.int64)


def _aligned(ws):
    return [A.AlignedWord(**w) for w in ws]


def _word_json(w: A.AlignedWord):
    return {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence, "tokens": w.tokens}


def py_call(fn: str, a: dict):
    if fn == "edit_prompt_tokens":
        config = TTSPromptConfig(n_codebooks=a["n_codebooks"], audio_pad_id=a["audio_pad_id"], text_vocab=a["text_vocab"])
        rows = E.edit_prompt_tokens(torch.tensor(a["text_rows"], dtype=torch.int32), torch.tensor(a["codes"]),
                                    a["spans"], [torch.tensor(f).reshape(-1, a["n_codebooks"]) for f in a["fills"]],
                                    config, span_end_id=a["span_end_id"])
        return rows.tolist()
    if fn == "extract_fill":
        k = len(a["codebook_sizes"])
        rows = torch.tensor(a["delayed_rows"], dtype=torch.int64).reshape(-1, k)
        fill, ended = E.extract_fill(rows, a["eos_frame"], audio_pad_id=a["audio_pad_id"], codebook_sizes=a["codebook_sizes"])
        return [fill.tolist(), ended]
    if fn == "splice_edit":
        k = len(a["codes"][0])
        return E.splice_edit(torch.tensor(a["codes"]), a["spans"], [torch.tensor(f).reshape(-1, k) for f in a["fills"]]).tolist()
    if fn == "validate_edit_spans":
        return E.validate_edit_spans(a["spans"], a["total_frames"], a["max_spans"])
    if fn == "seconds_to_spans":
        return [list(s) for s in A.seconds_to_spans(a["pairs"], a["total_frames"], a["frame_rate"])]
    if fn == "sequence_opcodes":
        return [list(o) for o in difflib.SequenceMatcher(None, a["a"], a["b"], autojunk=False).get_opcodes()]
    if fn == "diff_spans":
        d = A.diff_spans(_aligned(a["words"]), a["target_tokens"], total_frames=a["total_frames"], frame_rate=a["frame_rate"],
                         margin_seconds=a["margin_seconds"], min_gap_frames=a["min_gap_frames"], max_spans=a["max_spans"])
        return {"spans": [list(s) for s in d.spans], "new_target_tokens": list(d.new_target_tokens)}
    if fn == "plan_edit_windows":
        ws = A.plan_edit_windows(a["spans"], _aligned(a["words"]), a["target_words"], a["target_word_tokens"],
                                 total_frames=a["total_frames"], frame_rate=a["frame_rate"],
                                 context_seconds=a["context_seconds"], target_text=a["target_text"])
        return [{"start": w.start, "end": w.end, "regions": w.regions, "text": w.text,
                 "spans": [list(s) for s in w.spans(a["spans"])]} for w in ws]
    if fn == "expected_fill_seconds":
        return A.expected_fill_seconds(a["spans"], _aligned(a["words"]), a["target_words"], a["target_word_tokens"],
                                       frame_rate=a["frame_rate"])
    if fn == "wildcard_verify":
        return A.wildcard_verify(a["reference_tokens"], a["hypothesis_tokens"])
    if fn == "fill_groups":
        return A.fill_groups(a["text"], a["heard"])
    if fn == "canonical_markup":
        return A.canonical_markup(a["text"])
    if fn == "target_tokens":
        return A.target_tokens(a["target_text"])
    if fn == "edit_words":
        return [_word_json(w) for w in A.edit_words(a["words"], a["word_tokens"], a["original_tokens"], duration=a["duration"])]
    if fn == "aligned_words":
        return [_word_json(w) for w in A.aligned_words(a["words"], a["word_tokens"])]
    if fn == "edit_language":
        return list(A.edit_language(a["language"], a["supported"]))
    if fn == "split_words":
        return a["text"].split()
    if fn == "token_chars":
        return A.token_chars(a["token"])
    if fn == "normalize":
        return T.normalize(a["text"])
    if fn == "tokens":
        return T.tokens(a["text"])
    if fn == "upper":
        return a["text"].upper()
    if fn == "lower":
        return a["text"].lower()
    if fn == "char_error_rate":
        return T.char_error_rate(a["reference"], a["hypothesis"])
    if fn == "word_errors":
        return T.word_errors(a["reference"], a["hypothesis"])
    if fn == "verify":
        return T.verify(a["reference"], a["hypothesis"])
    raise AssertionError(fn)


def py_run(case: dict):
    try:
        return py_call(case["fn"], case["args"])
    except ValueError as exc:
        return {"error": "ValueError", "message": str(exc)}


def _same(x, y) -> bool:
    if isinstance(x, bool) or isinstance(y, bool):
        return x == y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x == y or math.isclose(x, y, rel_tol=1e-9)
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    return x == y


CASES = _cases()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_edit_replay(tool, case):
    want = json.loads(json.dumps(py_run(case)))
    got = tool("edit-replay", {"fn": case["fn"], "args": case["args"]})
    assert _same(got, want), f"{case['name']}: ggml {got!r} != python {want!r}"


def test_fixture_file_is_current():
    want = [{**c, "expected": json.loads(json.dumps(py_run(c)))} for c in CASES]
    if os.environ.get("ZONOS3_WRITE_EDIT_FIXTURES"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(want, ensure_ascii=False, indent=1) + "\n")
    assert FIXTURE.exists(), f"{FIXTURE} is missing (ZONOS3_WRITE_EDIT_FIXTURES=1)"
    assert _same(json.loads(FIXTURE.read_text()), want), "regenerate with ZONOS3_WRITE_EDIT_FIXTURES=1"
