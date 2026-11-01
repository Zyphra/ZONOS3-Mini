"""Context windows of long-clip edits (zonos3.tts.edit_align.plan_edit_windows)."""
import re

import pytest
from zonos3.tts.edit_align import AlignedWord, plan_edit_windows, seconds_to_spans

FPS = 15


def tokens(word):
    return re.sub(r"[^\w']+", " ", word.upper()).split()


def clip(texts):
    """Word k spoken over [k + 0.1, k + 0.9] s; the clip lasts len(texts) seconds."""
    return [AlignedWord(t, k + 0.1, k + 0.9, 1.0, tokens(t)) for k, t in enumerate(texts)]


def plan(original, target, regions_seconds, context=4.0):
    words = clip(original)
    total = len(original) * FPS
    spans = seconds_to_spans(regions_seconds, total, FPS)
    target_words = target.split()
    windows = plan_edit_windows(spans, words, target_words, [tokens(w) for w in target_words],
                                total_frames=total, frame_rate=FPS, context_seconds=context, target_text=target)
    for w in windows:  # every region lies inside its window
        assert all(w.start <= spans[k][0] and spans[k][0] + spans[k][1] <= w.end for k in w.regions)
    return spans, windows


ORIGINAL = [f"w{k}" for k in range(30)]


def test_far_regions_get_their_own_windows_with_their_target_text():
    target = " ".join(ORIGINAL).replace("w2 ", "x2a x2b ").replace("w25", "y25.")
    spans, windows = plan(ORIGINAL, target, [[2.02, 2.98], [25.02, 25.98]])
    assert [w.regions for w in windows] == [[0], [1]]
    first, second = windows
    # words 0-6 (word 6 ends before 7 s = region end + 4 s); the cut is halfway to word 7
    assert (first.start, first.end) == (0, 105) and first.text == "w0 w1 x2a x2b w3 w4 w5 w6"
    # word 21 is the first to end after 21 s; the window runs to the clip end
    assert (second.start, second.end) == (315, 450)
    assert second.text == "w21 w22 w23 w24 y25. w26 w27 w28 w29"
    assert second.spans(spans) == [(spans[1][0] - 315, spans[1][1])]


def test_overlapping_windows_merge():
    spans, windows = plan(ORIGINAL, " ".join(ORIGINAL).replace("w10", "a").replace("w14", "b"),
                          [[10.0, 11.0], [14.0, 15.0]])
    assert len(windows) == 1 and windows[0].regions == [0, 1]
    assert windows[0].text == "w6 w7 w8 w9 a w11 w12 w13 b w15 w16 w17 w18"


def test_an_edge_inside_a_changed_block_moves_to_the_block_edge():
    # w3..w6 become "z"; a region at w9 would start its window at w5, inside that block
    target = " ".join(ORIGINAL).replace("w3 w4 w5 w6", "z").replace("w9", "q")
    _, (window,) = plan(ORIGINAL, target, [[9.0, 10.0]])
    assert window.text.startswith("z w7 w8 q") and window.start == round(3.0 * FPS)


def test_an_edge_inside_a_target_word_moves_outward():
    # "twenty five" (two original words) is one target word: no cut may fall between them
    original = ORIGINAL[:4] + ["twenty", "five"] + ORIGINAL[6:]
    target = " ".join(original).replace("twenty five", "twenty-five").replace("w9", "q")
    _, (window,) = plan(original, target, [[9.0, 10.0]])   # 4 s of context starts the window at "five"
    assert window.text.startswith("twenty-five w6") and window.start == round(4.0 * FPS)


def test_a_window_over_the_whole_clip_keeps_the_target_text():
    original = ORIGINAL[:12]
    target = "w0  w1 w2 new w3 w4 w5 w6 w7 w8 w9 w10 w11"
    _, (window,) = plan(original, target, [[2.0, 4.0]], context=10.0)
    assert (window.start, window.end, window.text) == (0, 12 * FPS, target)


def test_insertions_inside_a_window_are_kept():
    target = " ".join(ORIGINAL).replace("w20 w21", "w20 brand new w21")
    _, (window,) = plan(ORIGINAL, target, [[20.5, 21.5]])
    assert "w20 brand new w21" in window.text


def test_no_words_cannot_be_windowed():
    with pytest.raises(ValueError, match="no words"):
        plan_edit_windows([(0, 5)], [], ["a"], [["A"]], total_frames=450, frame_rate=FPS)


def expected(original, target, regions_seconds):
    from zonos3.tts.edit_align import expected_fill_seconds

    target_words = target.split()
    spans = seconds_to_spans(regions_seconds, len(original) * FPS, FPS)
    return expected_fill_seconds(spans, clip(original), target_words, [tokens(w) for w in target_words], frame_rate=FPS)


def test_expected_fill_scales_the_region_by_its_text_length():
    original = ["i", "was", "twelve", "years", "old"]
    # re-say: the region's own length; "twelve" -> "thirteen": 8 / 6 of it (zsa token characters)
    assert expected(original, "i was twelve years old", [[2.0, 3.0]]) == pytest.approx([1.0])
    assert expected(original, "i was thirteen years old", [[2.0, 3.0]]) == pytest.approx([8 / 6])
    # the region holds "was twelve" (word midpoints), and a deletion shrinks it to "was"
    assert expected(original, "i was years old", [[1.0, 3.0]]) == pytest.approx([2.0 * 3 / 9])


def test_expected_fill_of_a_pure_insertion_uses_the_clip_rate():
    original = ["i", "was", "twelve", "years", "old"]   # 18 characters in 4 s of words
    (seconds,) = expected(original, "i was nearly twelve years old", [[1.92, 2.08]])
    assert seconds == pytest.approx(6 / (18 / 4.0))


def _given(*triples):
    return [{"word": w, "start": s, "end": e} for w, s, e in triples]


@pytest.mark.parametrize("words, message", [
    ([], "empty"),
    (_given(("born", -0.1, 0.3), ("and", 0.3, 0.5), ("raised", 0.5, 0.9)), "before the clip"),
    (_given(("born", 0.1, float("nan")), ("and", 0.3, 0.5), ("raised", 0.5, 0.9)), "non-finite"),
    (_given(("born", 0.1, 0.3), ("and", 0.5, 0.5), ("raised", 0.5, 0.9)), "start before it ends"),
    (_given(("born", 0.1, 0.3), ("and", 0.25, 0.5), ("raised", 0.5, 0.9)), "before the previous word ends"),
    (_given(("born", 0.1, 0.3), ("and", 0.3, 0.5), ("raised", 0.5, 2.5)), "after the clip"),
    (_given(("born", 0.1, 0.3), ("raised", 0.5, 0.9)), "do not spell original_text"),
])
def test_bad_original_words_are_refused(words, message):
    from zonos3.tts.edit_align import edit_words

    with pytest.raises(ValueError, match=message):
        edit_words(words, [tokens(w["word"]) for w in words], ["BORN", "AND", "RAISED"], duration=2.0)


def test_corrected_word_timings_move_expected_lengths_and_windows():
    """MMS stretched "raised" over a pause; the corrected word lies inside the region."""
    from zonos3.tts.edit_align import edit_words, expected_fill_seconds

    text = "i was born and raised near minneapolis".split()
    mms = _given(("i", 2.24, 2.35), ("was", 2.35, 2.53), ("born", 2.53, 2.83), ("and", 2.83, 2.99),
                 ("raised", 3.05, 4.40), ("near", 4.40, 4.64), ("minneapolis", 4.64, 5.53))
    fixed = [dict(w, end=3.30) if w["word"] == "raised" else w for w in mms]
    span = seconds_to_spans([[3.0, 3.35]], 100, FPS)
    word_tokens = [tokens(w) for w in text]
    plans = []
    for given in (mms, fixed):
        words = edit_words(given, word_tokens, [t for ts in word_tokens for t in ts], duration=6.0)
        plans.append((expected_fill_seconds(span, words, text, word_tokens, frame_rate=FPS),
                      plan_edit_windows(span, words, text, word_tokens, total_frames=100, frame_rate=FPS,
                                        context_seconds=0.5)[0]))
    (mms_expected, mms_window), (fixed_expected, fixed_window) = plans
    assert mms_expected == [0.0] and fixed_expected == pytest.approx([span[0][1] / FPS])   # "raised" now counts
    assert mms_window.end == round(4.40 * FPS) and fixed_window.end == round((3.30 + 4.40) / 2 * FPS)


GROUP = "\ue000θɝtin\ue001"


def _norm(text):
    return tokens(text)


def test_a_pronunciation_group_is_one_target_word():
    """MMS normalize would upper-case the IPA and split at private-use markers or tie bars; a group
    is kept as one token of its own, its length its IPA symbol count."""
    from zonos3.tts.edit_align import expected_fill_seconds, target_normalize_request, token_chars

    target = f"i was {GROUP}, years old"
    texts, assemble = target_normalize_request(target)
    assert "\ue000" not in "".join(texts)
    normalized = assemble([_norm(t) for t in texts])
    assert normalized[0] == ["I", "WAS", GROUP, "YEARS", "OLD"] and normalized[3] == [GROUP]
    assert token_chars(GROUP) == pytest.approx(5 * 1.4)   # IPA symbols in letter-equivalents
    original = ["i", "was", "twelve", "years", "old"]
    spans = seconds_to_spans([[2.0, 3.0]], len(original) * FPS, FPS)
    (seconds,) = expected_fill_seconds(spans, clip(original), target.split(), normalized[1:], frame_rate=FPS)
    assert seconds == pytest.approx(5 * 1.4 / 6)   # "twelve" (6 letters) -> 5 IPA symbols = 7 letters
    plain_texts, plain_assemble = target_normalize_request("i was twelve")
    assert plain_texts == ["i was twelve", "i", "was", "twelve"]


def test_window_text_keeps_groups_as_typed():
    original = [f"w{k}" for k in range(30)]
    target = " ".join(original).replace("w20", GROUP)
    texts, assemble = target_normalize_request_tokens(target)
    words = clip(original)
    spans = seconds_to_spans([[20.0, 21.0]], 450, FPS)
    (window,) = plan_edit_windows(spans, words, target.split(), assemble[1:], total_frames=450, frame_rate=FPS,
                                  context_seconds=2.0, target_text=target)
    assert window.text == f"w18 w19 {GROUP} w21 w22"


def target_normalize_request_tokens(target):
    from zonos3.tts.edit_align import target_normalize_request

    texts, assemble = target_normalize_request(target)
    return texts, assemble([_norm(t) for t in texts])


@pytest.mark.parametrize("heard, errors, word", [
    (["I", "WAS", "THIRTEEN", "YEARS"], 0, "THIRTEEN"),
    (["I", "WAS", "YEARS"], 1, None),                     # the group was not said
    (["I", "WAS", "THIR", "TEEN", "YEARS"], 1, "TEEN"),   # one word matches, the other is inserted
])
def test_verify_counts_a_group_as_a_wildcard_word(heard, errors, word):
    from zonos3.tts.edit_align import fill_groups, wildcard_verify

    scored = wildcard_verify(["I", "WAS", GROUP, "YEARS"], heard)
    assert scored["word_errors"] == errors and scored["groups"] == [{"ipa": "θɝtin", "heard": word}]
    assert fill_groups(f"I was {GROUP} years.", [word]) == ("I was thirteen years." if word == "THIRTEEN"
                                                             else "I was years." if word is None else f"I was {word.lower()} years.")
