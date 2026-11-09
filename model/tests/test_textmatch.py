"""zonos3.tts.textmatch: the edit tokens and error rates equal z-speech-alignment's (fixtures generated once from its
``z_speech_alignment.textmatch`` and the edit's former recognition worker; no runtime dependency on it)."""

import json
from pathlib import Path

import pytest
from zonos3.tts import textmatch

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "textmatch_zsa.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("text, expected", FIXTURES["normalize"])
def test_normalize_matches_zsa(text, expected):
    assert textmatch.normalize(text) == expected


@pytest.mark.parametrize("reference, hypothesis, expected", FIXTURES["char_error_rate"])
def test_char_error_rate_matches_zsa(reference, hypothesis, expected):
    assert textmatch.char_error_rate(reference, hypothesis) == expected


@pytest.mark.parametrize("reference, hypothesis, expected", FIXTURES["word_errors"])
def test_word_errors_match_the_former_worker(reference, hypothesis, expected):
    assert textmatch.word_errors(reference, hypothesis) == expected


def test_verify_reports_word_errors_reference_words_and_cer():
    assert textmatch.verify("I was twelve years old.", "i was thirteen years old") == {
        "word_errors": 1,
        "ref_words": 5,
        "cer": pytest.approx(textmatch.char_error_rate("I was twelve years old.", "i was thirteen years old")),
    }
    assert textmatch.verify("", "anything") == {"word_errors": 1, "ref_words": 0, "cer": 0.0}
