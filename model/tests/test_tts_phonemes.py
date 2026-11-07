from __future__ import annotations

import json

import lmdb
import pytest
from zonos3.tts.phonemes import PhonemeLexicon, harmonize_espeak

ENTRIES = {
    "eng": {"i": "aɪ", "met": "mɛt", "don't": "doʊnt", "read": "", "cat": "kæt"},
    "jpn": {"大事": "daiʑi", "美味": "bimi", "絵": "e", "スポット": "sɯpotːo", "は": "ha"},
    "cmn": {"如果": "ʐu³⁵kuɔ²¹⁴", "我们": "wo²¹⁴mən", "走": "tsoʊ²¹⁴"},
    "kor": {"그": "kɯ", "막": "mak̚"},
}
MODES = {"eng": "words", "jpn": "japanese", "cmn": "chinese", "kor": "korean"}


@pytest.fixture(scope="module")
def lexicon(tmp_path_factory):
    path = tmp_path_factory.mktemp("lexicon")
    env = lmdb.open(str(path), map_size=1 << 24, max_dbs=8)
    for code, entries in ENTRIES.items():
        db = env.open_db(code.encode())
        with env.begin(write=True, db=db) as txn:
            for key, value in entries.items():
                txn.put(key.encode(), value.encode())
    languages = {code: {"mode": MODES[code], "script": None, "words": 0, "gold": 0, "sources": []} for code in ENTRIES}
    with env.begin(write=True) as txn:
        txn.put(b"meta", json.dumps({"version": 1, "languages": languages, "sources": []}).encode())
    env.close()
    return PhonemeLexicon(str(path))


def words(result):
    return [(s["text"], s["ipa"] or s["status"]) for s in result["segments"] if s["status"]]


@pytest.mark.parametrize(
    "text, language",
    [
        ("I met the cat, don’t I?", "en_us"),
        ("大事にスポット絵を見た。美味しい", "ja"),
        ("如果我们走", "cmn"),
        ("그 막 학교에", "ko"),
    ],
)
def test_segments_reassemble_the_text(lexicon, text, language):
    assert "".join(s["text"] for s in lexicon.phonemize(text, language)["segments"]) == text


def test_words_report_gold_ambiguous_and_unknown(lexicon):
    result = lexicon.phonemize("I read the CAT, don’t I?", "en_gb", guess=False)
    assert result["language"] == "eng"
    assert words(result) == [
        ("I", "aɪ"),
        ("read", "ambiguous"),
        ("the", "unknown"),
        ("CAT", "kæt"),
        ("don’t", "doʊnt"),
        ("I", "aɪ"),
    ]


def test_japanese_skips_kana_single_kanji_and_okurigana(lexicon):
    # は is hiragana, 絵/見 are single kanji, 美味 is followed by okurigana (美味しい).
    result = lexicon.phonemize("大事はスポット絵を見た美味しい", "ja")
    assert words(result) == [
        ("大事", "daiʑi"),
        ("スポット", "sɯpotːo"),
        ("絵", "skipped"),
        ("見", "skipped"),
        ("美味", "skipped"),
    ]


def test_chinese_matches_multi_character_words_only(lexicon):
    assert words(lexicon.phonemize("如果我们走", "cmn")) == [
        ("如果", "ʐu³⁵kuɔ²¹⁴"),
        ("我们", "wo²¹⁴mən"),
        ("走", "skipped"),
    ]


def test_korean_looks_up_whole_eojeol(lexicon):
    assert words(lexicon.phonemize("그 막 학교에", "ko")) == [("그", "kɯ"), ("막", "mak̚"), ("학교에", "unknown")]


def test_language_without_lexicon_is_rejected(lexicon):
    assert lexicon.server_languages == ["en_us", "en_gb", "ja", "cmn", "ko"]
    with pytest.raises(ValueError):
        lexicon.phonemize("Bonjour", "fr_fr")


@pytest.mark.parametrize(
    "raw, code, ties, expected",
    [
        ("bˈʌʔn̩", "eng", False, "bʌtən"),  # glottal stop and syllabic n in CMUdict notation
        ("bˈɜːd", "eng", False, "bɝd"),
        ("(en)ɐbˈaʊt(fr)", "eng", False, "əbaʊt"),  # voice-switch markers dropped
        ("tsˈuːk", "deu", True, "t͡suːk"),
        ("fatʃːˈamo", "ita", True, "fatt͡ʃamo"),  # geminates spelled double, tied affricate
        ("tʃˈaː2w", "vie", False, None),  # ASCII tone digits never appear in training IPA
    ],
)
def test_espeak_output_is_mapped_to_lexicon_notation(raw, code, ties, expected):
    assert harmonize_espeak(raw, code, ties) == expected


def test_espeak_guesses_ambiguous_and_unknown_words_only(lexicon):
    pytest.importorskip("espeakng_loader")
    assert lexicon.guess_languages == ["en_us", "en_gb"]
    result = words(lexicon.phonemize("I met Siobhan, read on.", "en_us"))
    assert [(text, ipa) for text, ipa in result if text in ("I", "met", "Siobhan")] == [
        ("I", "aɪ"),
        ("met", "mɛt"),
        ("Siobhan", "ʃɪvɔn"),
    ]
    statuses = {s["text"]: s["status"] for s in lexicon.phonemize("I met Siobhan, read on.", "en_us")["segments"]}
    assert statuses["Siobhan"] == statuses["read"] == "guessed" and statuses["met"] == "ok"
    assert words(lexicon.phonemize("如果我们走", "cmn"))[-1] == ("走", "skipped")
