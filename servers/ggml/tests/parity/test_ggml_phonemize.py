"""``zonos3-cli --phonemize`` (src/zonos3-phonemes.cpp) against ``zonos3.tts.phonemes.PhonemeLexicon``.

A 20-word lexicon in PhonemeLexicon's LMDB layout (a ``meta`` JSON record in the main DB, one named DB per
language keyed by ``normalize_headword`` forms, "" = ambiguous) covers every status, mixed case, NFD input,
casefolding (ß), apostrophes and the Korean/Japanese/Chinese segmenters.
"""

from __future__ import annotations

import json
import os
import subprocess

import lmdb
import pytest
from zonos3.tts.phonemes import PhonemeLexicon, normalize_headword

ENTRIES = {
    ("eng", "words"): {
        "hello": "hɛloʊ", "world": "wɝld", "read": "", "don't": "doʊnt", "café": "kæfeɪ", "the": "ðə",
    },
    ("deu", "words"): {"strasse": "ʃtʁaːsə", "über": "yːbɐ"},
    ("fra", "words"): {"été": "ete", "garçon": "ɡaʁsɔ̃"},
    ("ita", "words"): {"ciao": "t͡ʃao"},
    ("rus", "words"): {"привет": "prʲɪvʲet"},
    ("jpn", "japanese"): {"日本": "nihoɴ", "東京": "toːkjoː", "カタカナ": "katakana"},
    ("cmn", "chinese"): {"你好": "ni˨˩˦xau˨˩˦", "中国": "ʈʂʊŋ˥kwo˧˥", "银行": ""},
    ("kor", "korean"): {"안녕하세요": "annjʌŋhasejo", "감사합니다": "kamsahamnida"},
}

CASES = [
    ("en_us", "Hello, WORLD! I read the book; don’t worry about the CAFE\u0301 or naïve co-op 日本."),
    ("en_gb", "  hello\tworld\n"),
    ("eng", "THE end's the end"),
    ("de", "Die Straße ist ÜBER alles, 2024."),
    ("fra", "L'été du garçon. Été!"),
    ("it", "Ciao, amici!"),
    ("rus", "Привет, мир!"),
    ("ja", "日本と東京へ行きます。日本語のカタカナ、食べる。"),
    ("cmn", "你好，中国银行很大。中国人"),
    ("ko", "안녕하세요 여러분, 감사합니다!"),
    ("en_us", ""),
    ("en_us", "... 123 !!!"),
]


@pytest.fixture(scope="module")
def lexicon_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("lexicon") / "lexicon.lmdb"
    env = lmdb.open(str(path), map_size=1 << 24, max_dbs=64, subdir=True)
    meta = {"version": 1, "languages": {code: {"mode": mode} for code, mode in ENTRIES}}
    with env.begin(write=True) as txn:
        txn.put(b"meta", json.dumps(meta).encode())
    count = 0
    for (code, _), words in ENTRIES.items():
        db = env.open_db(code.encode())
        with env.begin(write=True, db=db) as txn:
            for word, ipa in words.items():
                txn.put(normalize_headword(word).encode(), ipa.encode())
                count += 1
    env.close()
    assert count == 20
    return str(path)


@pytest.fixture(scope="module")
def reference(lexicon_dir):
    """One PhonemeLexicon: py-lmdb refuses to open an environment twice in one process."""
    return PhonemeLexicon(lexicon_dir)


def cli(ggml_build, lexicon_dir, language, text, *flags, env=None):
    exe = ggml_build / "zonos3-cli"
    if not exe.exists():
        pytest.fail(f"{exe} is not built")
    return subprocess.run([str(exe), "--phonemize", lexicon_dir, "--language", language, *flags, "--text", text],
                          capture_output=True, text=True, env=env)


def no_espeak_env():
    return {k: v for k, v in os.environ.items() if not k.startswith("ZONOS3_ESPEAK_NG_")}


@pytest.mark.parametrize("language,text", CASES)
def test_phonemize_matches_reference(ggml_build, lexicon_dir, reference, language, text):
    expected = reference.phonemize(text, language, guess=False)
    out = cli(ggml_build, lexicon_dir, language, text, "--no-guess", env=no_espeak_env())
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got == expected
    assert "".join(s["text"] for s in got["segments"]) == text


def test_statuses_covered(reference):
    statuses = {s["status"] for language, text in CASES
                for s in reference.phonemize(text, language, guess=False)["segments"]}
    assert statuses == {"ok", "ambiguous", "unknown", "skipped", None}


def test_unknown_language(ggml_build, lexicon_dir, reference):
    with pytest.raises(ValueError):
        reference.phonemize("hello", "xx_yy", guess=False)
    out = cli(ggml_build, lexicon_dir, "xx_yy", "hello", "--no-guess", env=no_espeak_env())
    assert out.returncode == 2


def test_unsupported_rewrite(ggml_build, lexicon_dir, reference, tmp_path):
    """api_server.tts_phonemize: a pronunciation with a symbol outside the phone inventory is "unsupported"."""
    symbols = ["h", "ɛ", "l", "o", "ʊ", "ð", "ə", "t͡ʃ"]
    text = "hello world, the read"
    expected = reference.phonemize(text, "en_us", guess=False)
    for segment in expected["segments"]:
        if segment["ipa"] and not set(symbols).issuperset(segment["ipa"]):
            segment.update(ipa=None, status="unsupported")
    path = tmp_path / "symbols.json"
    path.write_text(json.dumps(symbols))
    out = cli(ggml_build, lexicon_dir, "en_us", text, "--no-guess", "--symbols", str(path), env=no_espeak_env())
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got == expected
    assert {s["status"] for s in got["segments"]} >= {"ok", "unsupported", "ambiguous"}


def test_guess_off_without_espeak(ggml_build, lexicon_dir, reference):
    """No ZONOS3_ESPEAK_NG_LIBRARY: guessing is off even without --no-guess."""
    text = "Hello unknownword"
    out = cli(ggml_build, lexicon_dir, "en_us", text, env=no_espeak_env())
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == reference.phonemize(text, "en_us", guess=False)


GUESS_CASES = [
    ("en_us", "Hello, I read the strange water bottle; don't flutter about it."),
    ("de", "Die Straße ist über dem Rathaus, Kinder spielen."),
    ("fra", "Le garçon mange une pomme rouge."),
    ("it", "Ciao, la pizza e il caffè sono buoni."),
    ("rus", "Привет, хорошая жизнь и шапка."),
    ("ko", "안녕하세요 여러분"),
]


@pytest.mark.parametrize("language,text", GUESS_CASES)
def test_phonemize_guess_matches_reference(ggml_build, lexicon_dir, reference, language, text):
    espeakng_loader = pytest.importorskip("espeakng_loader")
    expected = reference.phonemize(text, language, guess=True)
    env = dict(os.environ, ZONOS3_ESPEAK_NG_LIBRARY=str(espeakng_loader.get_library_path()),
               ZONOS3_ESPEAK_NG_DATA=str(espeakng_loader.get_data_path()))
    out = cli(ggml_build, lexicon_dir, language, text, env=env)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == expected
    if language != "ko":
        assert any(s["status"] == "guessed" for s in expected["segments"])
