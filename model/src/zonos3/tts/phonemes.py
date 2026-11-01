"""Gold-lexicon phonemization for the ``(IPA)`` pronunciation input.

Phoneme-trained checkpoints (E2-TTS "X2" substitution, lingua
``apps/zonos/phonemes.py``) read a word written as ``(IPA)`` as its
pronunciation. This module looks words up in the same LMDB lexicon the training
data used (``apps/zonos/build_phoneme_lexicon.py``), so a phonemized word gets
exactly the notation the model saw. Words whose deciding source has several
pronunciations are reported as ambiguous. With the optional ``phonemize``
dependency group installed, espeak-ng guesses ambiguous and unknown words of the
space-delimited languages, mapped to the lexicon's notation.

The lexicon format is shared with lingua: a ``meta`` JSON record and one named
DB per language whose keys are ``normalize_headword`` forms and whose values are
the IPA ("" = ambiguous).
"""

from __future__ import annotations

import ctypes
import itertools
import json
import re
import threading
import unicodedata
from functools import lru_cache

import lmdb

LEXICON_VERSION = 1
_MAX_KEY_BYTES = 500  # LMDB key limit; the builder skips longer headwords
_CJK_MAX = 8  # longest headword tried by maximum matching
_JA_PARTICLES = frozenset("はがをにでとのもへ")  # hiragana that may follow a kanji word
# Server language codes (text normalization) -> lexicon language codes.
SERVER_LANGUAGES = {
    "en_us": "eng", "en_gb": "eng", "fr_fr": "fra", "de": "deu", "es": "spa",
    "it": "ita", "pt_br": "por", "ja": "jpn", "cmn": "cmn", "ko": "kor",
}
# Lexicon languages espeak-ng can guess. Chinese, Japanese, Korean and Vietnamese
# are absent: espeak-ng writes their tones as ASCII digits or misreads the script.
ESPEAK_VOICES = {
    "eng": "en-us", "deu": "de", "fra": "fr", "spa": "es-419", "ita": "it", "por": "pt-br",
    "nld": "nl", "pol": "pl", "ces": "cs", "swe": "sv", "fin": "fi", "dan": "da", "nob": "nb",
    "isl": "is", "hun": "hu", "ron": "ro", "cat": "ca", "tur": "tr", "ind": "id", "msa": "ms",
    "rus": "ru", "ukr": "uk", "ell": "el", "heb": "he", "ara": "ar", "fas": "fa", "hin": "hi",
}


def normalize_headword(word: str) -> str:
    """Lexicon key of a word; must match lingua ``apps.zonos.phonemes.normalize_headword``."""
    return unicodedata.normalize("NFC", word.strip()).replace("’", "'").casefold()


@lru_cache(maxsize=None)
def _char_class(c: str) -> str:
    if not unicodedata.category(c).startswith("L"):
        return ""
    name = unicodedata.name(c, "")
    if name.startswith("HIRAGANA"):
        return "hira"
    if name.startswith("KATAKANA"):
        return "kata"
    if name.startswith("HANGUL SYLLABLE"):
        return "hangul"
    if "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name or c == "々":
        return "han"
    return "letter"


def _status(value: str | None) -> str:
    return "unknown" if value is None else "ambiguous" if value == "" else "ok"


class PhonemeLexicon:
    def __init__(self, path: str):
        self._env = lmdb.open(path, readonly=True, lock=False, readahead=False, max_dbs=64, subdir=True)
        with self._env.begin() as txn:
            raw = txn.get(b"meta")
        if raw is None:
            raise ValueError(f"{path} is not a phoneme lexicon (no meta record)")
        meta = json.loads(raw)
        if meta.get("version") != LEXICON_VERSION:
            raise ValueError(f"Unsupported phoneme lexicon version {meta.get('version')!r}")
        self.languages: dict[str, dict] = meta["languages"]
        self._dbs = {code: self._env.open_db(code.encode(), create=False) for code in self.languages}
        self._ties: dict[str, bool] = {}

    def resolve_language(self, language: str) -> str:
        """Lexicon code for a server language code or a lexicon code."""
        code = SERVER_LANGUAGES.get(language, language)
        if code not in self.languages:
            raise ValueError(f"No pronunciation lexicon for language {language!r}")
        return code

    @property
    def server_languages(self) -> list[str]:
        return [lang for lang, code in SERVER_LANGUAGES.items() if code in self.languages]

    def get(self, code: str, word: str) -> str | None:
        """None: unknown; "": ambiguous; otherwise the gold IPA."""
        key = normalize_headword(word).encode()
        if not key or len(key) > _MAX_KEY_BYTES:
            return None
        with self._env.begin(db=self._dbs[code]) as txn:
            value = txn.get(key)
        return None if value is None else value.decode()

    @property
    def guess_languages(self) -> list[str]:
        """Server languages whose unknown words espeak-ng can guess."""
        espeak = _espeak()
        if espeak is None:
            return []
        return [lang for lang, code in SERVER_LANGUAGES.items() if code in self.languages and code in espeak.voices]

    def uses_ties(self, code: str) -> bool:
        """Whether the language's gold IPA writes affricates with a tie bar (t͡s vs ts)."""
        if code not in self._ties:
            tied = untied = 0
            with self._env.begin(db=self._dbs[code]) as txn:
                for _, value in itertools.islice(txn.cursor(), 20000):
                    value = value.decode()
                    tied += value.count("\u0361")
                    untied += len(_AFFRICATE.findall(value))
            self._ties[code] = tied > untied
        return self._ties[code]

    def phonemize(self, text: str, language: str, guess: bool = True) -> dict:
        """Split ``text`` into segments; word segments carry their lexicon result.

        Segment texts concatenate back to ``text``. A word segment has ``status``
        "ok" (with ``ipa``), "ambiguous" (several pronunciations), "unknown" (not in
        the lexicon) or, for Japanese/Chinese characters outside any multi-character
        word, "skipped"; other text has ``status`` None. With ``guess``, ambiguous
        and unknown words of an espeak-ng language get its pronunciation instead,
        with ``status`` "guessed".
        """
        code = self.resolve_language(language)
        mode = self.languages[code]["mode"]
        espeak = _espeak() if guess else None
        if espeak is not None and code not in espeak.voices:
            espeak = None
        if mode == "words":
            spans = []
            for a, b in self._word_spans(text):
                value = self.get(code, text[a:b])
                guessed = None
                if not value and espeak is not None:
                    guessed = harmonize_espeak(espeak.ipa(code, text[a:b]), code, self.uses_ties(code))
                if guessed:
                    spans.append((a, b, guessed, "guessed"))
                else:
                    spans.append((a, b, value or None, _status(value)))
        else:
            spans = self._cjk_spans(text, code, mode)
        segments, pos = [], 0
        for a, b, ipa, status in spans:
            if a > pos:
                segments.append(dict(text=text[pos:a], ipa=None, status=None))
            segments.append(dict(text=text[a:b], ipa=ipa, status=status))
            pos = b
        if pos < len(text):
            segments.append(dict(text=text[pos:], ipa=None, status=None))
        return dict(language=code, segments=segments)

    @staticmethod
    def _word_spans(text: str) -> list[tuple[int, int]]:
        """Runs that start at a letter and continue over letters, marks and inner apostrophes."""
        spans, i, n = [], 0, len(text)
        while i < n:
            if not _char_class(text[i]):
                i += 1
                continue
            j = i + 1
            while j < n:
                if unicodedata.category(text[j])[0] in "LM":
                    j += 1
                elif text[j] in "'’" and j + 1 < n and _char_class(text[j + 1]):
                    j += 2
                else:
                    break
            spans.append((i, j))
            i = j
        return spans

    def _cjk_spans(self, text: str, code: str, mode: str) -> list[tuple[int, int, str | None, str]]:
        """Training-compatible units: Korean eojeol; Japanese/Chinese headwords of 2+ characters.

        Hiragana and single kanji/hanzi are never units (particles and polyphonic
        characters have no single reading). A Japanese kanji run followed by
        okurigana (hiragana other than a particle) is skipped whole, since its
        reading belongs to the inflected word. Both rules match the training data.
        """
        if mode == "korean":
            targets = {"hangul"}
        elif mode == "japanese":
            targets = {"han", "kata"}
        else:
            targets = {"han"}
        spans, i, n = [], 0, len(text)
        while i < n:
            cls = _char_class(text[i])
            if cls not in targets:
                i += 1
                continue
            j = i + 1
            while j < n and _char_class(text[j]) == cls:
                j += 1
            if mode == "korean":
                value = self.get(code, text[i:j])
                spans.append((i, j, value or None, _status(value)))
            elif cls == "han" and mode == "japanese" and j < n and _char_class(text[j]) == "hira" \
                    and text[j] not in _JA_PARTICLES:
                spans.append((i, j, None, "skipped"))
            else:
                spans.extend(self._max_match(text, i, j, code))
            i = j
        return spans

    def _max_match(self, text: str, start: int, end: int, code: str) -> list[tuple[int, int, str | None, str]]:
        """Forward maximum matching of 2+ character headwords inside one run.

        Characters no such headword covers are grouped into "skipped" spans.
        """
        spans, i, pending = [], start, None
        while i < end:
            for length in range(min(_CJK_MAX, end - i), 1, -1):
                value = self.get(code, text[i:i + length])
                if value is not None:
                    if pending is not None:
                        spans.append((pending, i, None, "skipped"))
                        pending = None
                    spans.append((i, i + length, value or None, _status(value)))
                    i += length
                    break
            else:
                pending = i if pending is None else pending
                i += 1
        if pending is not None:
            spans.append((pending, end, None, "skipped"))
        return spans


# espeak-ng symbols -> the lexicon's notation where the two systematically differ.
_ESPEAK_SYMBOLS = {
    # CMUdict-derived English: no length marks, rhotic ɝ/ɚ, no flaps or glottal stops.
    "eng": (
        ("ɜː", "ɝ"), ("ɜ", "ɝ"), ("oːɹ", "ɔɹ"), ("ː", ""), ("ɐ", "ə"), ("ᵻ", "ɪ"), ("ɾ", "t"),
        ("ʔ", "t"), ("n̩", "ən"), ("l̩", "əl"), ("m̩", "əm"), ("ɒ", "ɑ"), ("x", "k"),
    ),
    # WikiPron German: uvular ʁ, vocalized -er as ɐ, front long a, marked diphthongs.
    "deu": (("ɾ", "ʁ"), ("r", "ʁ"), ("ɜ", "ɐ"), ("ɑː", "aː"), ("aʊ", "aʊ̯"), ("aɪ", "aɪ̯"), ("ɔø", "ɔɪ̯")),
    # WikiPron Italian: geminates spelled double, no lax final vowels or taps.
    "ita": (
        ("tʃː", "ttʃ"), ("dʒː", "ddʒ"), ("tsː", "tts"), ("dzː", "ddz"),
        *((c + "ː", c + c) for c in "bdfklmnprstvzʃʎɲɡ"),
        ("ɪ", "i"), ("ʊ", "u"), ("ɾ", "r"),
    ),
    # WikiPron Russian: retroflex ш/ж, ɨ for ы, plain l with ʲ, a/ə vowels.
    "rus": (("ɭ", "l"), ("ʃ", "ʂ"), ("ʒ", "ʐ"), ("y", "ɨ"), ("ɑ", "a"), ("ʌ", "ə")),
}
_ESPEAK_DROP = str.maketrans("", "", " ˈˌ.‿'-_")
_AFFRICATE = re.compile(r"([td])([sʃzʒɕʑʂʐ])|(p)(f)")


def harmonize_espeak(raw: str, code: str, ties: bool) -> str | None:
    """espeak-ng IPA in the lexicon's notation (the training ``canonical_ipa`` rules).

    Returns None for output the model never saw in parentheses: tones written as
    ASCII digits, or leftover brackets.
    """
    s = unicodedata.normalize("NFC", re.sub(r"\([a-z-]+\)", "", raw))  # (en) voice switches
    s = s.translate(_ESPEAK_DROP).replace(":", "ː").replace("g", "ɡ")
    for a, b in _ESPEAK_SYMBOLS.get(code, ()):
        s = s.replace(a, b)
    if ties:
        s = _AFFRICATE.sub(lambda m: "\u0361".join(g for g in m.groups() if g), s)
    if not s or any(c in "()/[]" or "0" <= c <= "9" for c in s):
        return None
    return s


class _Espeak:
    """espeak-ng through ctypes (``espeakng-loader`` ships the library and its data)."""

    def __init__(self, library: str, data: str):
        self._lib = ctypes.cdll.LoadLibrary(library)
        self._lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self._lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        self._lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
        self._lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        if self._lib.espeak_Initialize(2, 0, data.encode(), 0) <= 0:  # AUDIO_OUTPUT_SYNCHRONOUS
            raise RuntimeError(f"espeak-ng failed to initialize from {data}")
        self._lock = threading.Lock()
        self._voice = None
        self.voices = {code: voice for code, voice in ESPEAK_VOICES.items() if self._select(voice)}

    def _select(self, voice: str) -> bool:
        ok = self._lib.espeak_SetVoiceByName(voice.encode()) == 0
        self._voice = voice if ok else None
        return ok

    def ipa(self, code: str, word: str) -> str:
        """Raw espeak-ng IPA of one word."""
        with self._lock:
            if self._voice != self.voices[code] and not self._select(self.voices[code]):
                raise RuntimeError(f"espeak-ng voice {self.voices[code]} vanished")
            text = ctypes.c_char_p(word.encode())
            parts = []
            while text.value:
                phonemes = self._lib.espeak_TextToPhonemes(ctypes.byref(text), 1, 0x02)  # UTF-8 in, IPA out
                if phonemes:
                    parts.append(phonemes.decode())
            return "".join(parts)


@lru_cache(maxsize=1)
def _espeak() -> _Espeak | None:
    """The optional guesser; None when the ``phonemize`` dependency group is not installed."""
    try:
        import espeakng_loader
    except ImportError:
        return None
    return _Espeak(espeakng_loader.get_library_path(), espeakng_loader.get_data_path())
