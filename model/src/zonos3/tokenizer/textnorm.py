"""TTS text normalization using the vendored NeMo forward-TN grammars.

Converts written text to its spoken form ("$5.32" -> "five dollars thirty two
cents") before tokenization. Grammar construction is expensive (~10-30s per
language) the first time; compiled WFSTs are cached as .far files under
ZONOS3_TTS_NORM_CACHE_DIR (default ~/.cache/zonos3-tts-norm) and reload in
well under a second. Use tools/build_tts_norm_fars.py to prewarm all caches.

Disable globally with ZONOS3_TTS_NORM=0, or per request via the
text_normalization flag.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Dict, Optional

from zonos3 import languages
from zonos3.tts.markup import has_nonverbal_markup, split_nonverbal_markup

logger = logging.getLogger(__name__)

# Server language codes -> NeMo text_normalization language packages.
SERVER_TO_NEMO_LANG: Dict[str, str] = {lang.code: lang.nemo for lang in languages.SERVER}

# Upstream's own tests run Korean grammars with lower_cased input; everything
# else uses cased.
LOWER_CASED_LANGS = {"ko"}

# zh/ja verbalizers read a cached .far but never write one (upstream quirk);
# we post-write it after first compile so later loads are fast. Note ja reads
# a "jp_" prefixed name.
_VERBALIZER_FAR_PREFIX = {"zh": "zh", "ja": "jp"}

# A digit directly followed by sentence punctuation confuses several upstream
# r1.2.0 grammars: pt's tagger raises FstOpError outright and de reads dates
# digit-by-digit (reproduced with upstream code and its pinned pynini
# 2.1.6.post1 — upstream bugs, not vendoring artifacts). Space the punctuation
# off before normalization; it is re-attached afterwards.
_DIGIT_PUNCT_RE = re.compile(r"(\d)([.!?,;:])(?=\s|$)")
_SPACE_PUNCT_RE = re.compile(r" +([.!?,;:])(?=\s|$)")

# Moses-based punct_post_process re-attaches punctuation well for these
# languages. For the European languages it also glues currency symbols to the
# following word ("5,32 € am" -> "€am"), so there we skip moses and only
# collapse the spacing we introduced ourselves.
_MOSES_POSTPROCESS_LANGS = {"en", "zh", "ja", "ko"}

# Line breaks. NeMo's tagger treats "\n" as whitespace, but post_process_punct (the Moses path
# above) re-matches the spacing around every ASCII punctuation mark to the ORIGINAL input and only knows " ": a period
# followed by a newline came back glued to the next word ("spend less.\n\nHunt" -> "spend less.Hunt", a bare "." token
# that the model speaks as "dot"), and after a digit the re-attach below failed too ("$5.\nNext" -> "five dollars .Next").
# Any whitespace run that contains a line break, tab or other non-space whitespace becomes ONE plain space before
# normalization (the comparison input must already carry the space); runs of plain spaces are untouched, so single-line
# text is byte-identical. Training transcripts were single-line, so this is also the in-distribution shape for the
# tokenizer when normalization is off (a raw "\n\n" tokenizes to a never-seen ".ĊĊ" token).
_LINE_BREAK_RE = re.compile(r"\s*[^\S ]\s*")


def collapse_line_breaks(text: str) -> str:
    """One plain space for every whitespace run that contains a line break, tab or non-breaking space."""
    return _LINE_BREAK_RE.sub(" ", text) if text else text


# Pronunciation groups of phoneme-token checkpoints (zonos3.tts.superbpe): never normalized.
_PHONEME_OPEN = "\ue000"
_PHONEME_GROUP_RE = re.compile("(\ue000[^\ue000\ue001]*\ue001)")


# Per-language pre-passes. Each one works around a reproduced upstream r1.2.0 grammar
# behaviour; single-line text in every other language is byte-identical to before.
#
# ja / zh: the ja grammar raises FstOpError on an ASCII space after sentence punctuation ("です。 次" -> the whole request
# used to come back raw), and neither language writes spaces between its own characters, so whitespace between two CJK
# characters (incl. CJK punctuation) is removed; spaces next to Latin letters or digits ("3.5 km") are kept.
_CJK = "　-〿぀-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿＀-￯"
_CJK_SPACE_RE = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])")
_CJK_LANGS = {"ja", "zh"}

# de: "3." is both the ordinal "dritte" and a sentence-final cardinal. The generic digit rule turned every mid-sentence
# ordinal and day of month into a cardinal ("am 3. Oktober" -> "am drei. Oktober"), while leaving the period glued reads a
# sentence-final year as an ordinal ("Seit 1990." -> "neunzigste"). Heuristic: space the period off only at the end of
# the text (optionally before closing quotes / brackets) or after a number of three or more digits (years, counts);
# other punctuation after digits as in the generic rule. Known residue: "Platz 3. Dann" -> "dritte Dann".
_DE_OTHER_PUNCT_RE = re.compile(r"(\d)([!?,;:])(?=\s|$)")
_DE_FINAL_PERIOD_RE = re.compile(r"(\d)(\.)(?=[\"'»“”)\]]*\s*$)")
_DE_LONG_PERIOD_RE = re.compile(r"(\d{3,})(\.)(?=\s|$)")

# pt: the grammar raises FstOpError whenever a token carrying a digit is directly followed by punctuation or a closing
# bracket ("10%;", "10 %;", "1º,", "9h.", "(de 10)."), or an opening bracket precedes one; the generic rule only covered
# a bare digit before the mark. Tokens carrying a digit, "%", "º" or "ª" count. Every such mark is spaced off and re-attached afterwards.
_PT_TRAIL_PUNCT_RE = re.compile(r"(\S*[\d%ºª]\S*?)([.!?,;:)\]]+)(?=\s|$)")
_PT_LEAD_BRACKET_RE = re.compile(r"(^|\s)([(\[])(?=\S*\d)")
_PT_SPACE_TRAIL_RE = re.compile(r" +([.!?,;:)\]]+)(?=\s|$)")
_PT_SPACE_LEAD_RE = re.compile(r"([(\[]) +")

# en: spoken-form conventions the fraction / money grammars get wrong: "24/7" -> "twenty four sevenths", "50/50" ->
# "fifty fiftieths", "9/11" -> "nine elevenths", "$20k" -> "dollar twenty k". Rewritten before normalization.
_EN_SLASH_PAIRS = {("24", "7"), ("9", "11"), ("7", "11")}
_EN_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
_EN_MONEY_K_RE = re.compile(r"([$£€])(\d+(?:\.\d+)?)[kK]\b")

# Sentence boundaries for the per-sentence retry after a grammar failure.
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+|(?<=[。！？])")


def _en_slash(m: "re.Match[str]") -> str:
    a, b = m.group(1), m.group(2)
    if (a, b) in _EN_SLASH_PAIRS or (len(a) == 2 and len(b) == 2 and int(a) + int(b) == 100):
        return f"{a} {b}"
    return m.group(0)


def _en_money_k(m: "re.Match[str]") -> str:
    return f"{m.group(1)}{int(round(float(m.group(2)) * 1000)):,}"


def pre_normalize(text: str, lang: str) -> str:
    """The language-specific rewrite applied right before the NeMo grammar (see the notes above)."""
    if lang in _CJK_LANGS:
        return _CJK_SPACE_RE.sub("", text)
    if lang == "de":
        text = _DE_OTHER_PUNCT_RE.sub(r"\1 \2", text)
        text = _DE_FINAL_PERIOD_RE.sub(r"\1 \2", text)
        return _DE_LONG_PERIOD_RE.sub(r"\1 \2", text)
    if lang == "pt":
        text = _PT_TRAIL_PUNCT_RE.sub(r"\1 \2", text)
        return _PT_LEAD_BRACKET_RE.sub(r"\1\2 ", text)
    text = _DIGIT_PUNCT_RE.sub(r"\1 \2", text)
    if lang == "en":
        text = _EN_SLASH_RE.sub(_en_slash, text)
        text = _EN_MONEY_K_RE.sub(_en_money_k, text)
    return text


def post_normalize(result: str, lang: str) -> str:
    """Re-attach the punctuation pre_normalize() spaced off."""
    if lang == "pt":
        result = _PT_SPACE_TRAIL_RE.sub(r"\1", result)
        return _PT_SPACE_LEAD_RE.sub(r"\1", result)
    return _SPACE_PUNCT_RE.sub(r"\1", result)


def default_cache_root() -> str:
    return os.environ.get(
        "ZONOS3_TTS_NORM_CACHE_DIR",
        os.path.expanduser("~/.cache/zonos3-tts-norm"),
    )


def normalization_enabled() -> bool:
    return os.environ.get("ZONOS3_TTS_NORM", "1") != "0"


class TTSTextNormalizer:
    """Lazy per-language NeMo normalizers with .far caching.

    Construction and calls are serialized per language: the upstream
    Normalizer shares a mutable TokenParser and is not thread-safe.
    """

    def __init__(self, cache_root: str | None = None):
        self.cache_root = cache_root or default_cache_root()
        self._normalizers: Dict[str, object] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    @staticmethod
    def nemo_lang(language: str) -> Optional[str]:
        return SERVER_TO_NEMO_LANG.get(language)

    def supported(self, language: str) -> bool:
        return language in SERVER_TO_NEMO_LANG

    def _lang_lock(self, lang: str) -> threading.Lock:
        with self._global_lock:
            if lang not in self._locks:
                self._locks[lang] = threading.Lock()
            return self._locks[lang]

    def _build(self, lang: str):
        from zonos3.vendor.nemo_text_processing.text_normalization import (
            Normalizer,
        )

        input_case = "lower_cased" if lang in LOWER_CASED_LANGS else "cased"
        # One cache dir per (lang, case): upstream .far filenames collide
        # across languages (e.g. ja's tagger writes a zh_-prefixed file).
        cache_dir = os.path.join(self.cache_root, f"{lang}_{input_case}")
        os.makedirs(cache_dir, exist_ok=True)

        logger.info("Loading TTS text normalizer for '%s' (%s)...", lang, input_case)
        normalizer = Normalizer(
            input_case=input_case,
            lang=lang,
            cache_dir=cache_dir,
            overwrite_cache=False,
        )

        prefix = _VERBALIZER_FAR_PREFIX.get(lang)
        if prefix is not None:
            far_path = os.path.join(cache_dir, f"{prefix}_tn_True_deterministic_verbalizer.far")
            if not os.path.exists(far_path):
                from zonos3.vendor.nemo_text_processing.text_normalization.en.graph_utils import (
                    generator_main,
                )

                generator_main(far_path, {"verbalize": normalizer.verbalizer.fst})
        return normalizer

    def get(self, lang: str):
        with self._lang_lock(lang):
            if lang not in self._normalizers:
                self._normalizers[lang] = self._build(lang)
            return self._normalizers[lang]

    def warmup(self, languages: list[str] | None = None) -> None:
        """Construct normalizers ahead of time (server codes or NeMo codes)."""
        langs = languages or sorted(set(SERVER_TO_NEMO_LANG.values()))
        for lang in langs:
            lang = SERVER_TO_NEMO_LANG.get(lang, lang)
            try:
                self.get(lang)
            except Exception:  # noqa: BLE001
                logger.exception("TTS text normalizer warmup failed for '%s'", lang)

    def _normalize_once(self, text: str, lang: str) -> str:
        normalizer = self.get(lang)
        text_in = pre_normalize(text, lang)
        with self._lang_lock(lang):
            result = normalizer.normalize(text_in, punct_post_process=lang in _MOSES_POSTPROCESS_LANGS)
        if not isinstance(result, str) or not result.strip():
            raise ValueError("empty normalizer output")
        return post_normalize(result, lang)

    def normalize(self, text: str, language: str) -> str:
        """Normalize text for the given server language code.

        Returns the input (line breaks collapsed) for unsupported languages. A grammar failure on the whole text
        (upstream FstOpError on some span) is retried sentence by sentence so only the failing sentence stays as
        typed — normalization must never fail a request. Pronunciation groups (U+E000 IPA U+E001, phoneme-token
        checkpoints) are kept as typed; the text between them is normalized stretch by stretch.

        Nonverbal markup ("<laugh/>", "<shouting>...</shouting>") is never handed to a grammar either (the source
        NV runtime's rule): the text between the markers is normalized piece by piece (each piece
        with its pronunciation groups as above) and re-joined around the unchanged markers, keeping a single space on
        each side of a marker where the input had whitespace (the ja grammar raises on an ASCII space next to a
        marker, which left the whole sentence unnormalized). Text without markup takes the unchanged path.
        """
        text = collapse_line_breaks(text)  # every return path hands back single-line text
        if SERVER_TO_NEMO_LANG.get(language) is None or not has_nonverbal_markup(text):
            return self._normalize_groups(text, language)
        out = []
        for index, piece in enumerate(split_nonverbal_markup(text)):
            core = piece.strip()
            if index % 2 or not core:
                out.append(piece if index % 2 else (" " if piece else ""))
                continue
            lead = " " if piece[0].isspace() else ""
            trail = " " if piece[-1].isspace() else ""
            out.append(lead + self._normalize_groups(core, language) + trail)
        result = "".join(out).strip()
        logger.debug("TTS norm [%s] around nonverbal markup: %r -> %r", language, text, result)
        return result

    def _normalize_groups(self, text: str, language: str) -> str:
        """Normalize markup-free single-line text, pronunciation groups kept as typed (see normalize)."""
        if _PHONEME_OPEN in text:
            pieces = _PHONEME_GROUP_RE.split(text)
            for i in range(0, len(pieces), 2):
                stretch = pieces[i]
                if stretch.strip():
                    lead, trail = stretch[: len(stretch) - len(stretch.lstrip())], stretch[len(stretch.rstrip()) :]
                    pieces[i] = lead + self._normalize_text(stretch.strip(), language) + trail
            return "".join(pieces)
        return self._normalize_text(text, language)

    def _normalize_text(self, text: str, language: str) -> str:
        lang = SERVER_TO_NEMO_LANG.get(language)
        text = collapse_line_breaks(text)  # every return path below hands back single-line text
        if lang is None or not text.strip():
            return text
        try:
            result = self._normalize_once(text, lang)
        except Exception:  # noqa: BLE001
            logger.exception("TTS text normalization failed for lang=%s; retrying sentence by sentence", language)
            pieces = []
            for piece in _SENTENCE_RE.split(text):
                if not piece or not piece.strip():
                    continue
                try:
                    pieces.append(self._normalize_once(piece, lang))
                except Exception:  # noqa: BLE001
                    logger.warning("TTS text normalization failed for lang=%s on %r; kept as typed", language, piece)
                    pieces.append(piece)
            result = ("" if lang in _CJK_LANGS else " ").join(pieces)
        logger.debug("TTS norm [%s]: %r -> %r", language, text, result)
        return result


def load_normalizer(*, warmup: bool) -> TTSTextNormalizer | None:
    """The server's normalizer (None with ZONOS3_TTS_NORM=0); ``warmup`` compiles the English grammars on a daemon
    thread so the first request does not pay the one-time WFST construction."""
    if not normalization_enabled():
        return None
    normalizer = TTSTextNormalizer()
    if warmup:
        threading.Thread(target=normalizer.warmup, args=(["en"],), daemon=True).start()
    return normalizer


def normalize_request_text(text: str, language: str, normalizer) -> str:
    """A request's text as the prompt builder gets it: line breaks collapsed (also when normalization is off), then
    ``normalizer.normalize`` (anything with that method; None skips it)."""
    text = collapse_line_breaks(text)
    return text if normalizer is None else normalizer.normalize(text, language)
