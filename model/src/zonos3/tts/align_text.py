"""The tokens a transcript is force-aligned by (FuseAlign: zonos3/workers/fusealign.py places one interval per token).

FuseAlign was trained on transcripts whose unspaced scripts were segmented into words (z-speech-alignment's wordseg:
jieba for Mandarin and Cantonese, MeCab/UniDic for Japanese, each over the maximal runs of Han and kana). Here those
runs are cut by ICU's word break (root locale): the same dictionaries as the browser's ``Intl.Segmenter``, so the
in-browser engine and the studio cut CJK text exactly as the servers do. ICU and jieba / MeCab agree on most words
(東京タワー is one ICU word, two MeCab morphemes)."""

from __future__ import annotations

import re
import unicodedata

from zonos3.languages import UNSPACED

# z-speech-alignment wordseg's _HAN and _KANA: the runs a word segmenter cuts.
_CJK_RUN = re.compile("[\u3005\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]+")
_TOKEN = re.compile(r"\S+")
# ICU rule statuses at or above UBRK_WORD_NUMBER (100): the segment is a word (Intl.Segmenter isWordLike).
_ICU_WORD = 100


def _wordy(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LN" or ch == "'"


def cjk_words(run: str) -> list[tuple[int, int]]:
    """``[start, end)`` of the words ICU's word break (root locale) finds in a run of Han / kana (``_CJK_RUN``);
    segments that are not words (``・``) are left out."""
    import icu  # pyicu-wheels: loaded on the first CJK text

    breaks = icu.BreakIterator.createWordInstance(icu.Locale.getRoot())
    breaks.setText(run)  # BMP-only run: ICU's UTF-16 offsets are str offsets
    words, start = [], breaks.first()
    for end in breaks:
        if breaks.getRuleStatus() >= _ICU_WORD:
            words.append((start, end))
        start = end
    return words


def split_cjk(word: str) -> list[str]:
    """A whitespace-free string cut into the words of its Han / kana runs (``cjk_words``) and the text between them;
    a string without such runs is returned whole."""
    out, pos = [], 0
    for m in _CJK_RUN.finditer(word):
        if m.start() > pos:
            out.append(word[pos : m.start()])
        out += [m.group()[a:b] for a, b in cjk_words(m.group())]
        pos = m.end()
    if pos < len(word):
        out.append(word[pos:])
    return out


def text_units(text: str, lang: str) -> list[dict]:
    """Split the target text into word units ``{"text", "start", "end"}`` (char offsets); trailing punctuation sticks
    to its unit. Unspaced languages (``UNSPACED``) take one unit per word of each Han / kana run (``cjk_words``) and
    per run of other letters / digits; every other language its whitespace words."""
    spans: list[list[int]] = []  # [start, end)
    lead: int | None = None  # start of non-wordy chars seen before the first unit

    def attach(start: int, end: int) -> None:
        nonlocal lead
        if spans:
            spans[-1][1] = end
        elif lead is None:
            lead = start

    if lang in UNSPACED:
        in_run: set[int] = set()  # offsets inside a Han / kana run
        starts: set[int] = set()  # first char of each run word
        inside: set[int] = set()  # every char of each run word
        for m in _CJK_RUN.finditer(text):
            in_run.update(range(m.start(), m.end()))
            for a, b in cjk_words(m.group()):
                starts.add(m.start() + a)
                inside.update(range(m.start() + a, m.start() + b))
        run_end = -1  # end of the open run of other wordy chars
        for i, ch in enumerate(text):
            if ch.isspace():
                continue
            if i in in_run:
                run_end = -1
                if i in starts:
                    spans.append([i, i + 1])
                elif i in inside:
                    spans[-1][1] = i + 1
                else:
                    attach(i, i + 1)
            elif _wordy(ch):
                if run_end == i:
                    spans[-1][1] = i + 1
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


def align_tokens(text: str, lang: str) -> list[str]:
    """The words of ``text`` for forced alignment; ``lang`` is a Whisper-style code (``en``, ``ja``, ``zh``, ...).

    Unspaced languages (``UNSPACED``: ja, zh, yue) take the unit texts of ``text_units`` (one unit per ICU word of a
    Han / kana run or per Latin/digit run, trailing punctuation attached; whitespace inside a unit removed), every
    other language the whitespace words."""
    if lang in UNSPACED:
        return [token for token in ("".join(u["text"].split()) for u in text_units(text, lang)) if token]
    return text.split()
