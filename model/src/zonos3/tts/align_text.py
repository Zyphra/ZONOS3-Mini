"""The tokens a transcript is force-aligned by (FuseAlign: zonos3/workers/fusealign.py places one interval per token)."""

from __future__ import annotations

import re
import unicodedata

from zonos3 import languages

UNSPACED = frozenset(lang.code for lang in languages.SPOKEN if lang.unspaced)
_CJK_BLOCKS = (
    (0x3040, 0x30FF),
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF66, 0xFF9F),
)
_TOKEN = re.compile(r"\S+")


def _wordy(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LN" or ch == "'"


def _cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_BLOCKS)


def text_units(text: str, lang: str) -> list[dict]:
    """Split the target text into word units ``{"text", "start", "end"}`` (char offsets); trailing punctuation sticks to its unit."""
    spans: list[list[int]] = []  # [start, end)
    lead: int | None = None  # start of non-wordy chars seen before the first unit

    def attach(start: int, end: int) -> None:
        nonlocal lead
        if spans:
            spans[-1][1] = end
        elif lead is None:
            lead = start

    if lang in UNSPACED:
        run_end = -1  # end of the open run of non-CJK wordy chars
        for i, ch in enumerate(text):
            if ch.isspace():
                continue
            if _wordy(ch):
                if _cjk(ch):
                    spans.append([i, i + 1])
                    run_end = -1
                elif run_end == i:
                    spans[-1][1] = i + 1
                    run_end = i + 1
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

    Unspaced languages (``UNSPACED``: ja, zh, yue) take the unit texts of ``text_units`` (one unit per CJK character
    or Latin/digit run, trailing punctuation attached; whitespace inside a unit removed), every other language the
    whitespace words."""
    if lang in UNSPACED:
        return [token for token in ("".join(u["text"].split()) for u in text_units(text, lang)) if token]
    return text.split()
