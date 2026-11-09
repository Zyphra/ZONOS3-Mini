"""Transcript tokens and error rates of speech editing: the edit's word diff, verify WER and CER.

``normalize`` and ``char_error_rate`` are copied verbatim from ``z_speech_alignment/textmatch.py`` (Zyphra's
z-speech-alignment, MIT License, Copyright (c) 2026 Sofian Mejjoute; third_party/licenses/z-speech-alignment-MIT.txt), which ported
``normalize`` from the k2-fsa/libriheavy corpus pipeline (``scripts/filter_by_cer.py``). ``word_errors`` is the word
Levenshtein of the edit's former recognition worker.
"""

from __future__ import annotations

import re

from zonos3.tts.align_text import split_cjk

# Curly quotes are folded so "don't" and "don’t" are one token. Every letter
# or digit of any script is kept: libriheavy's ``[A-Z0-9']`` rule reduced a
# Mandarin token to the empty string, so a guide in an unspaced script
# matched 5% of its own transcript (the empty tokens, against each other).
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201b": "'", "\u02bc": "'"})
_KEEP = re.compile(r"[^\w']+")


def normalize(text: str) -> str:
    """Uppercase; keep word characters and ``'``; everything else becomes one space.

    Curly quotes are mapped to ``'`` first. Libriheavy's rule, widened from
    ASCII to every script; ``_`` is a word character to ``\\w`` and is dropped.
    """
    return _KEEP.sub(" ", text.translate(_QUOTES).replace("_", " ").upper()).strip()


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein over :func:`normalize`d characters divided by ``len(reference)``.

    Returns 0.0 when the normalised reference is empty. This is
    ``O(len(ref) * len(hyp))`` in time and belongs on segment-sized strings:
    a whole-file rate is the length-weighted mean of the per-segment rates,
    never one call over two transcripts.

    One deliberate departure from libriheavy's ``filter_by_cer``: word
    delimiters are dropped before the DP rather than counted as characters.
    Their reference and hypothesis come from the same tokenizer, so a space
    is a real character there; here the hypothesis is a separate model's
    greedy decode, and charging its word segmentation as character errors
    would measure the tokenizer, not the transcript.
    """
    ref = "".join(normalize(reference).split())
    hyp = "".join(normalize(hypothesis).split())
    if not ref:
        return 0.0
    if not hyp:
        return 1.0
    prev = list(range(len(hyp) + 1))
    cur = [0] * (len(hyp) + 1)
    for i, rc in enumerate(ref, start=1):
        cur[0] = i
        for j, hc in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev, cur = cur, prev
    return prev[len(hyp)] / len(ref)


def tokens(text: str) -> list[str]:
    """The edit tokens of a text: the whitespace words of its ``normalize``, each Han / kana run cut into its words
    (``zonos3.tts.align_text.split_cjk``: the units FuseAlign aligns unspaced text by)."""
    return [token for word in normalize(text).split() for token in split_cjk(word)]


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Word-level Levenshtein distance (substitutions, insertions, deletions) between two token lists."""
    prev = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        row = [i]
        for j, hyp in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (ref != hyp)))
        prev = row
    return prev[-1]


def verify(reference: str, hypothesis: str) -> dict:
    """``{"word_errors", "ref_words", "cer"}`` of a transcript ``hypothesis`` against the ``reference`` text: the word
    errors over the tokens of each, the reference's token count and ``char_error_rate``."""
    ref = tokens(reference)
    return {
        "word_errors": word_errors(ref, tokens(hypothesis)),
        "ref_words": len(ref),
        "cer": char_error_rate(reference, hypothesis),
    }
