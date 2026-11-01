"""Nonverbal (NV) markup of the nonverbal fine-tune recipe (torch-free)."""

from __future__ import annotations

import re

# Rendered into training transcripts as point tags ("<laugh/>") and span tags ("<shouting>Get out!</shouting>").
# Whispering, singing and clear-throat were dropped from that recipe, and synonyms like <giggle/> were never trained.
NONVERBAL_MARKUP_TAGS = ("laugh", "breath", "sigh", "cough", "panting", "shouting", "crying")
_NONVERBAL_MARKUP = re.compile(rf"</?(?:{'|'.join(NONVERBAL_MARKUP_TAGS)})\s*/?>")
_NONVERBAL_MARKUP_SPLIT = re.compile(f"({_NONVERBAL_MARKUP.pattern})")


def has_nonverbal_markup(text: str) -> bool:
    """Whether text contains a recognized nonverbal annotation."""
    return _NONVERBAL_MARKUP.search(text) is not None


def split_nonverbal_markup(text: str) -> list[str]:
    """Alternate text / marker pieces: even indices are the (possibly empty) text around the markers, odd ones the
    markers themselves, so ``"".join(pieces) == text``."""
    return _NONVERBAL_MARKUP_SPLIT.split(text)


def nonverbal_marker_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges [start, end) of the recognized markers in ``text``."""
    return [m.span() for m in _NONVERBAL_MARKUP.finditer(text)]


def strip_nonverbal_markup(text: str) -> str:
    """Remove learned nonverbal markers, preserving their enclosed spoken words."""
    if not has_nonverbal_markup(text):
        return text
    return " ".join(_NONVERBAL_MARKUP.sub(" ", text).split())


def trained_on_tags(checkpoint_dir) -> bool:
    """Whether a checkpoint was fine-tuned on nonverbal markup: its training config (``config.yaml`` next to
    ``params.json``, kept by the converter and by tools/add_text_extra_rows.py) has a data source with
    ``nonverbal: true`` (the NV recipe's tagged ``zv3subsample``). Other checkpoints read tags as ordinary text."""
    from pathlib import Path

    import yaml

    path = Path(checkpoint_dir) / "config.yaml"
    if not path.is_file():
        return False
    try:
        sources = ((yaml.safe_load(path.read_text()) or {}).get("data") or {}).get("sources") or {}
    except (OSError, yaml.YAMLError, AttributeError):
        return False
    return isinstance(sources, dict) and any(isinstance(s, dict) and s.get("nonverbal") is True for s in sources.values())
