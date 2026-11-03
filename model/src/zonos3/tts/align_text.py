"""The tokens a transcript is force-aligned by (FuseAlign: zonos3/workers/fusealign.py places one interval per token)."""

from zonos3.dub.plan import UNSPACED, text_units


def align_tokens(text: str, lang: str) -> list[str]:
    """The words of ``text`` for forced alignment; ``lang`` is a Whisper-style code (``en``, ``ja``, ``zh``, ...).

    Unspaced languages (``zonos3.dub.plan.UNSPACED``: ja, zh, yue) take the unit texts of ``text_units`` (one unit
    per CJK character or Latin/digit run, trailing punctuation attached; whitespace inside a unit removed), every
    other language the whitespace words."""
    if lang in UNSPACED:
        return [token for token in ("".join(u["text"].split()) for u in text_units(text, lang)) if token]
    return text.split()
