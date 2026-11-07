"""Accent adapters."""

from __future__ import annotations

from .support import error, generate, needs, pcm


def test_first_accent(client, caps):
    needs(caps["accents"], "the server has no accents")
    pcm(generate(client, {"accent": caps["accents"][0]}))


def test_unknown_accent_is_400(client, caps):
    needs(caps["accents"], "the server has no accents")
    error(generate(client, {"accent": "no_such_accent"}), 400)
