"""Tag CFG (tagged-vs-plain guidance on nonverbal markup)."""

from __future__ import annotations

from .support import error, generate, needs, pcm


def test_tag_cfg(client, caps):
    needs(caps["tag_cfg"]["available"], "tag CFG is unavailable")
    pcm(generate(client, {"text": "That is so funny <laugh/> really.", "tag_cfg_scale": 2.0}))


def test_tag_cfg_without_markup_is_400(client, caps):
    needs(caps["tag_cfg"]["available"], "tag CFG is unavailable")
    error(generate(client, {"text": "No markup here.", "tag_cfg_scale": 2.0}), 400)


def test_tag_cfg_out_of_range_is_422(client):
    error(generate(client, {"text": "Funny <laugh/>.", "tag_cfg_scale": 5}), 422)
