"""GET /v2/capabilities."""

from __future__ import annotations


def test_capabilities_validate(caps):
    # the caps fixture validates against Capabilities
    assert caps["speaker_mode_default"] in [*caps["speaker_modes"], None]


def test_stream_decode_default_is_a_mode(caps):
    assert caps["stream_decode_default"] in [*caps["stream_decode_modes"], None]
