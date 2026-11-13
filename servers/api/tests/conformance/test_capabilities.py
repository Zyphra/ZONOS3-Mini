"""GET /v2/capabilities."""

from __future__ import annotations


def test_capabilities_validate(caps):
    # the caps fixture validates against Capabilities
    assert caps["speaker_mode_default"] in [*caps["speaker_modes"], None]
    assert "tag_cfg" not in caps


def test_stream_decode_default_is_a_mode(caps):
    assert caps["stream_decode_default"] in [*caps["stream_decode_modes"], None]


def test_decoders_list_the_default_first(caps, validate):
    decoders = caps["decoders"]
    if caps["decoder"] is None or "error" in caps["decoder"]:
        assert decoders == []
        return
    assert decoders and decoders[0] == caps["decoder"]
    ids = [d["id"] for d in decoders]
    assert len(set(ids)) == len(ids)
    for info in decoders:
        validate(info, "#/components/schemas/DecoderInfo")
        assert info["codec"] == caps["audio_codec"]
