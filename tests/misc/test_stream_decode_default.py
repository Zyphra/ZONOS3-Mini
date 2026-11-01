"""The server default for streaming responses is the progressive prefix decode (2026-10-15)."""
from types import SimpleNamespace

import pytest

api_server = pytest.importorskip("zonos3.server.api_server")
from zonos3.tokenizer.matched_codec import codec_names  # noqa: E402

MATCHED = SimpleNamespace(model_config=SimpleNamespace(audio_codec=sorted(codec_names())[0]))


def test_matched_codec_streams_progressively_by_default(monkeypatch):
    monkeypatch.delenv("ZONOS3_STREAM_DECODE_DEFAULT", raising=False)
    assert api_server._resolve_stream_decode(MATCHED, None) == "prefix"
    assert api_server._resolve_stream_decode(MATCHED, "") == "prefix"
    assert api_server._resolve_stream_decode(MATCHED, "prefix") == "prefix"
    assert api_server._resolve_stream_decode(MATCHED, "whole") is None          # explicit opt-out
    assert api_server._resolve_stream_decode(MATCHED, None, streaming=False) is None   # non-streaming = whole decode
    assert api_server._resolve_stream_decode(MATCHED, "prefix", streaming=False) is None


def test_env_restores_the_whole_decode_default(monkeypatch):
    monkeypatch.setenv("ZONOS3_STREAM_DECODE_DEFAULT", "whole")
    assert api_server._resolve_stream_decode(MATCHED, None) is None
    assert api_server._resolve_stream_decode(MATCHED, "prefix") == "prefix"      # an explicit request still wins


def test_unknown_mode_is_refused():
    with pytest.raises(api_server.HTTPException):
        api_server._resolve_stream_decode(MATCHED, "sideways")
