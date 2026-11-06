"""The server default for streaming responses is the progressive prefix decode (2026-10-15)."""

import pytest
from pydantic import ValidationError
from zonos3.server import speech
from zonos3.server.schemas import GenerateRequest
from zonos3.tokenizer.matched_codec import codec_names

MATCHED = speech.prefix_stream_supported(sorted(codec_names())[0])


def resolve(value, streaming=True):
    return speech.resolve_stream_decode(value, streaming=streaming, prefix_supported=MATCHED)


def test_matched_codec_streams_progressively_by_default(monkeypatch):
    monkeypatch.delenv("ZONOS3_STREAM_DECODE_DEFAULT", raising=False)
    assert MATCHED and not speech.prefix_stream_supported("not-a-codec")
    assert resolve(None) == "prefix"
    assert resolve("prefix") == "prefix"
    assert resolve("whole") is None  # explicit opt-out
    assert resolve(None, streaming=False) is None  # non-streaming = whole decode
    assert resolve("prefix", streaming=False) is None


def test_env_restores_the_whole_decode_default(monkeypatch):
    monkeypatch.setenv("ZONOS3_STREAM_DECODE_DEFAULT", "whole")
    assert resolve(None) is None
    assert resolve("prefix") == "prefix"  # an explicit request still wins


def test_prefix_needs_a_matched_codec():
    with pytest.raises(ValueError):
        speech.resolve_stream_decode("prefix", streaming=True, prefix_supported=False)
    assert speech.resolve_stream_decode(None, streaming=True, prefix_supported=False) is None


def test_unknown_mode_fails_the_schema():
    with pytest.raises(ValidationError):
        GenerateRequest(text="hi", stream_decode="sideways")
