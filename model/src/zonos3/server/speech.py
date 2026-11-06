"""The speech responses of the Python servers (/v2/generate, /v1/audio/speech): the stream-decode rule, the format
check and the response headers."""

from __future__ import annotations

import os

from zonos3.generation import SAMPLE_RATE

STREAM_DECODE_MODES = ("prefix", "whole")


def prefix_stream_supported(audio_codec: str) -> bool:
    """Progressive (prefix re-decode) streaming exists for the matched long-context codecs only."""
    from zonos3.tokenizer.matched_codec import codec_names

    return str(audio_codec) in codec_names()


def stream_decode_default(prefix_supported: bool) -> str | None:
    """The server default for a streaming response: progressive prefix re-decode whenever the codec supports it
    (first audio after a few frames instead of at EOS). ZONOS3_STREAM_DECODE_DEFAULT=whole restores the one exact
    decode at EOS for every request that does not ask for a mode (None)."""
    wanted = os.environ.get("ZONOS3_STREAM_DECODE_DEFAULT", "prefix").strip().lower()
    return "prefix" if wanted == "prefix" and prefix_supported else None


def resolve_stream_decode(value: str | None, *, streaming: bool, prefix_supported: bool) -> str | None:
    """GenerateRequest.stream_decode: None = the server default (prefix for streaming responses of the matched
    codecs, whole decode otherwise), "prefix", or "whole" (one exact decode at EOS). Non-streaming responses are
    always the whole decode. Returns "prefix" or None (the whole decode); ValueError (HTTP 400) for prefix on a codec
    without it."""
    if value is None:
        return stream_decode_default(prefix_supported) if streaming else None
    if value == "whole" or not streaming:
        return None
    if not prefix_supported:
        raise ValueError("stream_decode=prefix is only available for the matched long-context codecs")
    return value


def check_format(stream: bool, fmt: str) -> None:
    """A 16-bit WAV is buffered: ValueError (HTTP 400) for format=wav on a streamed response."""
    if stream and fmt == "wav":
        raise ValueError("format=wav needs stream=false")


def speech_headers(*, fmt: str, seed: int, control_spans: bool, follows_reading: bool) -> dict[str, str]:
    """The headers of a speech response: its audio format, the seed it sampled with and, with ``control_spans``,
    whether the painted spans follow the reading (``follows_reading``) or the speaking rate."""
    headers = {"X-Audio-Sample-Rate": str(SAMPLE_RATE), "X-Audio-Channels": "1",
               "X-Audio-Format": "pcm16" if fmt == "wav" else "float32", "X-Seed": str(seed)}
    if control_spans:
        headers["X-Control-Span-Follow"] = "reading" if follows_reading else "rate"
    return headers
