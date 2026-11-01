"""POST /v2/generate: pcm (buffered, streamed), wav, seeds, budgets and request validation."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from .support import LEGACY_FIELDS, MAX_TOKENS, SAMPLES_PER_FRAME, TEXT, error, generate, pcm


def test_buffered_pcm(client):
    r = generate(client, {})
    pcm(r)
    assert r.headers["x-audio-sample-rate"] == "44100"
    assert r.headers["x-audio-channels"] == "1"
    assert r.headers["x-audio-format"] == "float32"
    assert r.headers["x-seed"] == "1"


def test_streamed_pcm(client):
    body = {"text": TEXT, "stream": True, "max_tokens": MAX_TOKENS, "seed": 2}
    with client.stream("POST", "/v2/generate", json=body) as r:
        assert r.status_code == 200
        data = b"".join(r.iter_bytes())
    assert r.headers["x-seed"] == "2"
    assert len(data) % 4 == 0 and len(data) > 0
    assert np.isfinite(np.frombuffer(data, dtype="<f4")).all()


def test_wav(client):
    r = generate(client, {"format": "wav"})
    assert r.status_code == 200, r.text[:300]
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"
    with wave.open(io.BytesIO(r.content)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (44100, 1, 2)
        assert w.getnframes() > 0


def test_stream_wav_is_rejected(client):
    error(generate(client, {"format": "wav", "stream": True}), 400)


def test_seed_is_echoed_and_drawn(client):
    r = generate(client, {"seed": 4294967295})
    pcm(r)
    assert r.headers["x-seed"] == "4294967295"
    body = {"text": TEXT, "stream": False, "max_tokens": MAX_TOKENS}
    r = client.post("/v2/generate", json=body)
    pcm(r)
    assert 0 <= int(r.headers["x-seed"]) < 2**31


def test_max_tokens_bounds_the_audio(client):
    # 8 delayed rows may hold no complete codec frame (the delay pattern spans n_codebooks - 1 rows): audio may be empty
    r = generate(client, {"max_tokens": 8})
    assert r.status_code == 200 and len(r.content) % 4 == 0
    audio = np.frombuffer(r.content, dtype="<f4")
    assert np.isfinite(audio).all() and audio.size <= (8 + 30) * SAMPLES_PER_FRAME


def test_unknown_field_is_422(client):
    body = error(generate(client, {"no_such_field": 1}), 422)
    assert body["detail"]


@pytest.mark.parametrize("field", sorted(LEGACY_FIELDS))
def test_removed_legacy_field_is_422(client, field):
    error(generate(client, {field: LEGACY_FIELDS[field]}), 422)


@pytest.mark.parametrize(
    "fields",
    [{"headroom_db": 25}, {"speaker_mode": "x"}, {"seed": -1}, {"seed": 2**32}, {"stream_decode": "x"},
     {"format": "mp3"}, {"control_span_follow": "x"}, {"text": 5}, {"speaker_blend_t": 2}],
)
def test_schema_violations_are_422(client, fields):
    error(generate(client, fields), 422)


def test_missing_text_is_422(client):
    error(client.post("/v2/generate", json={"stream": False}), 422)
