"""POST /v1/audio/speech."""

from __future__ import annotations

import io
import wave

from .support import TEXT, error, generate, pcm


def speech(client, **fields):
    return generate(client, {"model": "zonos3", "input": TEXT, **fields}, path="/v1/audio/speech")


def test_pcm(client):
    r = speech(client)
    pcm(r)
    assert r.headers["x-audio-format"] == "float32"
    assert r.headers["x-seed"] == "1"


def test_wav(client):
    r = speech(client, response_format="wav")
    assert r.status_code == 200, r.text[:300]
    with wave.open(io.BytesIO(r.content)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (44100, 1, 2)


def test_voice_is_a_speaker_id(client, default_voice):
    pcm(speech(client, voice=default_voice["id"]))


def test_unknown_voice_is_400(client):
    body = error(speech(client, voice="alloy"), 400)
    assert "unknown voice" in body["detail"]


def test_voice_with_speaker_fields_is_400(client, default_voice, voice_b64):
    error(speech(client, voice=default_voice["id"], speaker_audio_base64=voice_b64), 400)


def test_speed_is_422(client):
    error(speech(client, speed=1.0), 422)


def test_missing_model_is_422(client):
    error(client.post("/v1/audio/speech", json={"input": TEXT}), 422)
