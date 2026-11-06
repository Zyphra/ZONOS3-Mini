"""Speech editing: align and a one-word edit of a clip whose transcript is known (every editing server), transcribe and
verify (servers with speech recognition, ``edit_transcribe``), the 503 of a server without it, and the keepalive wire
format."""

from __future__ import annotations

import base64
import json

import pytest

from .support import TEXT, check, error, needs


@pytest.fixture(scope="module")
def editing(caps):
    needs(caps["speech_editing"], "the checkpoint or server cannot edit speech")


@pytest.fixture(scope="module")
def known(client, caps, editing) -> dict:
    """A clip the server speaks (``TEXT``, a WAV) and its transcript: what a caller without speech recognition sends."""
    r = client.post("/v2/generate", json={"text": TEXT, "language": "en_us", "stream": False, "format": "wav",
                                          "max_tokens": 150, "seed": 3})
    assert r.status_code == 200, r.text[:500]
    return {"audio_base64": base64.b64encode(r.content).decode("ascii"), "text": TEXT}


@pytest.fixture(scope="module")
def transcript(client, caps, editing, voice_b64):
    needs(caps["edit_transcribe"], "the server has no speech recognition")
    r = client.post("/v2/edit/transcribe", json={"audio_base64": voice_b64, "language": "en_us"})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/EditTranscript")
    assert r.json()["words"]
    return r.json()


def _edit_body(text: str, audio_base64: str) -> dict:
    words = text.split()
    target = " ".join(words[:-1] + ["tomorrow"]) if len(words) > 1 else "tomorrow"
    return {"audio_base64": audio_base64, "original_text": text, "target_text": target, "num_takes": 1, "seed": 7}


def test_align_known_transcript(client, known):
    r = client.post("/v2/edit/align", json={"audio_base64": known["audio_base64"], "text": known["text"],
                                            "language": "en_us"})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/EditTranscript")
    assert r.json()["text"] == known["text"] and r.json()["words"]


def test_one_word_edit_of_a_known_transcript(client, known):
    r = client.post("/v2/edit", json=_edit_body(known["text"], known["audio_base64"]))
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/EditResponse")
    assert r.json()["original_text"] == known["text"]
    take = r.json()["takes"][0]
    assert base64.b64decode(take["audio_base64"])[:4] == b"RIFF" and take["verify"] is None


def test_without_speech_recognition(client, caps, editing, known):
    needs(not caps["edit_transcribe"], "the server has speech recognition")
    clip = known["audio_base64"]
    error(client.post("/v2/edit/transcribe", json={"audio_base64": clip, "language": "en_us"}), 503)
    body = {"audio_base64": clip, "target_text": "Hello there, this is a long test.", "seed": 7}
    error(client.post("/v2/edit", json=body), 503)  # no original_text
    verify = {**_edit_body(known["text"], clip), "verify": True}
    error(client.post("/v2/edit", json=verify), 503)


def test_align(client, transcript, voice_b64):
    r = client.post("/v2/edit/align", json={"audio_base64": voice_b64, "text": transcript["text"], "language": "en_us"})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/EditTranscript")


def test_one_word_edit(client, transcript, voice_b64):
    r = client.post("/v2/edit", json=_edit_body(transcript["text"], voice_b64))
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/EditResponse")
    take = r.json()["takes"][0]
    assert base64.b64decode(take["audio_base64"])[:4] == b"RIFF"


def test_verify(client, caps, known):
    needs(caps["edit_transcribe"], "the server has no speech recognition")
    r = client.post("/v2/edit", json={**_edit_body(known["text"], known["audio_base64"]), "verify": True})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/EditResponse")
    assert r.json()["takes"][0]["verify"] is not None


def test_keepalive(client, known):
    with client.stream("POST", "/v2/edit", json=_edit_body(known["text"], known["audio_base64"]),
                       headers={"X-Keepalive": "1"}) as r:
        assert r.status_code == 200
        assert r.headers.get("x-keepalive") == "1"
        raw = b"".join(r.iter_bytes())
    assert raw[:1] == b" "
    body = json.loads(raw)
    assert "status_code" not in body, body
    check(body, "#/components/schemas/EditResponse")


def test_keepalive_error_is_in_the_body(client, editing):
    r = client.post("/v2/edit/align", json={"audio_base64": "AAAA", "text": "", "language": "en_us"},
                    headers={"X-Keepalive": "1"})
    assert r.status_code == 200
    body = json.loads(r.content)
    assert body["status_code"] == 400 and body["detail"]
