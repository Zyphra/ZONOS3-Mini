"""Speech editing: transcribe, align, a one-word edit, and the keepalive wire format."""

from __future__ import annotations

import base64
import json

import pytest

from .support import check, needs


@pytest.fixture(scope="module")
def transcript(client, caps, voice_b64):
    needs(caps["speech_editing"], "the checkpoint or server cannot edit speech")
    r = client.post("/v2/edit/transcribe", json={"audio_base64": voice_b64, "language": "en_us"})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/EditTranscript")
    assert r.json()["words"]
    return r.json()


def test_align(client, transcript, voice_b64):
    r = client.post("/v2/edit/align", json={"audio_base64": voice_b64, "text": transcript["text"], "language": "en_us"})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/EditTranscript")


def _edit_body(transcript, voice_b64) -> dict:
    words = transcript["text"].split()
    target = " ".join(words[:-1] + ["tomorrow"]) if len(words) > 1 else "tomorrow"
    return {"audio_base64": voice_b64, "original_text": transcript["text"], "target_text": target,
            "num_takes": 1, "seed": 7}


def test_one_word_edit(client, transcript, voice_b64):
    r = client.post("/v2/edit", json=_edit_body(transcript, voice_b64))
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/EditResponse")
    take = r.json()["takes"][0]
    assert base64.b64decode(take["audio_base64"])[:4] == b"RIFF"


def test_keepalive(client, transcript, voice_b64):
    with client.stream("POST", "/v2/edit", json=_edit_body(transcript, voice_b64), headers={"X-Keepalive": "1"}) as r:
        assert r.status_code == 200
        assert r.headers.get("x-keepalive") == "1"
        raw = b"".join(r.iter_bytes())
    assert raw[:1] == b" "
    body = json.loads(raw)
    assert "status_code" not in body, body
    check(body, "#/components/schemas/EditResponse")


def test_keepalive_error_is_in_the_body(client, caps):
    needs(caps["speech_editing"], "the checkpoint or server cannot edit speech")
    r = client.post("/v2/edit/align", json={"audio_base64": "AAAA", "text": "", "language": "en_us"},
                    headers={"X-Keepalive": "1"})
    assert r.status_code == 200
    body = json.loads(r.content)
    assert body["status_code"] == 400 and body["detail"]
