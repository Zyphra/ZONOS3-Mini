"""Speakers: listing, session-scoped uploads, previews, embeddings and generation by id."""

from __future__ import annotations

import uuid

import pytest

from .support import check, error, generate, needs, pcm


@pytest.fixture
def uploaded(client, caps, session, voice_b64):
    needs(caps["speaker_audio_upload"], "the checkpoint takes no speaker audio")
    r = client.post(
        "/v2/speakers",
        json={"label": "conformance", "speaker_audio_base64": voice_b64, "speaker_audio_name": "voice.mp3"},
        headers=session,
    )
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/Speaker")
    return r.json()


def test_list_validates(speakers):
    for s in speakers:
        if s["source_type"] == "embedding":
            assert s["speaker_modes"] == ["embedding"]


def test_create_needs_a_session(client, voice_b64):
    error(client.post("/v2/speakers", json={"speaker_audio_base64": voice_b64}), 400)


def test_create_needs_exactly_one_source(client, session, voice_b64):
    error(client.post("/v2/speakers", json={}, headers=session), 400)
    error(
        client.post(
            "/v2/speakers",
            json={"speaker_audio_base64": voice_b64, "speaker_embedding_base64": voice_b64},
            headers=session,
        ),
        400,
    )


def test_create_rejects_unknown_fields(client, session, voice_b64):
    error(client.post("/v2/speakers", json={"speaker_wav_base64": voice_b64}, headers=session), 422)


def test_upload_is_visible_only_in_its_session(client, uploaded, session):
    assert uploaded["scope"] == "session" and not uploaded["is_default"]
    mine = client.get("/v2/speakers", headers=session).json()["speakers"]
    assert uploaded["id"] in [s["id"] for s in mine]
    other = client.get("/v2/speakers", headers={"X-TTS-Session-ID": uuid.uuid4().hex}).json()["speakers"]
    assert uploaded["id"] not in [s["id"] for s in other]
    assert uploaded["id"] not in [s["id"] for s in client.get("/v2/speakers").json()["speakers"]]


def test_preview(client, default_voice):
    needs(default_voice["has_preview"], "the default voice has no preview audio")
    r = client.get(f"/v2/speakers/{default_voice['id']}/preview")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")
    assert r.content


def test_embedding(client, caps, default_voice, validate):
    needs("embedding" in caps["speaker_modes"], "the checkpoint has no speaker-embedding path")
    r = client.get(f"/v2/speakers/{default_voice['id']}/embedding")
    assert r.status_code == 200, r.text[:300]
    validate(r.json(), "#/components/schemas/SpeakerEmbedding")
    assert len(r.json()["embedding"]) == caps["speaker_embedding_dim"]


def test_generate_by_id_in_every_mode(client, caps, default_voice):
    needs(caps["speaker_modes"], "the checkpoint has no speaker conditioning")
    for mode in caps["speaker_modes"]:
        if mode in default_voice["speaker_modes"]:
            pcm(generate(client, {"speaker_embedding_id": default_voice["id"], "speaker_mode": mode}))


def test_generate_by_uploaded_id(client, uploaded, session):
    pcm(generate(client, {"speaker_embedding_id": uploaded["id"]}, headers=session))


def test_blend(client, caps, speakers):
    needs(caps["speaker_embedding_blend"], "speaker_embedding_blend is off")
    defaults = [s for s in speakers if s["is_default"]]
    needs(len(defaults) >= 2, "blending needs two default voices")
    pcm(
        generate(
            client,
            {
                "speaker_mode": "embedding",
                "speaker_blend_embedding_id_a": defaults[0]["id"],
                "speaker_blend_embedding_id_b": defaults[1]["id"],
                "speaker_blend_t": 0.5,
            },
        )
    )


def test_unknown_id_is_400(client, caps):
    needs(caps["speaker_enabled"], "the checkpoint has no speaker conditioning")
    error(generate(client, {"speaker_embedding_id": "no_such_voice"}), 400)


def test_unknown_id_resources_are_errors(client):
    assert client.get("/v2/speakers/no_such_voice/embedding").status_code in (400, 404)
    assert client.get("/v2/speakers/no_such_voice/preview").status_code in (400, 404)
