"""Dubbing: reference upload, transcription, analysis, generation, regeneration and (optionally) mux."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from .support import check, error, needs


@pytest.fixture(scope="module")
def dub_caps(client, caps):
    needs(caps["dub"], "dubbing is not available on this server")
    r = client.get("/v2/dub/capabilities")
    assert r.status_code == 200, r.text[:300]
    check(r.json(), "#/components/schemas/DubCapabilities")
    needs(r.json()["enabled"] and r.json()["available"], "dubbing is not available on this server")
    return r.json()


@pytest.fixture(scope="module")
def ref(client, dub_caps, voice_bytes):
    r = client.post("/v2/dub/ref", params={"name": "voice.mp3"}, content=voice_bytes,
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/DubRef")
    return r.json()


@pytest.fixture(scope="module")
def analysis(client, ref, dub_caps):
    r = client.post("/v2/dub/transcribe", json={"ref_id": ref["ref_id"], "ref_lang": "en"})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/DubTranscript")
    target = next((t["code"] for t in dub_caps["target_languages"] if t["code"] == "fr"),
                  dub_caps["target_languages"][0]["code"])
    r = client.post("/v2/dub/analyze", json={"ref_id": ref["ref_id"], "ref_text": r.json()["text"], "ref_lang": "en",
                                             "target_text": "Bonjour, ceci est un court essai de doublage.",
                                             "target_lang": target})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/DubAnalysis")
    return {**r.json(), "target_lang": target}


@pytest.fixture(scope="module")
def dub(client, ref, analysis):
    segments = [{"ref_start": s["ref_start"], "ref_end": s["ref_end"], "target_text": s["target_text"]}
                for s in analysis["segments"]]
    r = client.post("/v2/dub/generate", json={"ref_id": ref["ref_id"], "target_lang": analysis["target_lang"],
                                              "segments": segments, "seed": 0, "lipsync": False})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/DubResult")
    return r.json()


def test_generate(dub):
    assert dub["segments"]


def test_regenerate(client, dub):
    seg = next(s for s in dub["segments"] if s["status"] == "ok")
    r = client.post("/v2/dub/regenerate", json={"dub_id": dub["dub_id"], "segment": seg["index"],
                                                "target_text": seg["target_text"], "seed": 1})
    assert r.status_code == 200, r.text[:500]
    check(r.json(), "#/components/schemas/DubResult")


def test_unknown_ref_is_404(client, dub_caps):
    error(client.post("/v2/dub/transcribe", json={"ref_id": "no_such_ref"}), 404)


def test_mux(client, dub):
    video = os.environ.get("ZONOS3_TEST_VIDEO")
    needs(video, "set ZONOS3_TEST_VIDEO to a video file")
    r = client.post(f"/v2/dub/mux/{dub['dub_id']}", params={"name": Path(video).name},
                    content=Path(video).read_bytes(), headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text[:300]
    assert r.headers["content-type"].startswith("video/mp4")
