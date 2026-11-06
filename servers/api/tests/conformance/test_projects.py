"""Server-side studio projects and content-addressed media."""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest

from .support import check, error, needs


@pytest.fixture(scope="module")
def media(client, caps):
    needs(caps["projects"]["enabled"], "server-side projects are disabled")
    data = os.urandom(4096)
    sha = hashlib.sha256(data).hexdigest()
    r = client.put(f"/v2/media/{sha}", params={"type": "application/octet-stream"}, content=data)
    assert r.status_code == 201, r.text[:300]
    check(r.json(), "#/components/schemas/MediaStored")
    assert r.json()["created"] is True
    return sha, data


def test_media(client, media):
    sha, data = media
    r = client.put(f"/v2/media/{sha}", content=data)
    assert r.status_code == 200 and r.json()["created"] is False
    assert client.head(f"/v2/media/{sha}").status_code == 200
    r = client.get(f"/v2/media/{sha}")
    assert r.status_code == 200 and r.content == data
    assert r.headers.get("etag") == f'"{sha}"'


def test_media_hash_mismatch_is_400(client, caps):
    needs(caps["projects"]["enabled"], "server-side projects are disabled")
    body = error(client.put(f"/v2/media/{'0' * 64}", content=b"not this"), 400)
    assert body["sha256"] == hashlib.sha256(b"not this").hexdigest()


def test_project_lifecycle(client, media):
    sha, data = media
    pid = f"conf-{uuid.uuid4().hex[:12]}"
    doc = {
        "name": "conformance",
        "data": {"media": {sha: {"type": "application/octet-stream", "size": len(data)}}, "summary": {"lines": 1}},
        "base_revision": None,
    }
    r = client.put(f"/v2/projects/{pid}", json=doc)
    assert r.status_code == 200, r.text[:300]
    check(r.json(), "#/components/schemas/ProjectSaved")
    assert r.json()["revision"] == 1
    r = client.get(f"/v2/projects/{pid}")
    assert r.status_code == 200
    check(r.json(), "#/components/schemas/Project")
    listed = client.get("/v2/projects").json()
    for item in listed:
        check(item, "#/components/schemas/ProjectSummary")
    assert pid in [p["id"] for p in listed]

    stale = error(client.put(f"/v2/projects/{pid}", json={**doc, "base_revision": 7}), 409)
    assert stale["revision"] == 1
    missing = error(
        client.put(
            f"/v2/projects/{pid}",
            json={**doc, "base_revision": 1, "data": {"media": {"f" * 64: {"type": "x/y", "size": 1}}}},
        ),
        400,
    )
    assert missing["missing"] == ["f" * 64]
    error(client.put(f"/v2/projects/{pid}", json={**doc, "base_revision": 1, "extra": 1}), 422)

    r = client.delete(f"/v2/projects/{pid}")
    assert r.status_code == 200 and r.json() == {"deleted": pid}
    error(client.get(f"/v2/projects/{pid}"), 404)
    r = client.post("/v2/projects/gc")
    assert r.status_code == 200
    check(r.json(), "#/components/schemas/ProjectGcResult")


def test_invalid_ids(client, caps):
    needs(caps["projects"]["enabled"], "server-side projects are disabled")
    assert client.get("/v2/projects/_bad").status_code in (400, 404, 422)
    assert client.get("/v2/media/XYZ").status_code in (400, 404, 422)
