import asyncio
import hashlib
import itertools
import json
import os
import time

import pytest
from fastapi import FastAPI

from zonos3.server import projects as P
from zonos3.server.projects import ProjectStore, make_router


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _call(app, method: str, path: str, body: bytes | list[bytes] = b"", headers: dict | None = None,
          query: str = "") -> tuple[int, dict, bytes]:
    """One request through the ASGI app (body chunks sent as separate messages): (status, headers, body)."""
    chunks = body if isinstance(body, list) else [body]
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "root_path": "", "query_string": query.encode(),
             "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
             "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80)}
    pending = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]
    sent: list[dict] = []

    async def receive():
        return pending.pop(0) if pending else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    return (start["status"], {k.decode(): v.decode() for k, v in start["headers"]},
            b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "store", max_project_bytes=4096, max_media_bytes=1024)


@pytest.fixture
def app(store):
    app = FastAPI()
    app.include_router(make_router(lambda: store))
    return app


def _put(app, pid, data, base_revision=None, name="Demo"):
    status, _, body = _call(app, "PUT", f"/projects/{pid}",
                            json.dumps({"name": name, "data": data, "base_revision": base_revision}).encode(),
                            {"content-type": "application/json"})
    return status, json.loads(body)


def _upload(app, blob: bytes, sha: str | None = None, **headers):
    status, _, body = _call(app, "PUT", f"/media/{sha or _sha(blob)}", [blob[:3], blob[3:]], headers)
    return status, json.loads(body)


def test_new_id_then_revisions_and_conflicts(app):
    status, saved = _put(app, "p1", {"lines": [1]})
    assert status == 200 and saved["revision"] == 1 and isinstance(saved["updated_at"], int)
    status, body = _put(app, "p1", {"lines": [1, 2]})                      # base_revision null on an existing id
    assert status == 409 and body["revision"] == 1
    status, saved = _put(app, "p1", {"lines": [1, 2]}, base_revision=1)
    assert status == 200 and saved["revision"] == 2
    status, body = _put(app, "p1", {"lines": ["stale tab"]}, base_revision=1)  # a second tab still on revision 1
    assert status == 409 and body["revision"] == 2
    status, _, raw = _call(app, "GET", "/projects/p1")
    doc = json.loads(raw)
    assert status == 200 and doc["data"] == {"lines": [1, 2]} and doc["revision"] == 2 and doc["name"] == "Demo"
    assert _call(app, "DELETE", "/projects/p1")[0] == 200
    status, body = _put(app, "p1", {"lines": [3]}, base_revision=2)        # deleted elsewhere
    assert status == 409 and body["revision"] is None
    assert _call(app, "GET", "/projects/p1")[0] == 404
    assert _call(app, "DELETE", "/projects/p1")[0] == 404


def test_media_upload_verifies_the_hash_and_serves_with_cache_headers(app, store):
    blob = b"RIFF fake wav bytes"
    status, body = _upload(app, blob, **{"content-type": "audio/wav"})
    assert status == 201 and body == {"sha256": _sha(blob), "size": len(blob), "created": True}
    assert _upload(app, blob)[0] == 200                                    # idempotent
    status, body = _upload(app, b"other bytes", sha=_sha(b"something else"))
    assert status == 400 and body["sha256"] == _sha(b"other bytes")
    assert not (store.media_dir / _sha(b"something else")).exists() and not list(store.tmp_dir.iterdir())
    assert _upload(app, b"x" * 1025)[0] == 413 and not list(store.tmp_dir.iterdir())

    status, headers, raw = _call(app, "GET", f"/media/{_sha(blob)}")
    assert status == 200 and raw == blob and headers["content-type"] == "audio/wav"
    assert headers["etag"] == f'"{_sha(blob)}"' and "immutable" in headers["cache-control"]
    status, headers, raw = _call(app, "HEAD", f"/media/{_sha(blob)}")
    assert status == 200 and raw == b"" and headers["content-length"] == str(len(blob))
    assert _call(app, "GET", f"/media/{_sha(blob)}", headers={"if-none-match": f'"{_sha(blob)}"'})[0] == 304
    assert _call(app, "HEAD", f"/media/{_sha(b'nope')}")[0] == 404
    assert _call(app, "GET", f"/media/{_sha(b'x').upper()}")[0] == 400     # lowercase hex only


def test_missing_media_are_listed(app):
    have, missing = b"have", b"missing"
    _upload(app, have)
    data = {"media": {_sha(have): {"type": "audio/wav", "size": 4}, _sha(missing): {"type": "audio/wav", "size": 7}}}
    status, body = _put(app, "p", data)
    assert status == 400 and body["missing"] == [_sha(missing)]
    _upload(app, missing)
    status, _ = _put(app, "p", data)
    assert status == 200
    assert _put(app, "q", {"media": {"../etc": {}}})[0] == 400


@pytest.mark.parametrize("pid", ["..", ".hidden", "a/b", "_x", "-x", "a b", "x" * 129, "a.json", "%2e%2e"])
def test_path_tricks_are_refused(app, store, pid):
    assert _put(app, pid, {})[0] in (400, 404, 405)
    with pytest.raises(P.ProjectError):
        store.put(pid, "x", {}, None)
    with pytest.raises(P.ProjectError):
        store.get_bytes(pid)
    assert not any(store.root.parent.glob("*.json"))
    assert _call(app, "GET", "/media/..%2F..%2Fsecret")[0] in (400, 404)


def test_list_newest_first_with_summary_and_sizes(app, store, monkeypatch):
    blob = b"0123456789"
    _upload(app, blob)
    clock = itertools.count(1000.0, 1000.0)   # saves in the same millisecond still order
    monkeypatch.setattr(P.time, "time", lambda: next(clock))
    _put(app, "old", {"summary": {"lines": 1}}, name="Old")
    _put(app, "mid", {"media": {_sha(blob): {"type": "x/y", "size": 10}}}, name="Mid")
    _put(app, "old", {"summary": {"lines": 2}}, base_revision=1, name="Old renamed")
    status, _, raw = _call(app, "GET", "/projects")
    listing = json.loads(raw)
    assert status == 200 and [e["id"] for e in listing] == ["old", "mid"]
    assert listing[0]["name"] == "Old renamed" and listing[0]["revision"] == 2
    assert listing[0]["updated_at"] > listing[1]["updated_at"] >= 1_000_000
    assert listing[0]["summary"] == {"lines": 2} and listing[1]["summary"] is None
    assert listing[1]["media_bytes"] == 10 and listing[0]["bytes"] == (store.projects_dir / "old.json").stat().st_size


def test_failed_write_leaves_the_old_file_and_no_temp(store, monkeypatch):
    store.put("p", "Demo", {"v": 1}, None)
    before = (store.projects_dir / "p.json").read_bytes()

    def boom(fd):
        raise OSError("disk full")

    monkeypatch.setattr(P.os, "fsync", boom)
    with pytest.raises(OSError):
        store.put("p", "Demo", {"v": 2}, 1)
    assert (store.projects_dir / "p.json").read_bytes() == before and not list(store.tmp_dir.iterdir())
    monkeypatch.undo()
    with pytest.raises(P.ProjectError) as err:                              # over the size limit: refused whole
        store.put("p", "Demo", {"v": "x" * 5000}, 1)
    assert err.value.status == 413 and (store.projects_dir / "p.json").read_bytes() == before


def test_gc_keeps_referenced_and_fresh_media(app, store):
    kept, dropped, fresh = b"kept media", b"dropped media", b"fresh upload"
    for blob in (kept, dropped, fresh):
        _upload(app, blob, **{"content-type": "audio/wav"})
    _put(app, "p", {"media": {_sha(kept): {}}})
    _put(app, "q", {"media": {_sha(dropped): {}}})
    _call(app, "DELETE", "/projects/q")
    old = time.time() - 7200
    for blob in (kept, dropped):
        os.utime(store.media_dir / _sha(blob), (old, old))
    status, _, raw = _call(app, "POST", "/projects/gc")
    assert status == 200 and json.loads(raw) == {"removed": 1, "freed_bytes": len(dropped), "kept": 2}
    assert sorted(p.name for p in store.media_dir.iterdir()) == sorted(
        [_sha(kept), _sha(kept) + ".type", _sha(fresh), _sha(fresh) + ".type"])
    status, _, raw = _call(app, "POST", "/projects/gc", query="min_age_seconds=0")
    assert json.loads(raw)["removed"] == 1 and (store.media_dir / _sha(kept)).exists()


def test_disabled_store_is_404():
    app = FastAPI()
    app.include_router(make_router(lambda: None))
    assert _call(app, "GET", "/projects")[0] == 404


def test_api_server_serves_the_store_behind_the_access_token(tmp_path, monkeypatch):
    from types import SimpleNamespace

    api_server = pytest.importorskip("zonos3.server.api_server")
    config = SimpleNamespace(studio_projects_dir=str(tmp_path), studio_max_project_bytes=4096,
                             studio_max_media_bytes=1024)
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", SimpleNamespace(config=config))
    monkeypatch.setattr(api_server, "_PROJECT_STORE", None)
    monkeypatch.setenv("ZONOS3_ACCESS_TOKEN", "secret")
    auth = {"authorization": "Bearer secret"}
    blob = b"media bytes"
    assert _call(api_server.app, "PUT", f"/media/{_sha(blob)}", blob)[0] == 403
    assert _call(api_server.app, "PUT", f"/media/{_sha(blob)}", blob, auth)[0] == 201
    status, headers, raw = _call(api_server.app, "GET", f"/media/{_sha(blob)}", headers=auth)
    assert status == 200 and raw == blob and headers["cache-control"] == "private, max-age=31536000, immutable"
    status, headers, _ = _call(api_server.app, "GET", "/projects", headers=auth)
    assert status == 200 and headers["cache-control"] == "no-store"
    assert api_server._project_store().capabilities() == {
        "enabled": True, "max_media_bytes": 1024, "max_project_bytes": 4096}
