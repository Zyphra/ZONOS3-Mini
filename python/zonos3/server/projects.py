"""Zonos Voice Studio projects on the server: shared by every browser with the studio link, kept across restarts.

Layout under ``--studio-projects-dir`` (README "Projects on the server"):

- ``projects/<id>.json``: ``{id, name, revision, updated_at, media_bytes, data}``; ``data`` is the studio's project
  document, ``revision`` an integer bumped by every save, ``updated_at`` epoch milliseconds.
- ``media/<sha256>``: content-addressed blobs (written once, verified against their hash), ``media/<sha256>.type``
  their optional MIME type.
- ``tmp/``: temp files, renamed into place (atomic writes on the same filesystem); ``.lock``: the write lock.

A project's media references are the keys of ``data.media`` (``{<sha256>: {type, size}}``, kept by the studio): a save
needs them all stored (400 lists the missing ones) and the GC keeps every blob some project lists. Saves carry the
revision they were based on, so two tabs or browsers never overwrite each other silently (409 with the stored one).
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, AsyncIterable, Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

MAX_PROJECT_BYTES = 20 * 1024**2
MAX_MEDIA_BYTES = 2 * 1024**3
GC_MIN_AGE_SECONDS = 3600.0   # newer unreferenced media may belong to a save in flight (media go up first)
_STALE_TMP_SECONDS = 86400.0
_MEDIA_CACHE = "public, max-age=31536000, immutable"

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MIME = re.compile(r"[A-Za-z0-9][\w.+-]*/[\w.+-]+(\s*;\s*[\w.+-]+=[\w.+\"-]+)*")
_MAX_NAME = 200


class ProjectError(Exception):
    """An HTTP error with extra JSON fields next to ``detail`` (409 ``revision``, 400 ``missing``)."""

    def __init__(self, status: int, detail: str, **extra: Any):
        super().__init__(detail)
        self.status, self.detail, self.extra = status, detail, extra

    def response(self) -> JSONResponse:
        return JSONResponse({"detail": self.detail, **self.extra}, status_code=self.status,
                            headers={"Cache-Control": "no-store"})


def check_id(project_id: str) -> str:
    if not isinstance(project_id, str) or not _ID.fullmatch(project_id):
        raise ProjectError(400, "invalid project id (1-128 of A-Z a-z 0-9 _ -, not starting with _ or -)")
    return project_id


def check_sha256(sha256: str) -> str:
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ProjectError(400, "invalid media hash (64 lowercase hex digits of the content's SHA-256)")
    return sha256


def media_refs(data: Any) -> list[str]:
    """The hashes a project document references: the keys of ``data.media``."""
    media = data.get("media") if isinstance(data, dict) else None
    if media is None:
        return []
    if not isinstance(media, dict):
        raise ProjectError(400, "data.media must be an object {sha256: {type, size}}")
    return [check_sha256(key) for key in media]


def _reject_constant(name: str):
    raise ValueError(f"{name} is not JSON")


def parse_json(body: bytes) -> Any:
    """Strict JSON (no NaN / Infinity, which browsers' JSON.parse refuses)."""
    try:
        return json.loads(body, parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProjectError(400, f"invalid JSON: {exc}") from None


class ProjectStore:
    def __init__(self, root: str | os.PathLike, max_project_bytes: int = MAX_PROJECT_BYTES,
                 max_media_bytes: int = MAX_MEDIA_BYTES):
        self.root = Path(root)
        self.projects_dir, self.media_dir, self.tmp_dir = (self.root / d for d in ("projects", "media", "tmp"))
        for d in (self.projects_dir, self.media_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.max_project_bytes, self.max_media_bytes = int(max_project_bytes), int(max_media_bytes)
        self._lock_path = self.root / ".lock"
        self._thread_lock = threading.Lock()
        self._list_cache: dict[str, tuple[tuple[int, int], dict]] = {}
        cutoff = time.time() - _STALE_TMP_SECONDS
        for tmp in self.tmp_dir.iterdir():   # leftovers of a killed server; fresh ones may be another server's
            with suppress(OSError):
                if tmp.stat().st_mtime < cutoff:
                    tmp.unlink()

    def capabilities(self) -> dict:
        return {"enabled": True, "max_media_bytes": self.max_media_bytes, "max_project_bytes": self.max_project_bytes}

    # ------------------------------------------------------------------ internals
    @contextmanager
    def _locked(self):
        """Serialize writers: threads of this server (thread lock) and other servers on the directory (flock)."""
        with self._thread_lock, open(self._lock_path, "a+b") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            yield

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        """Temp file in tmp/, fsync, rename over ``path``: readers see the old or the new file, never a part."""
        fd, tmp = tempfile.mkstemp(dir=self.tmp_dir, prefix=path.name + ".")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    def _project_path(self, project_id: str) -> Path:
        return self.projects_dir / f"{check_id(project_id)}.json"

    def _media_path(self, sha256: str) -> Path:
        return self.media_dir / check_sha256(sha256)

    @staticmethod
    def _read_record(path: Path) -> dict:
        return json.loads(path.read_bytes())

    # ------------------------------------------------------------------ projects
    def list(self) -> list[dict]:
        """``[{id, name, updated_at, revision, bytes, media_bytes, summary}]``, newest first."""
        entries, seen = [], set()
        for path in self.projects_dir.glob("*.json"):
            project_id = path.stem
            if not _ID.fullmatch(project_id):
                continue
            try:
                st = path.stat()
                key = (st.st_mtime_ns, st.st_size)
                cached = self._list_cache.get(project_id)
                if cached is None or cached[0] != key:
                    record = self._read_record(path)
                    data = record.get("data")
                    entry = {
                        "id": project_id, "name": record.get("name"), "updated_at": record.get("updated_at"),
                        "revision": record.get("revision"), "bytes": st.st_size,
                        "media_bytes": record.get("media_bytes", 0),
                        "summary": data.get("summary") if isinstance(data, dict) else None,
                    }
                    cached = self._list_cache[project_id] = (key, entry)
            except (OSError, ValueError):
                continue   # deleted meanwhile, or not ours
            seen.add(project_id)
            entries.append(cached[1])
        for gone in self._list_cache.keys() - seen:
            self._list_cache.pop(gone, None)
        entries.sort(key=lambda e: (-(e["updated_at"] or 0), e["id"]))
        return entries

    def get_bytes(self, project_id: str) -> bytes:
        """The stored ``{id, name, revision, updated_at, media_bytes, data}`` JSON, verbatim."""
        try:
            return self._project_path(project_id).read_bytes()
        except FileNotFoundError:
            raise ProjectError(404, "no such project") from None

    def get(self, project_id: str) -> dict:
        return json.loads(self.get_bytes(project_id))

    def put(self, project_id: str, name: Any, data: Any, base_revision: Any) -> dict:
        """Save ``data`` as ``project_id`` if ``base_revision`` is the stored revision (None for a new id)."""
        path = self._project_path(project_id)
        if not isinstance(name, str) or not name.strip() or len(name) > _MAX_NAME:
            raise ProjectError(400, f"name must be a non-empty string of at most {_MAX_NAME} characters")
        if not isinstance(data, dict):
            raise ProjectError(400, "data must be a JSON object")
        if base_revision is not None and (not isinstance(base_revision, int) or isinstance(base_revision, bool)):
            raise ProjectError(400, "base_revision must be an integer or null")
        refs = media_refs(data)
        with self._locked():
            try:
                stored = self._read_record(path)["revision"]
            except FileNotFoundError:
                stored = None
            if base_revision != stored:
                raise ProjectError(409, "the project changed elsewhere" if base_revision is not None and stored
                                   else "a project with this id exists" if stored
                                   else "the project does not exist (deleted elsewhere?)", revision=stored)
            media_bytes, missing = 0, []
            for sha in refs:
                try:
                    media_bytes += (self.media_dir / sha).stat().st_size
                except FileNotFoundError:
                    missing.append(sha)
            if missing:
                raise ProjectError(400, f"{len(missing)} media not uploaded (PUT /media/<sha256> first)",
                                   missing=missing)
            revision = (stored or 0) + 1
            updated_at = int(time.time() * 1000)
            record = {"id": project_id, "name": name, "revision": revision, "updated_at": updated_at,
                      "media_bytes": media_bytes, "data": data}
            payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            if len(payload) > self.max_project_bytes:
                raise ProjectError(413, f"project too large ({len(payload)} > {self.max_project_bytes} bytes)")
            self._atomic_write(path, payload)
        return {"id": project_id, "revision": revision, "updated_at": updated_at}

    def delete(self, project_id: str) -> None:
        path = self._project_path(project_id)
        with self._locked():
            try:
                path.unlink()
            except FileNotFoundError:
                raise ProjectError(404, "no such project") from None
        self._list_cache.pop(project_id, None)

    def gc(self, min_age_seconds: float = GC_MIN_AGE_SECONDS) -> dict:
        """Remove media no project's ``data.media`` lists, older than ``min_age_seconds`` (uploads of a save in
        flight are newer). Returns ``{removed, freed_bytes, kept}``."""
        with self._locked():
            referenced: set[str] = set()
            for path in self.projects_dir.glob("*.json"):
                try:   # an unreadable project could reference anything: keep every blob
                    referenced.update(media_refs(self._read_record(path).get("data")))
                except (OSError, ValueError, ProjectError) as exc:
                    raise ProjectError(500, f"gc aborted: {path.name} is unreadable ({exc})") from None
            cutoff = time.time() - min_age_seconds
            removed = freed = kept = 0
            for path in self.media_dir.iterdir():
                sha = path.name.removesuffix(".type")
                if not _SHA256.fullmatch(sha) or sha in referenced:
                    kept += path.name == sha
                    continue
                if path.name != sha:   # a type file goes with its blob
                    if not (self.media_dir / sha).exists():
                        path.unlink(missing_ok=True)
                    continue
                st = path.stat()
                if st.st_mtime > cutoff:
                    kept += 1
                    continue
                path.unlink(missing_ok=True)
                (self.media_dir / f"{sha}.type").unlink(missing_ok=True)
                removed, freed = removed + 1, freed + st.st_size
        return {"removed": removed, "freed_bytes": freed, "kept": kept}

    # ------------------------------------------------------------------ media
    def media(self, sha256: str) -> tuple[Path, str | None]:
        """The blob's path and its stored MIME type; 404 when absent."""
        path = self._media_path(sha256)
        if not path.is_file():
            raise ProjectError(404, "no such media")
        try:
            mime = (self.media_dir / f"{sha256}.type").read_text().strip() or None
        except FileNotFoundError:
            mime = None
        return path, mime

    def _store_type(self, sha256: str, mime: str | None) -> None:
        type_path = self.media_dir / f"{sha256}.type"
        if mime and not type_path.exists():
            self._atomic_write(type_path, mime.encode())

    async def put_media(self, sha256: str, chunks: AsyncIterable[bytes], mime: str | None = None) -> dict:
        """Store a blob streamed in ``chunks`` under its SHA-256: written to tmp/ while hashed (never held in
        memory), renamed into media/ when the hash matches (400 otherwise). Idempotent: a stored blob is kept and
        the body not read. Returns ``{sha256, size, created}``."""
        check_sha256(sha256)
        mime = (mime or "").strip() or None
        if mime is not None and (len(mime) > 200 or not _MIME.fullmatch(mime)):
            raise ProjectError(400, "invalid media type")
        path = self.media_dir / sha256
        if path.is_file():
            with suppress(OSError):
                os.utime(path)   # restart the GC grace period: a save referencing it follows
            await asyncio.to_thread(self._store_type, sha256, mime)
            return {"sha256": sha256, "size": path.stat().st_size, "created": False}
        digest, size = hashlib.sha256(), 0
        fd, tmp = tempfile.mkstemp(dir=self.tmp_dir, prefix=f"{sha256}.")
        try:
            with os.fdopen(fd, "wb") as fh:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > self.max_media_bytes:
                        raise ProjectError(413, f"media too large (> {self.max_media_bytes} bytes)")
                    digest.update(chunk)
                    fh.write(chunk)
                fh.flush()
                await asyncio.to_thread(os.fsync, fh.fileno())
            actual = digest.hexdigest()
            if actual != sha256:
                raise ProjectError(400, "content does not match its hash", sha256=actual)
            os.replace(tmp, path)   # a concurrent upload of the same blob renames identical bytes
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
        await asyncio.to_thread(self._store_type, sha256, mime)
        return {"sha256": sha256, "size": size, "created": True}


async def _read_body(request: Request, limit: int) -> bytes:
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > limit:
        raise ProjectError(413, f"request too large (> {limit} bytes)")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise ProjectError(413, f"request too large (> {limit} bytes)")
    return bytes(body)


def make_router(get_store: Callable[[], ProjectStore | None]) -> APIRouter:
    """``/projects`` and ``/media`` routes on the store ``get_store()`` returns (None: 404, projects disabled)."""
    router = APIRouter()
    no_store = {"Cache-Control": "no-store"}

    def store() -> ProjectStore:
        s = get_store()
        if s is None:
            raise HTTPException(status_code=404, detail="server-side projects are disabled (--studio-projects-dir)")
        return s

    async def call(fn, *args):
        try:
            return await asyncio.to_thread(fn, *args)
        except ProjectError as exc:
            return exc.response()

    @router.get("/projects")
    async def list_projects():
        result = await call(store().list)
        return result if isinstance(result, Response) else JSONResponse(result, headers=no_store)

    @router.post("/projects/gc")
    async def gc_projects(min_age_seconds: float = GC_MIN_AGE_SECONDS):
        if not min_age_seconds >= 0:
            raise HTTPException(status_code=400, detail="min_age_seconds must be >= 0")
        result = await call(store().gc, min_age_seconds)
        return result if isinstance(result, Response) else JSONResponse(result, headers=no_store)

    @router.get("/projects/{project_id}")
    async def get_project(project_id: str):
        result = await call(store().get_bytes, project_id)
        if isinstance(result, Response):
            return result
        return Response(result, media_type="application/json", headers=no_store)

    @router.put("/projects/{project_id}")
    async def put_project(project_id: str, request: Request):
        s = store()
        try:
            check_id(project_id)
            body = parse_json(await _read_body(request, s.max_project_bytes))
            if not isinstance(body, dict) or "data" not in body or "name" not in body:
                raise ProjectError(400, "body must be {name, data, base_revision}")
        except ProjectError as exc:
            return exc.response()
        result = await call(s.put, project_id, body["name"], body["data"], body.get("base_revision"))
        return result if isinstance(result, Response) else JSONResponse(result, headers=no_store)

    @router.delete("/projects/{project_id}")
    async def delete_project(project_id: str):
        result = await call(store().delete, project_id)
        return result if isinstance(result, Response) else JSONResponse({"deleted": project_id}, headers=no_store)

    @router.api_route("/media/{sha256}", methods=["GET", "HEAD"])
    async def get_media(sha256: str, request: Request):
        result = await call(store().media, sha256)
        if isinstance(result, Response):
            return result
        path, mime = result
        headers = {"ETag": f'"{sha256}"', "Cache-Control": _MEDIA_CACHE}
        if sha256 in request.headers.get("if-none-match", ""):
            return Response(status_code=304, headers=headers)
        return FileResponse(path, media_type=mime or "application/octet-stream", headers=headers)

    @router.put("/media/{sha256}")
    async def put_media(sha256: str, request: Request, type: str | None = None):
        s = store()
        length = request.headers.get("content-length")
        try:
            if length is not None and length.isdigit() and int(length) > s.max_media_bytes:
                raise ProjectError(413, f"media too large (> {s.max_media_bytes} bytes)")
            result = await s.put_media(sha256, request.stream(), type or request.headers.get("content-type"))
        except ProjectError as exc:
            return exc.response()
        return JSONResponse(result, status_code=201 if result["created"] else 200, headers=no_store)

    return router
