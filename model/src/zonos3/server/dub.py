"""The dubbing routes of the Python servers (/v2/dub/*) over the server's ``zonos3.dub.runtime.DubRuntime``."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import Response

from zonos3.server.edit import call
from zonos3.server.schemas import (
    DubAnalyzeRequest,
    DubGenerateRequest,
    DubRegenerateRequest,
    DubTranscribeRequest,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


async def upload(request: Request, name: str, directory: str) -> Path:
    """Stream a request body (a file named ``name``) into ``directory``; 413 past the dub's upload cap."""
    from zonos3.dub.plan import MAX_VIDEO_BYTES

    declared = request.headers.get("content-length")
    if declared is not None and int(declared) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")
    suffix = Path(name).suffix.lower()
    path = Path(directory) / f"input{suffix if suffix in VIDEO_SUFFIXES | AUDIO_SUFFIXES else '.bin'}"
    size = 0
    with open(path, "wb") as fh:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                raise HTTPException(status_code=413, detail="upload too large")
            fh.write(chunk)
    return path


def download_stem(name: str) -> str:
    """The file-name stem of a muxed download (``<stem>_dub.mp4``)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "video"


def make_router(get_runtime: Callable[[], "DubRuntime"], available: Callable[[], bool],  # noqa: F821
                unavailable: str) -> APIRouter:
    """``available()``: the server can dub (LACI heads and a speaking-rate condition row); ``unavailable``: the 400
    detail when it cannot."""
    router = APIRouter()

    def ready():
        if not available():
            raise HTTPException(status_code=400, detail=unavailable)
        return get_runtime()

    @router.get("/v2/dub/capabilities")
    async def capabilities():
        """Target languages, rate buckets, lip-sync, voice modes and limits of the dub (zonos3.dub.runtime)."""
        return {**(await asyncio.to_thread(get_runtime().capabilities)), "enabled": available()}

    @router.post("/v2/dub/ref")
    async def ref(request: Request, name: str = "audio"):
        """A dub reference: raw audio bytes (any sound file), or a video whose audio track is taken."""
        rt = ready()
        with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
            path = await upload(request, name, tmp)
            return await call(asyncio.to_thread(rt.add_ref, str(path)))

    @router.post("/v2/dub/lips/{ref_id}")
    async def lips(ref_id: str, request: Request, name: str = "video"):
        """Face track + lip embeddings of the reference's video, for generate / regenerate with lipsync (AV-sync)."""
        rt = ready()

        async def run():
            rt.refs.get(ref_id)  # 404 before reading the body
            with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
                video = await upload(request, name, tmp)
                return await asyncio.to_thread(rt.prepare_lips, ref_id, str(video))

        return await call(run())

    @router.post("/v2/dub/transcribe")
    async def transcribe(req: DubTranscribeRequest):
        return await call(ready().transcribe(req.ref_id, req.ref_lang))

    @router.post("/v2/dub/analyze")
    async def analyze(req: DubAnalyzeRequest):
        return await call(ready().analyze(req.model_dump()))

    @router.post("/v2/dub/generate")
    async def generate(req: DubGenerateRequest,
                       x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        return await call(ready().generate(req.model_dump(), x_tts_session_id))

    @router.post("/v2/dub/regenerate")
    async def regenerate(req: DubRegenerateRequest,
                         x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        return await call(ready().regenerate(req.model_dump(), x_tts_session_id))

    @router.post("/v2/dub/mux/{dub_id}")
    async def mux(dub_id: str, request: Request, name: str = "video"):
        """The video with the dub as its audio track (raw video bytes in, MP4 out)."""
        rt = ready()

        async def run():
            rt.dubs.get(dub_id)  # 404 before reading the body
            with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
                video, out = await upload(request, name, tmp), Path(tmp) / "out.mp4"
                await asyncio.to_thread(rt.mux, dub_id, str(video), str(out))
                return out.read_bytes()

        data = await call(run())
        return Response(data, media_type="video/mp4",
                        headers={"Content-Disposition": f'attachment; filename="{download_stem(name)}_dub.mp4"'})

    return router
