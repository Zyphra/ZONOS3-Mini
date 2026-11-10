"""The speech-editing routes of the Python servers (/v2/edit/transcribe, /v2/edit/align, /v2/edit) over the server's
``zonos3.tts.edit_runtime.EditRuntime``; the HTTP mapping of its errors is ``zonos3.server.errors``. An edit's
``decoder`` is checked against the server's decoders (``decoders()``: the ids of Capabilities.decoders) first."""

from __future__ import annotations

from typing import Callable, Sequence

from fastapi import APIRouter

from zonos3.server.errors import call
from zonos3.server.schemas import EditAlignRequest, EditRequest, EditTranscribeRequest
from zonos3.server.speech import resolve_decoder


def make_router(
    get_runtime: Callable[[], "EditRuntime"],  # noqa: F821
    decoders: Callable[[], Sequence[str]],
) -> APIRouter:
    router = APIRouter()

    @router.post("/v2/edit/transcribe")
    async def transcribe(req: EditTranscribeRequest):
        """Qwen3-ASR transcript of the audio to edit (chunk-wise past 30 s) and its words' FuseAlign timings; 503
        without speech recognition."""
        return await call(get_runtime().transcribe(req))

    @router.post("/v2/edit/align")
    async def align(req: EditAlignRequest):
        """FuseAlign forced alignment of the audio to ``text``: word timings, presences (``confidence``) and edit
        tokens."""
        return await call(get_runtime().align(req))

    @router.post("/v2/edit")
    async def edit(req: EditRequest):
        """Regenerate the edited regions of recorded speech; unedited codec frames are kept exactly."""

        async def run():
            resolve_decoder(req.decoder, decoders())  # 400 for an id this server does not serve
            return await get_runtime().edit(req)

        return await call(run())

    return router
