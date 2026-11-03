"""The speech-editing routes of the Python servers (/v2/edit/transcribe, /v2/edit/align, /v2/edit) over the server's
``zonos3.tts.edit_runtime.EditRuntime``, and the HTTP mapping of edit and dub errors."""

from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

from fastapi import APIRouter, HTTPException

from zonos3.server.schemas import EditAlignRequest, EditRequest, EditTranscribeRequest

T = TypeVar("T")


async def call(awaitable: Awaitable[T]) -> T:
    """Await an edit / dub runtime call, mapping its errors to HTTP statuses (zonos3.tts.edit_runtime.status_of)."""
    from zonos3.dub.runtime import AlignWorkerError, UnknownId
    from zonos3.tts.edit_runtime import status_of

    try:
        return await awaitable
    except UnknownId:
        raise HTTPException(status_code=404, detail="unknown ref_id/dub_id; upload or generate again") from None
    except AlignWorkerError as exc:
        raise HTTPException(status_code=503, detail=f"alignment worker: {exc}") from exc
    except Exception as exc:
        status = status_of(exc)
        if status is None:
            raise
        raise HTTPException(status_code=status, detail=str(exc)) from exc


def make_router(get_runtime: Callable[[], "EditRuntime"]) -> APIRouter:  # noqa: F821
    router = APIRouter()

    @router.post("/v2/edit/transcribe")
    async def transcribe(req: EditTranscribeRequest):
        """Qwen3-ASR transcript of the audio to edit (chunk-wise past 30 s) and its words' FuseAlign timings."""
        return await call(get_runtime().transcribe(req))

    @router.post("/v2/edit/align")
    async def align(req: EditAlignRequest):
        """FuseAlign forced alignment of the audio to ``text``: word timings, presences (``confidence``) and edit
        tokens."""
        return await call(get_runtime().align(req))

    @router.post("/v2/edit")
    async def edit(req: EditRequest):
        """Regenerate the edited regions of recorded speech; unedited codec frames are kept exactly."""
        return await call(get_runtime().edit(req))

    return router
