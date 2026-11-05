"""The HTTP mapping of runtime errors (speech editing, plugins): ``ValueError`` is the request's fault (400),
``NotFound`` a resource id no longer held (404), ``zonos3.tts.edit_align.WorkerError`` an unavailable helper worker
(503), ``RuntimeError`` a failed generation or decode (500)."""

from __future__ import annotations

from typing import Awaitable, TypeVar

from fastapi import HTTPException

T = TypeVar("T")


class NotFound(LookupError):
    """A resource id a runtime no longer holds (HTTP 404; str(exc) is the detail)."""


def status_of(exc: BaseException) -> int | None:
    """The HTTP status of a runtime error (None: not one of theirs)."""
    from zonos3.tts.edit_align import WorkerError

    if isinstance(exc, NotFound):
        return 404
    if isinstance(exc, WorkerError):
        return 503
    if isinstance(exc, ValueError):
        return 400
    if isinstance(exc, RuntimeError):
        return 500
    return None


async def call(awaitable: Awaitable[T]) -> T:
    """Await a runtime call; any exception with a status_of becomes HTTPException(status, str(exc))."""
    try:
        return await awaitable
    except Exception as exc:
        status = status_of(exc)
        if status is None:
            raise
        raise HTTPException(status_code=status, detail=str(exc)) from exc
