"""The HTTP middleware of the Python servers: CORS for studio clients on another origin and the optional access token
(ZONOS3_ACCESS_TOKEN)."""

from __future__ import annotations

import os
import secrets

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response

# Studio clients on another origin (?server=) read these response headers.
EXPOSED_HEADERS: tuple[str, ...] = (
    "X-Audio-Sample-Rate",
    "X-Audio-Channels",
    "X-Audio-Format",
    "X-Control-Span-Follow",
    "X-Seed",
)


def install_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=list(EXPOSED_HEADERS),
    )


def access_log_enabled() -> bool:
    """Request logs would record the access_token query of a private server."""
    return not os.environ.get("ZONOS3_ACCESS_TOKEN")


def install_private_access(app: FastAPI) -> None:
    @app.middleware("http")
    async def optional_private_access(request: Request, call_next):
        """Keep the inference UI and API private when ZONOS3_ACCESS_TOKEN is set.

        Clients send ``Authorization: Bearer <token>``; a browser opens
        ``/?access_token=<token>`` once and then carries an HttpOnly cookie.
        """
        token = os.environ.get("ZONOS3_ACCESS_TOKEN")
        if not token:
            return await call_next(request)
        candidate = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not candidate:
            candidate = request.cookies.get("zonos3_access", "")
        if candidate and secrets.compare_digest(candidate, token):
            response = await call_next(request)
        elif (
            request.method == "GET"
            and request.url.path == "/"
            and (candidate := request.query_params.get("access_token", ""))
            and secrets.compare_digest(candidate, token)
        ):
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(
                "zonos3_access",
                token,
                httponly=True,
                secure=request.headers.get("x-forwarded-proto") == "https",
                samesite="strict",
            )
        else:
            response = Response("Access denied", status_code=403, media_type="text/plain")
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/v2/media/") and response.status_code in (200, 206, 304):
            # content-addressed and immutable: the browser may keep it (private: never a shared cache)
            response.headers["Cache-Control"] = "private, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-store"
        return response
