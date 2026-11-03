"""The FastAPI app of the MLX server: the v2 API of api/openapi.yaml, the access-token gate, CORS, the keepalive
wrapper, studio projects and the studio's production build at / (the sgl api_server app, route for route)."""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from zonos3.server.dub import make_router as dub_router
from zonos3.server.edit import make_router as edit_router
from zonos3.server.keepalive import KeepaliveMiddleware
from zonos3.server.projects import make_router as projects_router

from zonos3_mlx.server import capabilities, dub, edit, generate, speakers
from zonos3_mlx.server.state import ServerState

KEEPALIVE_PATHS = ("/v2/edit", "/v2/dub/ref", "/v2/dub/lips", "/v2/dub/transcribe", "/v2/dub/analyze",
                   "/v2/dub/generate", "/v2/dub/regenerate")
EXPOSED_HEADERS = ["X-Audio-Sample-Rate", "X-Audio-Channels", "X-Audio-Format", "X-Control-Span-Follow", "X-Seed"]


def create_app(state: ServerState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            state.close()

    # The spec of this API is api/openapi.yaml; FastAPI's generated docs would describe a second one.
    app = FastAPI(title="Zonos3 API Server (MLX)", version="2.0.0", lifespan=lifespan, openapi_url=None,
                  docs_url=None, redoc_url=None)
    app.state.zonos3 = state

    # Innermost, so CORS and the access check wrap its early status line. The /v2/edit prefix covers
    # /v2/edit/transcribe and /v2/edit/align.
    app.add_middleware(KeepaliveMiddleware, paths=KEEPALIVE_PATHS)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=EXPOSED_HEADERS)

    @app.middleware("http")
    async def private_access(request: Request, call_next):
        """ZONOS3_ACCESS_TOKEN: ``Authorization: Bearer <token>``, or the HttpOnly cookie a browser gets from
        ``/?access_token=<token>``."""
        token = os.environ.get("ZONOS3_ACCESS_TOKEN")
        if not token:
            return await call_next(request)
        candidate = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not candidate:
            candidate = request.cookies.get("zonos3_access", "")
        if candidate and secrets.compare_digest(candidate, token):
            response = await call_next(request)
        elif request.method == "GET" and request.url.path == "/" and (
                candidate := request.query_params.get("access_token", "")) and secrets.compare_digest(candidate, token):
            response = RedirectResponse("/", status_code=303)
            response.set_cookie("zonos3_access", token, httponly=True,
                                secure=request.headers.get("x-forwarded-proto") == "https", samesite="strict")
        else:
            response = Response("Access denied", status_code=403, media_type="text/plain")
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/v2/media/") and response.status_code in (200, 206, 304):
            response.headers["Cache-Control"] = "private, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-store"
        return response

    app.include_router(projects_router(lambda: state.projects), prefix="/v2")
    for module in (capabilities, speakers, generate):
        app.include_router(module.router(state))
    app.include_router(edit_router(lambda: edit.runtime(state)))
    app.include_router(dub_router(lambda: dub.runtime(state), lambda: capabilities.dub_available(state),
                                  dub.UNAVAILABLE))

    # Registered last, so every API route matches first: GET|HEAD / is the studio's index.html, GET|HEAD /<path> a
    # file inside the studio dir (never under /v1/ or /v2/); anything else is a 404. No SPA fallback. Pages (*.html)
    # are no-store, every other file immutable.
    @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
    async def serve_studio(request: Request, path: str = ""):
        if request.method not in ("GET", "HEAD") or path.split("/", 1)[0] in ("v1", "v2"):
            raise HTTPException(status_code=404, detail="Not Found")
        root = state.options.studio_dir
        if root is None:
            raise HTTPException(status_code=404, detail="Not Found")
        root = Path(root).resolve()
        if not path:
            index = root / "index.html"
            if not index.is_file():
                raise HTTPException(status_code=404, detail="The studio is not built (pnpm --dir studio build).")
            return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-store, max-age=0"})
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(status_code=404, detail="Not Found")
        cache = "no-store, max-age=0" if target.name.endswith(".html") else "public, max-age=31536000, immutable"
        return FileResponse(target, headers={"Cache-Control": cache})

    return app

