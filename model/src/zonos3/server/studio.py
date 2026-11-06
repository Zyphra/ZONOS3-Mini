"""Serving the studio's production build (studio/dist) at /: the cache rule and the catch-all route of the servers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from zonos3.server.plugins import CATCH_ALL

NOT_BUILT = "The studio is not built (pnpm --dir studio build)."


def studio_cache_control(rel: str) -> str:
    """Cache-Control of a studio file by its path relative to the studio dir: hashed ``assets/`` are immutable; pages,
    the service worker and the manifest are never stored (a rebuild is picked up at once); every other unhashed file
    (icons, the in-browser engine, theme-boot.js) is revalidated."""
    if rel.startswith("assets/"):
        return "public, max-age=31536000, immutable"
    if rel.endswith(".html") or rel in ("sw.js", "manifest.webmanifest"):
        return "no-store, max-age=0"
    return "no-cache"


def resolve_studio_file(root: Path, path: str) -> Path | None:
    """The studio file a request path names: ``<root>/<path>``, else the page ``<root>/<path>.html``; None outside the
    studio dir or missing."""
    for target in ((root / path).resolve(), (root / f"{path}.html").resolve()):
        if target.is_relative_to(root) and target.is_file():
            return target
    return None


def install_studio(app: FastAPI, root: Callable[[], Path | None]) -> None:
    """The studio catch-all (install it after every other route; plugins.mount keeps it last): GET|HEAD / is the
    studio's index.html (the chooser); GET|HEAD /<path> a file inside the studio dir, else the page <path>.html
    (/studio, /lite; never under /v1/ or /v2/); anything else is a 404. No SPA fallback: the studio routes with the URL
    hash."""

    @app.api_route(CATCH_ALL, methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
    async def serve_studio(request: Request, path: str = ""):
        if request.method not in ("GET", "HEAD") or path.split("/", 1)[0] in ("v1", "v2"):
            raise HTTPException(status_code=404, detail="Not Found")
        directory = root()
        directory = None if directory is None else Path(directory).resolve()
        if not path:
            index = None if directory is None else directory / "index.html"
            if index is None or not index.is_file():
                raise HTTPException(status_code=404, detail=NOT_BUILT)
            return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-store, max-age=0"})
        target = None if directory is None else resolve_studio_file(directory, path)
        if target is None:
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(target, headers={"Cache-Control": studio_cache_control(target.relative_to(directory)
                                                                                   .as_posix())})
