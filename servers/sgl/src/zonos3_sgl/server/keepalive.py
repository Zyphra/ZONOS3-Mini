"""Keep long JSON requests alive behind a proxy with a response deadline.

A Cloudflare quick tunnel answers 524 when the origin has not responded within 100 s, which a cold alignment
worker or a long dub on a busy node can exceed. A POST to one of ``paths`` with the request header
``X-Keepalive: 1`` therefore gets its 200 status line at once, a space every ``interval`` seconds while the
endpoint works, and then the endpoint's JSON body (leading whitespace is valid JSON). An endpoint error keeps its
detail but travels in the body, since the status line is already sent: ``{"detail": ..., "status_code": 4xx}``.
Requests without the header take the normal path.
"""
from __future__ import annotations

import asyncio
import json

KEEPALIVE_HEADER = b"x-keepalive"


class KeepaliveMiddleware:
    def __init__(self, app, paths: tuple[str, ...], interval: float = 10.0):
        self.app, self.paths, self.interval = app, tuple(paths), float(interval)

    def _applies(self, scope) -> bool:
        if scope["type"] != "http" or scope["method"] != "POST":
            return False
        path = scope["path"]
        if not any(path == p or path.startswith(p.rstrip("/") + "/") for p in self.paths):
            return False
        return dict(scope.get("headers") or ()).get(KEEPALIVE_HEADER) == b"1"

    async def __call__(self, scope, receive, send):
        if not self._applies(scope):
            await self.app(scope, receive, send)
            return
        start: dict = {}
        chunks: list[bytes] = []

        async def capture(message):
            if message["type"] == "http.response.start":
                start.update(message)
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        task = asyncio.ensure_future(self.app(scope, receive, capture))
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store"),
                                (b"x-keepalive", b"1")]})
        await send({"type": "http.response.body", "body": b" ", "more_body": True})
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=self.interval)
            if not done:
                await send({"type": "http.response.body", "body": b" ", "more_body": True})
        payload = b"".join(chunks)
        exc = task.exception()
        status = 500 if exc is not None else int(start.get("status", 500))
        if exc is not None:
            payload = json.dumps({"detail": f"{type(exc).__name__}: {exc}"}).encode()
        if status >= 400:
            try:
                detail = json.loads(payload).get("detail")
            except (ValueError, AttributeError):
                detail = payload.decode(errors="replace")
            payload = json.dumps({"detail": detail, "status_code": status}).encode()
        await send({"type": "http.response.body", "body": payload, "more_body": False})
