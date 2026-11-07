import asyncio
import json

from fastapi import FastAPI, HTTPException
from zonos3.server.keepalive import KeepaliveMiddleware


def _app(interval: float = 0.02):
    app = FastAPI()
    app.add_middleware(KeepaliveMiddleware, paths=("/slow", "/fail"), interval=interval)

    @app.post("/slow")
    async def slow():
        await asyncio.sleep(0.1)
        return {"ok": True}

    @app.post("/fail")
    async def fail():
        await asyncio.sleep(0.05)
        raise HTTPException(status_code=422, detail="bad clip")

    @app.post("/other")
    async def other():
        raise HTTPException(status_code=400, detail="plain")

    return app


def _post(path: str, keepalive: bool) -> tuple[int, dict, list[bytes]]:
    """One POST through the ASGI app: (status, headers, body chunks)."""
    headers = [(b"content-type", b"application/json")] + ([(b"x-keepalive", b"1")] if keepalive else [])
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(_app()(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    chunks = [m.get("body", b"") for m in sent if m["type"] == "http.response.body"]
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}, chunks


def test_status_at_once_heartbeats_then_the_json_body():
    status, headers, chunks = _post("/slow", keepalive=True)
    assert status == 200 and headers["x-keepalive"] == "1"
    assert chunks[0] == b" " and chunks[1] == b" "  # sent before the endpoint finished
    assert json.loads(b"".join(chunks)) == {"ok": True}


def test_an_endpoint_error_travels_in_the_body():
    status, _, chunks = _post("/fail", keepalive=True)
    assert status == 200
    body = json.loads(b"".join(chunks))
    assert body["status_code"] == 422 and isinstance(body["detail"], str)


def test_without_the_header_or_outside_the_paths_nothing_changes():
    status, _, chunks = _post("/fail", keepalive=False)
    assert status == 422 and set(json.loads(b"".join(chunks))) == {"detail"}
    status, _, chunks = _post("/other", keepalive=True)
    assert status == 400 and set(json.loads(b"".join(chunks))) == {"detail"}
