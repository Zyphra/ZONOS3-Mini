#!/usr/bin/env python3
"""The ggml server's Python host: speech editing (``zonos3.tts.edit_runtime.EditRuntime``) and the server plugins
(``zonos3.server.plugins``, ``--plugins``) in a persistent process whose model is the server, reached over the same
pipe.

Runs with the workspace interpreter and ``PYTHONPATH=<model/src>`` (plugins installed in that interpreter). stdin and
stdout carry one JSON object per line (UTF-8) in both directions, multiplexed: every request runs as its own asyncio
task, so many are in flight and answer in any order. Stray output of the libraries goes to stderr (fd 1 points there;
the protocol writes on a dup of the real stdout). The host exits when stdin closes, after closing its plugins and its
ASR and FuseAlign workers.

Requests ``{"id", "op", ...}`` are answered ``{"id", "status": "ok", "result"}`` or ``{"id", "status": "error",
"code", "message"}``: the HTTP status and detail of ``zonos3.server.errors.call``; a body the schema
(``zonos3.server.schemas``) rejects is 422 with FastAPI's detail list as ``message``; any other failure is 500 with
``str(exc)`` (traceback on stderr).

Ops:
  init {edit: EditModel fields | null, host: HostFacts JSON | null, plugins: [name]}
                                                  -> {plugins: [{name, keepalive_paths}]}
  edit_transcribe {request: EditTranscribeRequest}                                    -> the /v2/edit/transcribe body
  edit_align {request: EditAlignRequest}                                              -> the /v2/edit/align body
  edit {request: EditRequest}                                                         -> the /v2/edit body
  http {plugin, method, path, query, headers: [[k, v]], body: path | null, out: path} -> {status, headers: [[k, v]]}

``init`` comes first after every start; it builds the edit runtime over this process's own ``AsrWorker`` and
``FuseAlignWorker`` (ZONOS3_ASR_PYTHON, ZONOS3_ASR_DEVICE, ZONOS3_FUSEALIGN_DEVICE) and loads and mounts the plugins
(``host``: ``zonos3.server.plugins.HostFacts.to_json``); a re-init closes the previous plugins. The ASR worker is
optional: without its interpreter the runtime aligns and edits clips sent with their transcript and answers the
requests that need speech recognition 503 (``zonos3.tts.edit_align.AsrUnavailable``). ``http`` runs one
request of a plugin's routes: ``query`` is the encoded query string, ``body`` a file of the request body, and the
response body is written to ``out``; error statuses are ``ok`` replies with that status.

While a request runs, the model's work goes back to the server as engine calls ``{"call": N, "op", ...}`` (N unique per
process), answered ``{"call": N, "status": "ok", ...}`` or ``{"call": N, "status": "error", "code", "message"}``
(code 400: ValueError, else RuntimeError, with the message):

  decode_audio {audio, out}                       -> {sample_rate}; float32 mono WAV at out
  embed {audio, max_seconds}                      -> {embedding: [float]}
  encode {audio, max_seconds}                     -> {codes: [[int] * K] * T}
  check {target_text, doses, condition_buckets, condition_values}                   -> {raw: [id]}
  fill {every FillJob field}                      -> {rows: [[int] * K] * n, eos_frame (-1: none)}
  decode {codes, out}                             -> {}; float32 44.1 kHz WAV at out
  laci_takes {takes: [{request: GenerateRequest body, target_frames, adjust_rate,
              scale: {edges, first, bucket, doses: {"<bucket>": dose}} | null, out}], session}
                                                  -> {takes: [{audio: bool, record: LACI record}]}; WAV at out if audio

Audio crosses as file paths (same host): ``audio`` is a file of the upload's bytes (``encode`` of a plugin's reference
clip: a float WAV of the clip); the host creates and removes the temp dirs of its engine calls.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import tempfile
import threading
import traceback
from typing import TextIO

import numpy as np
import soundfile as sf

BODY_CHUNK = 1 << 20


class InvalidBody(Exception):
    """A request body the schema rejects (HTTP 422); ``errors`` is FastAPI's detail list."""

    def __init__(self, errors: list[dict]):
        super().__init__("invalid request body")
        self.errors = errors


def validate(schema, body):
    """``body`` as a ``zonos3.server.schemas`` model (InvalidBody, never ValueError, when it does not fit)."""
    from pydantic import ValidationError

    try:
        return schema.model_validate(body)
    except ValidationError as exc:
        errors = json.loads(exc.json(include_url=False))
        raise InvalidBody([{**e, "loc": ["body", *e["loc"]]} for e in errors]) from None


class Pipe:
    """The protocol side of stdout (writes serialized) and the engine calls awaiting their answers."""

    def __init__(self, out: TextIO):
        self._out = out
        self._lock = threading.Lock()
        self._calls: dict[int, asyncio.Future] = {}
        self._next_call = 0
        self._closed = False

    def send(self, message: dict) -> None:
        line = json.dumps(message, ensure_ascii=False) + "\n"
        with self._lock:
            self._out.write(line)
            self._out.flush()

    async def call(self, op: str, **payload) -> dict:
        """One engine call; ValueError (code 400) or RuntimeError with the server's message when it fails."""
        if self._closed:
            raise RuntimeError("the server closed the Python host's pipe")
        self._next_call += 1
        call = self._next_call
        future = asyncio.get_running_loop().create_future()
        self._calls[call] = future
        try:
            self.send({"call": call, "op": op, **payload})
            reply = await future
        finally:
            self._calls.pop(call, None)
        if reply.get("status") != "ok":
            message = reply.get("message") or f"engine call {op} failed"
            raise (ValueError if reply.get("code") == 400 else RuntimeError)(message)
        return reply

    def answer(self, reply: dict) -> None:
        future = self._calls.get(reply.get("call"))
        if future is None or future.done():
            print(f"Python host: answer to no pending engine call: {reply.get('call')!r}", file=sys.stderr)
            return
        future.set_result(reply)

    def close(self) -> None:
        """Fail every pending engine call (stdin closed: no answer can come)."""
        self._closed = True
        for future in self._calls.values():
            if not future.done():
                future.set_exception(RuntimeError("the server closed the Python host's pipe"))


def read_wav(path: str) -> np.ndarray:
    return sf.read(path, dtype="float32")[0]


def write_upload(directory: str, audio: bytes) -> str:
    path = os.path.join(directory, "upload.bin")
    with open(path, "wb") as fh:
        fh.write(audio)
    return path


class RpcEditEngine:
    """``zonos3.tts.edit_runtime.EditEngine`` over the server's engine calls."""

    def __init__(self, pipe: Pipe, model):
        self.pipe = pipe
        self.model = model

    async def decode_audio(self, audio: bytes) -> tuple[np.ndarray, int]:
        with tempfile.TemporaryDirectory(prefix="zonos3-pyhost-") as tmp:
            out = os.path.join(tmp, "decoded.wav")
            reply = await self.pipe.call("decode_audio", audio=write_upload(tmp, audio), out=out)
            return read_wav(out), int(reply["sample_rate"])

    async def embed(self, audio: bytes) -> list[float]:
        from zonos3.generation import EDIT_MAX_SECONDS

        with tempfile.TemporaryDirectory(prefix="zonos3-pyhost-") as tmp:
            reply = await self.pipe.call("embed", audio=write_upload(tmp, audio), max_seconds=EDIT_MAX_SECONDS)
        return [float(x) for x in reply["embedding"]]

    async def encode(self, audio: bytes, max_seconds: float) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="zonos3-pyhost-") as tmp:
            reply = await self.pipe.call("encode", audio=write_upload(tmp, audio), max_seconds=max_seconds)
        return np.asarray(reply["codes"], dtype=np.int64).reshape(-1, self.model.n_codebooks)

    async def check(self, target_text: str, doses, condition_buckets, condition_values) -> set[str]:
        reply = await self.pipe.call(
            "check",
            target_text=target_text,
            doses=doses,
            condition_buckets=condition_buckets,
            condition_values=condition_values,
        )
        return set(reply["raw"])

    async def fill(self, job) -> tuple[np.ndarray, int]:
        payload = {f.name: getattr(job, f.name) for f in dataclasses.fields(job)}
        payload["window_codes"] = np.asarray(job.window_codes).tolist()
        payload["fills"] = [np.asarray(fill).tolist() for fill in job.fills]
        reply = await self.pipe.call("fill", **payload)
        return np.asarray(reply["rows"], dtype=np.int64).reshape(-1, self.model.n_codebooks), int(reply["eos_frame"])

    async def decode(self, codes: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="zonos3-pyhost-") as tmp:
            out = os.path.join(tmp, "decoded.wav")
            await self.pipe.call("decode", codes=np.asarray(codes).tolist(), out=out)
            return read_wav(out)


class RpcHost:
    """``zonos3.server.plugins.Host`` over the server's engine calls."""

    def __init__(self, pipe: Pipe, facts, asr, aligner):
        self.pipe = pipe
        self._facts = facts
        self.asr = asr
        self.aligner = aligner

    def facts(self):
        return self._facts

    async def synthesize_laci(self, takes, session_id: str | None) -> list:
        with tempfile.TemporaryDirectory(prefix="zonos3-pyhost-") as tmp:
            outs = [os.path.join(tmp, f"take{k}.wav") for k in range(len(takes))]
            reply = await self.pipe.call(
                "laci_takes", session=session_id, takes=[take.to_json(out) for take, out in zip(takes, outs)]
            )
            return [
                (read_wav(out) if take["audio"] else None, take["record"])
                for take, out in zip(reply["takes"], outs, strict=True)
            ]

    async def encode_reference(self, audio: np.ndarray, sample_rate: int, max_seconds: float) -> list[list[int]]:
        with tempfile.TemporaryDirectory(prefix="zonos3-pyhost-") as tmp:
            path = os.path.join(tmp, "reference.wav")
            sf.write(path, audio, sample_rate, subtype="FLOAT")
            return (await self.pipe.call("encode", audio=path, max_seconds=max_seconds))["codes"]


async def asgi_request(
    app, method: str, path: str, query: str, headers: list[list[str]], body_path: str | None, out_path: str
) -> tuple[int, list[list[str]]]:
    """Run one HTTP request through the ASGI ``app``: the body streamed from ``body_path`` in 1 MiB chunks, the
    response body written to ``out_path``. Returns the status and response headers."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": (query or "").encode("latin-1"),
        "root_path": "",
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers],
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 80),
    }
    body = open(body_path, "rb") if body_path else None
    start: dict = {}
    done = asyncio.Event()
    received = False

    async def receive() -> dict:
        nonlocal received
        if not received:
            chunk = body.read(BODY_CHUNK) if body is not None else b""
            received = len(chunk) < BODY_CHUNK
            return {"type": "http.request", "body": chunk, "more_body": not received}
        await done.wait()
        return {"type": "http.disconnect"}

    with open(out_path, "wb") as out:

        async def send(message: dict) -> None:
            if message["type"] == "http.response.start":
                start["status"] = int(message["status"])
                start["headers"] = [[k.decode("latin-1"), v.decode("latin-1")] for k, v in message.get("headers", ())]
            elif message["type"] == "http.response.body":
                out.write(message.get("body", b""))
                if not message.get("more_body", False):
                    done.set()

        try:
            await app(scope, receive, send)
        except Exception:
            if "status" not in start:  # nothing sent: the app's error is this request's 500
                raise
            traceback.print_exc()
        finally:
            done.set()
            if body is not None and not body.closed:
                body.close()
    if "status" not in start:
        raise RuntimeError(f"{method} {path}: the app sent no response")
    return start["status"], start["headers"]


class Worker:
    def __init__(self, pipe: Pipe):
        self.pipe = pipe
        self.asr = self.aligner = None
        self.edit = None
        self.plugins: list = []
        self.app = None

    # ---- runtimes ---------------------------------------------------------------------------------------------

    async def init(self, edit: dict | None, host: dict | None, names: list[str] | None) -> dict:
        from fastapi import FastAPI

        from zonos3.server import plugins
        from zonos3.tts.edit_align import AsrWorker, FuseAlignWorker
        from zonos3.tts.edit_runtime import EditModel, EditRuntime

        if self.asr is None:
            self.asr, self.aligner = AsrWorker(), FuseAlignWorker()
        engine = None
        if edit is not None:
            engine = RpcEditEngine(self.pipe, EditModel(**{**edit, "codebook_sizes": tuple(edit["codebook_sizes"])}))
        self.edit = None if engine is None else EditRuntime(engine, self.asr, self.aligner)
        await asyncio.to_thread(self.close_plugins)
        names = list(names or ())
        if names and host is None:
            raise ValueError("init: plugins need the host facts")
        loaded = plugins.load(names)
        app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
        if loaded:
            rpc = RpcHost(self.pipe, plugins.HostFacts.from_json(host), self.asr, self.aligner)
            try:
                plugins.mount(app, loaded, rpc, [])
            except Exception:
                for plugin in loaded:
                    plugin.close()
                raise
        self.plugins, self.app = loaded, app
        return {"plugins": [{"name": p.name, "keepalive_paths": list(p.keepalive_paths)} for p in loaded]}

    def edit_runtime(self):
        if self.edit is None:
            raise RuntimeError("speech editing is not initialized")
        return self.edit

    async def http(self, request: dict) -> dict:
        name = request.get("plugin")
        if not any(p.name == name for p in self.plugins):
            raise ValueError(f"plugin {name!r} is not loaded")
        status, headers = await asgi_request(
            self.app,
            request["method"],
            request["path"],
            request.get("query") or "",
            request.get("headers") or [],
            request.get("body"),
            request["out"],
        )
        return {"status": status, "headers": headers}

    def close_plugins(self) -> None:
        plugins, self.plugins, self.app = self.plugins, [], None
        for plugin in plugins:
            try:
                plugin.close()
            except Exception:  # noqa: BLE001 - close the rest
                traceback.print_exc()

    def close(self) -> None:
        self.close_plugins()
        for worker in (self.asr, self.aligner):
            if worker is not None:
                worker.close()

    # ---- requests ---------------------------------------------------------------------------------------------

    async def run(self, request: dict):
        from zonos3.server import schemas as S

        op = request.get("op")
        if op == "init":
            return await self.init(request.get("edit"), request.get("host"), request.get("plugins"))
        if op == "edit_transcribe":
            return await self.edit_runtime().transcribe(validate(S.EditTranscribeRequest, request.get("request")))
        if op == "edit_align":
            return await self.edit_runtime().align(validate(S.EditAlignRequest, request.get("request")))
        if op == "edit":
            return await self.edit_runtime().edit(validate(S.EditRequest, request.get("request")))
        if op == "http":
            return await self.http(request)
        raise ValueError(f"unknown op {op!r}")

    async def respond(self, request: dict) -> None:
        """Run one request and send its answer (errors as ``zonos3.server.errors.call`` maps them)."""
        from fastapi import HTTPException

        from zonos3.server.errors import call

        rid = request.get("id")
        try:
            result = await call(self.run(request))
            reply = {"id": rid, "status": "ok", "result": result}
        except HTTPException as exc:
            reply = {"id": rid, "status": "error", "code": exc.status_code, "message": exc.detail}
        except InvalidBody as exc:
            reply = {"id": rid, "status": "error", "code": 422, "message": exc.errors}
        except Exception as exc:  # noqa: BLE001 - one failed request must not kill the worker
            traceback.print_exc()
            reply = {"id": rid, "status": "error", "code": 500, "message": str(exc)}
        try:
            self.pipe.send(reply)
        except (TypeError, ValueError) as exc:  # an unserializable result fails this request, not the pipe
            traceback.print_exc()
            self.pipe.send({"id": rid, "status": "error", "code": 500, "message": str(exc)})


async def serve(lines: TextIO, out: TextIO) -> None:
    """Answer the requests of ``lines`` on ``out`` until ``lines`` ends; engine answers come on ``lines`` too."""
    loop = asyncio.get_running_loop()
    inbox: asyncio.Queue[str | None] = asyncio.Queue()

    def read() -> None:  # blocking reads off the event loop
        for line in lines:
            loop.call_soon_threadsafe(inbox.put_nowait, line)
        loop.call_soon_threadsafe(inbox.put_nowait, None)

    threading.Thread(target=read, name="pyhost-stdin", daemon=True).start()
    pipe = Pipe(out)
    worker = Worker(pipe)
    tasks: set[asyncio.Task] = set()
    while (line := await inbox.get()) is not None:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("a message must be a JSON object")
        except ValueError as exc:
            pipe.send({"id": None, "status": "error", "code": 400, "message": f"invalid message: {exc}"})
            continue
        if "call" in message:
            pipe.answer(message)
            continue
        task = asyncio.create_task(worker.respond(message))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    pipe.close()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.to_thread(worker.close)


def main() -> None:
    # Keep stdout for the protocol only: library prints go to stderr.
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    sys.stdin.reconfigure(encoding="utf-8")
    asyncio.run(serve(sys.stdin, protocol))


if __name__ == "__main__":
    main()
