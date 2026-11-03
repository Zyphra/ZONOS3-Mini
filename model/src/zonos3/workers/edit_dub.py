#!/usr/bin/env python3
"""Speech editing and dubbing of the ggml server: ``zonos3.tts.edit_runtime.EditRuntime`` and
``zonos3.dub.runtime.DubRuntime`` in a persistent process whose model is the server, reached over the same pipe.

Runs with the workspace interpreter and ``PYTHONPATH=<model/src>``. stdin and stdout carry one JSON object per line
(UTF-8) in both directions, multiplexed: every request runs as its own asyncio task, so many are in flight and answer
in any order. Stray output of the libraries goes to stderr (fd 1 points there; the protocol writes on a dup of the
real stdout). The worker exits when stdin closes, after closing its ASR, FuseAlign and dubbing workers.

Requests ``{"id", "op", ...}`` are answered ``{"id", "status": "ok", "result"}`` or ``{"id", "status": "error",
"code", "message"}``: the HTTP status and detail of ``zonos3.server.edit.call``; a body the schema
(``zonos3.server.schemas``) rejects is 422 with FastAPI's detail list as ``message``; any other failure is 500 with
``str(exc)`` (traceback on stderr).

Ops:
  init {edit: EditModel fields | null, dub: {name, rate_edges, n_buckets, laci_languages, studio_voices, source_voice,
        raw_controls, phoneme_groups, rate_extension: {direction, row_edges, below, above,
        doses: {tts_lang: {"<bucket>": dose}}, reference_sides} | null} | null}       -> {}
  edit_transcribe {request: EditTranscribeRequest}                                    -> the /v2/edit/transcribe body
  edit_align {request: EditAlignRequest}                                              -> the /v2/edit/align body
  edit {request: EditRequest}                                                         -> the /v2/edit body
  dub_capabilities                                                                    -> DubRuntime.capabilities()
  dub_ref {path}                                                                      -> DubRuntime.add_ref
  dub_lips {ref_id, video}                                                            -> DubRuntime.prepare_lips
  dub_transcribe {ref_id, ref_lang}                                                   -> DubRuntime.transcribe
  dub_analyze {request: DubAnalyzeRequest}                                            -> DubRuntime.analyze
  dub_generate {request: DubGenerateRequest, session}                                 -> DubRuntime.generate
  dub_regenerate {request: DubRegenerateRequest, session}                             -> DubRuntime.regenerate
  dub_mux {dub_id, video, out}                                                        -> {} (the MP4 written to out)

``init`` comes first after every start; it builds the runtimes over this process's own ``AsrWorker`` and
``FuseAlignWorker`` (ZONOS3_ASR_PYTHON, ZONOS3_ASR_DEVICE, ZONOS3_FUSEALIGN_DEVICE) and ``WorkerSettings.from_env()``.

While a request runs, the model's work goes back to the server as engine calls ``{"call": N, "op", ...}`` (N unique per
process), answered ``{"call": N, "status": "ok", ...}`` or ``{"call": N, "status": "error", "code", "message"}``
(code 400: ValueError, else RuntimeError, with the message):

  decode_audio {audio, out}                       -> {sample_rate}; float32 mono WAV at out
  embed {audio, max_seconds}                      -> {embedding: [float]}
  encode {audio, max_seconds}                     -> {codes: [[int] * K] * T}
  check {target_text, doses, condition_buckets, condition_values}                   -> {raw: [id]}
  fill {every FillJob field}                      -> {rows: [[int] * K] * n, eos_frame (-1: none)}
  decode {codes, out}                             -> {}; float32 44.1 kHz WAV at out
  synthesize {jobs: [{text, language, seed, max_tokens, bucket, target_frames, adjust_rate, voice, controls, ref_codes,
              rate_scale: {edges, first, doses: {"<bucket>": dose}} | null, out}], session}
                                                  -> {takes: [{audio: bool, record: LACI record}]}; WAV at out if audio

Audio crosses as file paths (same host): ``audio`` is a file of the upload's bytes (``encode`` of a dub voice prompt:
a float WAV of the clip); the worker creates and removes the temp dirs of its engine calls.
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
            raise RuntimeError("the server closed the edit/dub worker's pipe")
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
            print(f"edit/dub worker: answer to no pending engine call: {reply.get('call')!r}", file=sys.stderr)
            return
        future.set_result(reply)

    def close(self) -> None:
        """Fail every pending engine call (stdin closed: no answer can come)."""
        self._closed = True
        for future in self._calls.values():
            if not future.done():
                future.set_exception(RuntimeError("the server closed the edit/dub worker's pipe"))


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
        with tempfile.TemporaryDirectory(prefix="zonos3-edit-dub-") as tmp:
            out = os.path.join(tmp, "decoded.wav")
            reply = await self.pipe.call("decode_audio", audio=write_upload(tmp, audio), out=out)
            return read_wav(out), int(reply["sample_rate"])

    async def embed(self, audio: bytes) -> list[float]:
        from zonos3.tts.edit_runtime import EDIT_MAX_SECONDS

        with tempfile.TemporaryDirectory(prefix="zonos3-edit-dub-") as tmp:
            reply = await self.pipe.call("embed", audio=write_upload(tmp, audio), max_seconds=EDIT_MAX_SECONDS)
        return [float(x) for x in reply["embedding"]]

    async def encode(self, audio: bytes, max_seconds: float) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="zonos3-edit-dub-") as tmp:
            reply = await self.pipe.call("encode", audio=write_upload(tmp, audio), max_seconds=max_seconds)
        return np.asarray(reply["codes"], dtype=np.int64).reshape(-1, self.model.n_codebooks)

    async def check(self, target_text: str, doses, condition_buckets, condition_values) -> set[str]:
        reply = await self.pipe.call("check", target_text=target_text, doses=doses,
                                     condition_buckets=condition_buckets, condition_values=condition_values)
        return set(reply["raw"])

    async def fill(self, job) -> tuple[np.ndarray, int]:
        payload = {f.name: getattr(job, f.name) for f in dataclasses.fields(job)}
        payload["window_codes"] = np.asarray(job.window_codes).tolist()
        payload["fills"] = [np.asarray(fill).tolist() for fill in job.fills]
        reply = await self.pipe.call("fill", **payload)
        return np.asarray(reply["rows"], dtype=np.int64).reshape(-1, self.model.n_codebooks), int(reply["eos_frame"])

    async def decode(self, codes: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="zonos3-edit-dub-") as tmp:
            out = os.path.join(tmp, "decoded.wav")
            await self.pipe.call("decode", codes=np.asarray(codes).tolist(), out=out)
            return read_wav(out)


def take_job(job: dict) -> dict:
    """A DubRuntime take job as the ``synthesize`` call sends it (the rate scale as ``{edges, first, doses}``)."""
    scale = job["rate_scale"]
    if scale is not None:
        scale = {"edges": [float(e) for e in scale.edges], "first": int(scale.first),
                 "doses": {str(b): float(d) for b, d in scale.doses.items()}}
    return {**job, "rate_scale": scale}


def dub_model(dub: dict):
    """The ``zonos3.dub.runtime.DubModel`` of the ``init`` payload's ``dub``."""
    from zonos3.dub.runtime import DubModel, rate_scales
    from zonos3.tts.steering import RateExtension

    ext = dub.get("rate_extension")
    if ext is not None:
        ext = RateExtension(direction=ext["direction"], row_edges=tuple(float(e) for e in ext["row_edges"]),
                            below=tuple(float(e) for e in ext["below"]), above=tuple(float(e) for e in ext["above"]),
                            doses={lang: {int(b): float(d) for b, d in doses.items()}
                                   for lang, doses in ext["doses"].items()},
                            reference_sides=tuple(ext["reference_sides"]))
    rate_edges = tuple(float(e) for e in dub["rate_edges"])
    return DubModel(name=dub["name"], rate_edges=rate_edges, n_buckets=int(dub["n_buckets"]),
                    laci_languages=frozenset(dub["laci_languages"]), studio_voices=bool(dub["studio_voices"]),
                    source_voice=bool(dub["source_voice"]), raw_controls=frozenset(dub["raw_controls"]),
                    phoneme_groups=bool(dub["phoneme_groups"]), rate_scales=rate_scales(ext, rate_edges))


class Worker:
    def __init__(self, pipe: Pipe):
        self.pipe = pipe
        self.asr = self.aligner = None
        self.edit = self.dub = None

    # ---- runtimes ---------------------------------------------------------------------------------------------

    async def init(self, edit: dict | None, dub: dict | None) -> dict:
        from zonos3.dub.plan import VOICE_PROMPT_MAX_S
        from zonos3.dub.runtime import DubRuntime, WorkerSettings
        from zonos3.tts.edit_align import AsrWorker, FuseAlignWorker
        from zonos3.tts.edit_runtime import EditModel, EditRuntime

        if self.asr is None:
            self.asr, self.aligner = AsrWorker(), FuseAlignWorker()
        engine = None
        if edit is not None:
            engine = RpcEditEngine(self.pipe, EditModel(**{**edit, "codebook_sizes": tuple(edit["codebook_sizes"])}))
        self.edit = None if engine is None else EditRuntime(engine, self.asr, self.aligner)
        if self.dub is not None:
            await asyncio.to_thread(self.dub.close)
            self.dub = None
        if dub is not None:
            pipe = self.pipe

            async def synthesize(jobs: list[dict], session_id: str | None) -> list[tuple[np.ndarray | None, dict]]:
                with tempfile.TemporaryDirectory(prefix="zonos3-edit-dub-") as tmp:
                    outs = [os.path.join(tmp, f"take{k}.wav") for k in range(len(jobs))]
                    reply = await pipe.call("synthesize", session=session_id,
                                            jobs=[{**take_job(job), "out": out} for job, out in zip(jobs, outs)])
                    return [(read_wav(out) if take["audio"] else None, take["record"])
                            for take, out in zip(reply["takes"], outs, strict=True)]

            async def encode_voice(clip: np.ndarray, sample_rate: int) -> list[list[int]]:
                with tempfile.TemporaryDirectory(prefix="zonos3-edit-dub-") as tmp:
                    path = os.path.join(tmp, "voice.wav")
                    sf.write(path, clip, sample_rate, subtype="FLOAT")
                    return (await pipe.call("encode", audio=path, max_seconds=VOICE_PROMPT_MAX_S))["codes"]

            self.dub = DubRuntime(dub_model(dub), synthesize, encode_voice, self.asr, self.aligner,
                                  WorkerSettings.from_env())
        return {}

    def edit_runtime(self):
        if self.edit is None:
            raise RuntimeError("speech editing is not initialized")
        return self.edit

    def dub_runtime(self):
        if self.dub is None:
            raise RuntimeError("dubbing is not initialized")
        return self.dub

    def close(self) -> None:
        for worker in (self.dub, self.asr, self.aligner):
            if worker is not None:
                worker.close()

    # ---- requests ---------------------------------------------------------------------------------------------

    async def run(self, request: dict):
        from zonos3.server import schemas as S

        op = request.get("op")
        if op == "init":
            return await self.init(request.get("edit"), request.get("dub"))
        if op == "edit_transcribe":
            return await self.edit_runtime().transcribe(validate(S.EditTranscribeRequest, request.get("request")))
        if op == "edit_align":
            return await self.edit_runtime().align(validate(S.EditAlignRequest, request.get("request")))
        if op == "edit":
            return await self.edit_runtime().edit(validate(S.EditRequest, request.get("request")))
        if op == "dub_capabilities":
            return await asyncio.to_thread(self.dub_runtime().capabilities)
        if op == "dub_ref":
            return await asyncio.to_thread(self.dub_runtime().add_ref, request["path"])
        if op == "dub_lips":
            rt = self.dub_runtime()
            rt.refs.get(request["ref_id"])
            return await asyncio.to_thread(rt.prepare_lips, request["ref_id"], request["video"])
        if op == "dub_transcribe":
            body = validate(S.DubTranscribeRequest, {"ref_id": request.get("ref_id"),
                                                     "ref_lang": request.get("ref_lang")})
            return await self.dub_runtime().transcribe(body.ref_id, body.ref_lang)
        if op == "dub_analyze":
            return await self.dub_runtime().analyze(validate(S.DubAnalyzeRequest, request.get("request")).model_dump())
        if op == "dub_generate":
            body = validate(S.DubGenerateRequest, request.get("request"))
            return await self.dub_runtime().generate(body.model_dump(), request.get("session"))
        if op == "dub_regenerate":
            body = validate(S.DubRegenerateRequest, request.get("request"))
            return await self.dub_runtime().regenerate(body.model_dump(), request.get("session"))
        if op == "dub_mux":
            await asyncio.to_thread(self.dub_runtime().mux, request["dub_id"], request["video"], request["out"])
            return {}
        raise ValueError(f"unknown op {op!r}")

    async def respond(self, request: dict) -> None:
        """Run one request and send its answer (errors as ``zonos3.server.edit.call`` maps them)."""
        from fastapi import HTTPException

        from zonos3.server.edit import call

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

    threading.Thread(target=read, name="edit-dub-stdin", daemon=True).start()
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
