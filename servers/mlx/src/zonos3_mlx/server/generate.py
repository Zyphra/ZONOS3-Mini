"""Speech generation over HTTP: /v2/generate and the OpenAI-compatible /v1/audio/speech.

``JobRun`` drives one scheduler job: its sink (called on the worker thread) hands the rows to a per-request
``StreamDecoder`` on the scheduler's codec thread (so the backbone keeps stepping while a chunk decodes; one FIFO keeps
the chunks in order) and posts the audio chunks to the event loop. Gain and fade-out follow the sgl FrontendManager
(``stream_tts_audio`` holds back the fade's tail until the end of stream).
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from zonos3.generation import SAMPLE_RATE
from zonos3.reference.synth import Sampling, fade_out
from zonos3.server.schemas import SPEAKER_INPUT_FIELDS, GenerateRequest, SpeechRequest
from zonos3.server.speech import check_format, resolve_stream_decode, speech_headers
from zonos3.server.voices import normalize_session_id
from zonos3.tts.audio import pcm16_wav_bytes
from zonos3.tts.plan import parse_request, request_head
from zonos3.tts.stream import StreamDecoder

from zonos3_mlx.engine import Plan, plan_request
from zonos3_mlx.scheduler import Job
from zonos3_mlx.server import speakers
from zonos3_mlx.server.state import ServerState

logger = logging.getLogger("zonos3_mlx.server")
# Progressive (prefix re-decode) streaming: StreamDecoder re-decodes the growing prefix with every zac decoder.
PREFIX_STREAM = True


class JobRun:
    """One scheduler job and its audio. ``mode``: "whole" / "prefix" decode, or None for the rows only (edit fills,
    which stop on ``stop_id``: the span-end token)."""

    def __init__(
        self,
        state: ServerState,
        plan: Plan,
        mode: str | None,
        sampling: Sampling | None = None,
        stop_id: int | None = None,
    ):
        self.state, self.plan = state, plan
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue = asyncio.Queue()
        c, dec = state.config, state.engine.decoder
        self.decoder = None
        if mode is not None:
            self.decoder = StreamDecoder(
                dec.decode,
                n_codebooks=c.n_codebooks,
                audio_pad_id=c.audio_pad_id,
                hop=dec.hop,
                prefill_rows=plan.prefill_rows,
                context_codes=plan.context_codes,
                mode=mode,
                settings=state.options.prefix,
            )
        self.rows: list = []
        self.eos_frame: int | None = None
        self.record: dict | None = None
        self.failed = False
        self.job = Job(
            plan,
            sampling or Sampling.from_request(plan.request),
            self._sink,
            c.eoa_id if stop_id is None else int(stop_id),
        )

    def start(self) -> "JobRun":
        self.state.scheduler.enqueue(self.job)  # ValueError: CFG on --batch 1
        return self

    def abort(self) -> None:
        self.state.scheduler.abort(self.job)

    def _post(self, event: tuple) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    def _sink(self, event: tuple) -> None:
        """Worker thread: record the rows and queue their decode on the codec thread. A failure aborts the job and is
        reported once."""
        if self.failed:
            return
        kind = event[0]
        if kind == "row":
            row, eos = event[1], event[2]
            self.rows.append(row)
            if eos is not None:
                self.eos_frame = int(eos)
            if self.decoder is not None:
                self.state.scheduler.codec_submit(self._decode, self.decoder.push, row, eos)
        elif kind == "done":
            self.record = event[1]
            if self.decoder is None:
                self._post(("done", None))
            else:
                self.state.scheduler.codec_submit(self._decode, self.decoder.finish, last=True)
        else:
            self.failed = True
            self._post(event)

    def _decode(self, fn, *args, last: bool = False) -> None:
        """Codec thread: one StreamDecoder call, its audio posted (then the end of stream after ``finish``); skipped
        once the job failed or was aborted."""
        if self.failed or self.job.aborted:
            return
        try:
            self._post(("audio", fn(*args).tobytes()))
        except Exception as exc:  # noqa: BLE001 - delivered to the request
            logger.exception("decode failed")
            self.failed = True
            self.job.aborted = True
            self._post(("error", str(exc)))
            return
        if last:
            self._post(("done", None))

    async def chunks(self):
        """float32 PCM chunks (before gain) until the job ends; RuntimeError on a generation failure."""
        while True:
            kind, payload = await self.queue.get()
            if kind == "audio":
                if payload:
                    yield payload
            elif kind == "done":
                return
            else:
                raise RuntimeError(payload)

    async def wait(self) -> bytes:
        """The whole audio (before gain) once the job ends."""
        return b"".join([chunk async for chunk in self.chunks()])


def finish_pcm(audio: bytes, gain: float, fade_out_ms: float) -> np.ndarray:
    pcm = np.frombuffer(audio, dtype=np.float32) * np.float32(gain)
    return fade_out(pcm.astype(np.float32), float(fade_out_ms))


async def plan_generate(state: ServerState, req: GenerateRequest, session_id: str | None, **plan_kwargs) -> Plan:
    """The scheduler plan of a GenerateRequest (HTTP 400 for a refused request, 500 for an encoder failure). The
    request head (tag CFG, speaker mode, condition rows) is checked before the speaker inputs are resolved, in the sgl
    server's order."""
    try:
        request = parse_request(req.model_dump())
        speaker = await speakers.resolve_speaker(state, session_id, req, request_head(state.engine, request).mode)
        return await asyncio.to_thread(plan_request, state.engine, request, speaker, **plan_kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def run_plan(state: ServerState, plan: Plan, mode: str = "whole") -> JobRun:
    try:
        return JobRun(state, plan, mode).start()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def generate_response(state: ServerState, req: GenerateRequest, session_id: str | None) -> Response:
    """Run a GenerateRequest: streamed or buffered float32 PCM, or a buffered 16-bit WAV."""
    try:
        check_format(bool(req.stream), req.format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    plan = await plan_generate(state, req, session_id)
    try:
        mode = resolve_stream_decode(req.stream_decode, streaming=bool(req.stream), prefix_supported=PREFIX_STREAM)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run = await run_plan(state, plan, mode or "whole")
    headers = speech_headers(
        fmt=req.format,
        seed=plan.seed,
        control_spans=bool(req.control_spans),
        follows_reading=plan.follow_tokens is not None,
    )
    gain = state.engine.output_gain(req.headroom_db)
    if req.stream:
        return StreamingResponse(_stream(run, gain, float(req.fade_out_ms)), media_type="audio/pcm", headers=headers)
    try:
        audio = await run.wait()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        run.abort()  # a no-op once finished; frees the slot of a cancelled request
    pcm = finish_pcm(audio, gain, float(req.fade_out_ms))
    if req.format == "wav":
        return Response(content=pcm16_wav_bytes(pcm, SAMPLE_RATE), media_type="audio/wav", headers=headers)
    return Response(content=pcm.tobytes(), media_type="audio/pcm", headers=headers)


async def _stream(run: JobRun, gain: float, fade_out_ms: float):
    """Gained chunks as they come; the last ``fade_out_ms`` is held back and faded at the end of stream."""
    fade_bytes = int(SAMPLE_RATE * max(0.0, fade_out_ms) / 1000.0) * 4
    held = b""
    try:
        async for chunk in run.chunks():
            data = (np.frombuffer(chunk, dtype=np.float32) * np.float32(gain)).astype(np.float32).tobytes()
            if fade_bytes > 0:
                held += data
                if len(held) > fade_bytes:
                    yield held[:-fade_bytes]
                    held = held[-fade_bytes:]
            else:
                yield data
        if held:
            yield fade_out(np.frombuffer(held, dtype=np.float32), fade_out_ms).astype(np.float32).tobytes()
    except RuntimeError:
        logger.exception("streamed generation failed")  # the status line is sent: the stream just ends
    finally:
        run.abort()


def router(state: ServerState) -> APIRouter:
    r = APIRouter()

    @r.post("/v2/generate")
    async def generate(
        req: GenerateRequest, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")
    ):
        return await generate_response(state, req, normalize_session_id(x_tts_session_id))

    @r.post("/v1/audio/speech")
    async def speech(req: SpeechRequest, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        """OpenAI-compatible: the /v2/generate request of ``input`` (pcm streams, wav is buffered), with ``voice``
        (a speaker id of GET /v2/speakers) as its speaker_embedding_id."""
        fields = req.model_dump(exclude={"model", "input", "voice", "response_format"})
        session_id = normalize_session_id(x_tts_session_id)
        if req.voice is not None:
            if any(fields[name] is not None for name in SPEAKER_INPUT_FIELDS):
                raise HTTPException(status_code=400, detail="voice cannot be combined with speaker_* fields")
            known = speakers.get_default_voice(state, req.voice) is not None
            if not known and session_id is not None:
                async with state.sessions_lock:
                    known = req.voice in state.sessions.get(session_id, {})
            if not known:
                raise HTTPException(status_code=400, detail=f"unknown voice '{req.voice}'")
            fields["speaker_embedding_id"] = req.voice
        request = GenerateRequest(
            text=req.input, stream=req.response_format == "pcm", format=req.response_format, **fields
        )
        return await generate_response(state, request, session_id)

    return r
