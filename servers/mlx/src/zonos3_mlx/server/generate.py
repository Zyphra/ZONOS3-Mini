"""Speech generation over HTTP: /v2/generate and the OpenAI-compatible /v1/audio/speech.

``JobRun`` drives one scheduler job: its sink (called on the worker thread) feeds the rows to a per-request
``StreamDecoder`` (decoding there, where every MLX call runs) and posts the audio chunks to the event loop. Gain and
fade-out follow the sgl FrontendManager (``stream_tts_audio`` holds back the fade's tail until the end of stream).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import wave

import numpy as np
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from zonos3.reference.synth import Sampling, fade_out
from zonos3.server.schemas import GenerateRequest, SpeechRequest

from zonos3_mlx.engine import Plan, plan_request
from zonos3_mlx.scheduler import Job
from zonos3_mlx.server import speakers
from zonos3_mlx.server.state import ServerState
from zonos3_mlx.vocoder import StreamDecoder

logger = logging.getLogger("zonos3_mlx.server")
SAMPLE_RATE = 44100

# The speaker input fields a SpeechRequest `voice` cannot be combined with.
SPEAKER_INPUT_FIELDS = (
    "speaker_audio_base64", "speaker_audio_name", "speaker_reference_codes", "speaker_embedding_base64",
    "speaker_embedding_name", "speaker_embedding_id", "speaker_blend_embedding_id_a", "speaker_blend_embedding_id_b",
    "speaker_blend_t",
)


class JobRun:
    """One scheduler job and its audio. ``mode``: "whole" / "prefix" decode, or None for the rows only (edit fills,
    which stop on ``stop_id``: the span-end token)."""

    def __init__(self, state: ServerState, plan: Plan, mode: str | None, sampling: Sampling | None = None,
                 stop_id: int | None = None):
        self.state, self.plan = state, plan
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue = asyncio.Queue()
        c, dec = state.config, state.engine.decoder
        self.decoder = None
        if mode is not None:
            chunk, hold, context = state.options.prefix
            self.decoder = StreamDecoder(
                dec.decode, n_codebooks=c.n_codebooks, audio_pad_id=c.audio_pad_id, hop=dec.hop,
                prefill_rows=None if plan.prefill_rows is None else plan.prefill_rows.numpy(),
                context_codes=None if plan.context_codes is None else plan.context_codes.numpy(), mode=mode,
                chunk_frames=chunk, hold_back_frames=hold, context_frames=context)
        self.rows: list = []
        self.eos_frame: int | None = None
        self.record: dict | None = None
        self.failed = False
        self.job = Job(plan, sampling or Sampling.from_request(plan.request), self._sink,
                       c.eoa_id if stop_id is None else int(stop_id))

    def start(self) -> "JobRun":
        self.state.scheduler.enqueue(self.job)  # ValueError: CFG on --batch 1
        return self

    def abort(self) -> None:
        self.state.scheduler.abort(self.job)

    def _post(self, event: tuple) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    def _sink(self, event: tuple) -> None:
        """Worker thread: decode and forward. A decode failure aborts the job and is reported once."""
        if self.failed:
            return
        try:
            kind = event[0]
            if kind == "row":
                row, eos = event[1], event[2]
                self.rows.append(row)
                if eos is not None:
                    self.eos_frame = int(eos)
                if self.decoder is not None:
                    self._post(("audio", self.decoder.push(row, eos)))
            elif kind == "done":
                self.record = event[1]
                if self.decoder is not None:
                    self._post(("audio", self.decoder.finish()))
                self._post(("done", None))
            else:
                self.failed = True
                self._post(event)
        except Exception as exc:  # noqa: BLE001 - delivered to the request
            logger.exception("decode failed")
            self.failed = True
            self.job.aborted = True
            self._post(("error", str(exc)))

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


def pcm16_wav(audio: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """float32 PCM bytes as a 16-bit mono WAV file, samples clipped to [-1, 1]."""
    samples = np.clip(np.frombuffer(audio, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes((samples * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def finish_pcm(audio: bytes, gain: float, fade_out_ms: float) -> np.ndarray:
    pcm = np.frombuffer(audio, dtype=np.float32) * np.float32(gain)
    return fade_out(pcm.astype(np.float32), float(fade_out_ms))


def stream_decode_mode(state: ServerState, req: GenerateRequest) -> str:
    """GenerateRequest.stream_decode: the server default (prefix unless ZONOS3_STREAM_DECODE_DEFAULT=whole) for a
    streaming response; non-streaming responses are always the whole decode."""
    if not req.stream:
        return "whole"
    return req.stream_decode or state.options.stream_decode_default


async def plan_generate(state: ServerState, req: GenerateRequest, session_id: str | None, **plan_kwargs) -> Plan:
    """The scheduler plan of a GenerateRequest (HTTP 400 for a refused request, 500 for an encoder failure)."""
    try:
        speaker = await speakers.resolve_speaker(state, session_id, req)
        return await asyncio.to_thread(plan_request, state.engine, req.model_dump(), speaker, **plan_kwargs)
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
    if req.stream and req.format == "wav":
        raise HTTPException(status_code=400, detail="format=wav needs stream=false")
    plan = await plan_generate(state, req, session_id)
    run = await run_plan(state, plan, stream_decode_mode(state, req))
    headers = {"X-Audio-Sample-Rate": str(SAMPLE_RATE), "X-Audio-Channels": "1",
               "X-Audio-Format": "pcm16" if req.format == "wav" else "float32", "X-Seed": str(plan.seed)}
    if req.control_spans:
        headers["X-Control-Span-Follow"] = "reading" if plan.follow_tokens is not None else "rate"
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
        return Response(content=pcm16_wav(pcm.tobytes()), media_type="audio/wav", headers=headers)
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
    async def generate(req: GenerateRequest, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        return await generate_response(state, req, _session(x_tts_session_id))

    @r.post("/v1/audio/speech")
    async def speech(req: SpeechRequest, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        """OpenAI-compatible: the /v2/generate request of ``input`` (pcm streams, wav is buffered), with ``voice``
        (a speaker id of GET /v2/speakers) as its speaker_embedding_id."""
        fields = req.model_dump(exclude={"model", "input", "voice", "response_format"})
        session_id = _session(x_tts_session_id)
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
        request = GenerateRequest(text=req.input, stream=req.response_format == "pcm", format=req.response_format,
                                  **fields)
        return await generate_response(state, request, session_id)

    return r


def _session(header: str | None) -> str | None:
    return (header.strip() or None) if header is not None else None


def default_stream_decode() -> str:
    wanted = os.environ.get("ZONOS3_STREAM_DECODE_DEFAULT", "prefix").strip().lower()
    return "whole" if wanted == "whole" else "prefix"
