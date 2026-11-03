"""Voices: default voices on disk, session voices (POST /v2/speakers) and a request's speaker resolution.

A port of the sgl api_server speaker code (default voices, the session cache, embedding files, slerp blends, the
/v2/speakers routes) with the ggml server's companions: embeddings come from spk-encoder.gguf, reference codes from a
default voice's ``<stem>.codes.npy`` or from the MLX reference encoder (dac-encoder.gguf + ssl-teacher.gguf).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
from zonos3.generation import REFERENCE_AUDIO_MAX_SECONDS
from zonos3.reference.synth import decode_base64_blob, load_embedding_vector
from zonos3.server.schemas import SpeakerCreateRequest
from zonos3.tts.audio import _transcode_audio_bytes_to_wav
from zonos3.tts.plan import Speaker, reference_codes, truncate_reference
from zonos3.tts.steering import disabled_controls

from zonos3_mlx.audio_io import decode_pcm
from zonos3_mlx.server.state import ServerState

AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
EMBEDDING_EXTENSIONS = {".npy", ".npz"}
REFERENCE_CACHE = 64
NO_SPEAKER_ENCODER = ("This server has no speaker encoder (spk-encoder.gguf); send speaker_embedding_base64 or use "
                      "speaker_mode=reference.")
NO_REFERENCE_ENCODER = ("This server has no reference encoder (dac-encoder.gguf + ssl-teacher.gguf); send "
                        "speaker_reference_codes or use speaker_mode=embedding.")


@dataclass
class Voice:
    """A default or session voice: its source (audio bytes or an embedding), and what was computed from it."""

    speaker_id: str
    label: str
    source_type: Literal["audio", "embedding"]
    created_at: float
    is_default: bool
    original_name: str | None = None
    path: Path | None = None  # default voices
    audio_bytes: bytes | None = None  # the uploaded / file bytes of an audio voice
    preview_wav: bytes | None = None
    embedding: torch.Tensor | None = None
    codes_path: Path | None = None  # a default voice's shipped <stem>.codes.npy
    transcript: str | None = None


# ----------------------------------------------------------------------------------------------------- modes


def voice_modes(state: ServerState, voice: Voice) -> list[str]:
    """The modes a voice supports given the loaded companions (ggml speakers.cpp ``voice_modes``)."""
    out = []
    for mode in state.engine.speaker_modes:
        if voice.source_type == "embedding":
            if mode == "embedding":
                out.append(mode)
            continue
        if mode == "embedding" and (state.engine.speaker_encoder is not None or voice.embedding is not None):
            out.append(mode)
        if mode == "reference" and (state.engine.has_reference_encoder or voice.codes_path is not None):
            out.append(mode)
    return out


def serialize(state: ServerState, voice: Voice) -> dict:
    modes = voice_modes(state, voice)
    dim = state.config.speaker_embedding_dim
    return {
        "id": voice.speaker_id, "label": voice.label, "source_type": voice.source_type,
        "scope": "default" if voice.is_default else "session", "is_default": voice.is_default,
        "original_name": voice.original_name, "created_at": voice.created_at,
        "has_preview": voice.source_type == "audio" and (voice.is_default or bool(voice.audio_bytes)),
        "speaker_modes": modes, "embedding_dim": dim if dim > 0 else None, "transcript": voice.transcript,
        "disabled_controls": disabled_controls(state.engine.steer_bundle, voice.speaker_id),
    }


# ----------------------------------------------------------------------------------------------------- defaults


def _source_type(path: Path):
    if path.name.lower().endswith(".codes.npy"):  # an audio voice's reference codes
        return None
    suffix = path.suffix.lower()
    if suffix in EMBEDDING_EXTENSIONS:
        return "embedding"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    return None


def _transcript(path: Path) -> str | None:
    try:
        return path.with_suffix(".txt").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def list_default_voices(state: ServerState) -> list[Voice]:
    """Default voices under --tts-default-voices-dir (ids ``default_<sha1[:16]>`` of the relative POSIX path)."""
    root = state.options.default_voices_dir
    if root is None or not root.is_dir():
        return []
    root = root.resolve()
    with state.default_lock:
        previous, current = state.default_voices, {}
        for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
            kind = _source_type(path)
            if kind is None:
                continue
            try:
                mtime = path.resolve().stat().st_mtime
            except OSError:
                continue
            speaker_id = "default_" + hashlib.sha1(path.relative_to(root).as_posix().encode("utf-8")).hexdigest()[:16]
            voice = previous.get(speaker_id)
            if voice is None or voice.path != path.resolve() or voice.created_at != mtime or voice.source_type != kind:
                codes = path.with_name(path.stem + ".codes.npy") if kind == "audio" else None
                label = path.stem.replace("_", " ").replace("-", " ").strip() or path.name
                voice = Voice(speaker_id, label, kind, mtime, True, path.name, path.resolve(),
                              codes_path=codes if codes is not None and codes.is_file() else None)
            voice.transcript = _transcript(voice.path)
            current[speaker_id] = voice
        state.default_voices = current
        return sorted(current.values(), key=lambda v: v.label.lower())


def get_default_voice(state: ServerState, speaker_id: str) -> Voice | None:
    return next((v for v in list_default_voices(state) if v.speaker_id == speaker_id), None)


def _voice_bytes(voice: Voice) -> bytes:
    if voice.audio_bytes is None:
        try:
            voice.audio_bytes = voice.path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Failed to read default voice '{voice.label}'.") from exc
    return voice.audio_bytes


def preview_wav(voice: Voice) -> bytes:
    if voice.source_type != "audio":
        raise ValueError("Embedding voices do not include preview audio.")
    if voice.preview_wav is None:
        voice.preview_wav = _transcode_audio_bytes_to_wav(_voice_bytes(voice))
    return voice.preview_wav


# ----------------------------------------------------------------------------------------------------- embeddings


async def embed_audio(state: ServerState, audio_bytes: bytes) -> torch.Tensor:
    """The speaker embedding of an audio clip (spk-encoder.gguf on the worker thread)."""
    encoder = state.engine.speaker_encoder
    if encoder is None:
        raise ValueError(NO_SPEAKER_ENCODER)
    pcm = decode_pcm(audio_bytes)
    return torch.from_numpy(await state.run(encoder.embed, pcm))


async def voice_embedding(state: ServerState, voice: Voice) -> torch.Tensor:
    dim = state.config.speaker_embedding_dim
    if voice.embedding is None:
        if voice.source_type == "embedding":
            voice.embedding = load_embedding_vector(_voice_bytes(voice), expected_dim=dim, file_name=voice.original_name)
        else:
            voice.embedding = await embed_audio(state, _voice_bytes(voice))
    if int(voice.embedding.numel()) != dim:
        raise ValueError(f"Speaker '{voice.label}' has {int(voice.embedding.numel())} dimensions, the model expects {dim}.")
    return voice.embedding.detach().to(torch.float32).clone()


async def session_voice(state: ServerState, session_id: str | None, speaker_id: str) -> Voice:
    if session_id is None:
        raise ValueError("Cached speaker operations require the X-TTS-Session-ID header.")
    async with state.sessions_lock:
        voice = state.sessions.get(session_id, {}).get(speaker_id)
    if voice is None:
        raise ValueError(f"Unknown cached speaker id '{speaker_id}'.")
    return voice


async def find_voice(state: ServerState, session_id: str | None, speaker_id: str) -> Voice:
    voice = get_default_voice(state, speaker_id)
    return voice if voice is not None else await session_voice(state, session_id, speaker_id)


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    v0, v1 = v0.reshape(-1).float(), v1.reshape(-1).float()
    if v0.shape != v1.shape:
        raise ValueError(f"Cannot blend speaker embeddings with different shapes {tuple(v0.shape)} and {tuple(v1.shape)}.")
    t = float(t)
    if not 0.0 <= t <= 1.0:
        raise ValueError("speaker_blend_t must be between 0.0 and 1.0.")
    a, b = v0 / (torch.linalg.vector_norm(v0) + 1e-8), v1 / (torch.linalg.vector_norm(v1) + 1e-8)
    omega = torch.acos(torch.clamp(torch.dot(a, b), -1.0, 1.0))
    if float(omega) < 1e-6:
        return ((1.0 - t) * v0 + t * v1).contiguous()
    s = torch.sin(omega)
    return ((torch.sin((1.0 - t) * omega) / s) * v0 + (torch.sin(t * omega) / s) * v1).contiguous()


async def resolve_embedding(state: ServerState, session_id: str | None, req) -> torch.Tensor | None:
    """speaker_mode=embedding: upload, embedding file, one voice id or a blend of two (sgl _resolve_speaker_embedding)."""
    direct = sum(bool(v) for v in (req.speaker_audio_base64, req.speaker_embedding_base64))
    if direct > 1:
        raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")
    cached, blended = bool(req.speaker_embedding_id), bool(req.speaker_blend_embedding_id_a or req.speaker_blend_embedding_id_b)
    selected = sum((direct == 1, cached, blended))
    if selected == 0:
        return None
    if selected > 1:
        raise ValueError("Provide speaker conditioning via upload, one cached speaker id, or two cached ids for blending.")
    if blended:
        if not req.speaker_blend_embedding_id_a or not req.speaker_blend_embedding_id_b:
            raise ValueError("Provide both cached speaker ids to blend between two speakers.")
        a = await voice_embedding(state, await find_voice(state, session_id, req.speaker_blend_embedding_id_a))
        b = await voice_embedding(state, await find_voice(state, session_id, req.speaker_blend_embedding_id_b))
        return slerp(a, b, 0.5 if req.speaker_blend_t is None else req.speaker_blend_t)
    if cached:
        voice = await find_voice(state, session_id, req.speaker_embedding_id)
        if voice.source_type == "audio" and voice.embedding is None and state.engine.speaker_encoder is None:
            raise ValueError(NO_SPEAKER_ENCODER)
        return await voice_embedding(state, voice)
    dim = state.config.speaker_embedding_dim
    if req.speaker_embedding_base64:
        return load_embedding_vector(decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64"),
                                     expected_dim=dim, file_name=req.speaker_embedding_name)
    return await embed_audio(state, decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64"))


# ----------------------------------------------------------------------------------------------------- reference


async def encode_reference(state: ServerState, audio_bytes: bytes,
                           max_seconds: float = REFERENCE_AUDIO_MAX_SECONDS) -> torch.Tensor:
    """Reference prefix codes of a clip (the MLX reference encoder), cached per sha1 of the bytes (LRU of 64)."""
    if not state.engine.has_reference_encoder:
        raise ValueError(NO_REFERENCE_ENCODER)
    key = (hashlib.sha1(audio_bytes).hexdigest(), float(max_seconds))
    cached = state.reference_codes.get(key)
    if cached is not None:
        state.reference_codes.move_to_end(key)
        return cached
    pcm = decode_pcm(audio_bytes, max_seconds=max_seconds)

    def encode():
        return state.engine.reference_encoder().encode(pcm, max_seconds=max_seconds)

    codes = torch.from_numpy(np.asarray(await state.run(encode), dtype=np.int32))
    state.reference_codes[key] = codes
    while len(state.reference_codes) > REFERENCE_CACHE:
        state.reference_codes.popitem(last=False)
    return codes


async def resolve_reference(state: ServerState, session_id: str | None, req) -> torch.Tensor | None:
    """speaker_mode=reference: inline codes, a default voice's codes or clip, a session clip, or an upload
    (sgl _resolve_speaker_reference_audio + the tokenizer's reference encode)."""
    if any((req.speaker_embedding_base64, req.speaker_blend_embedding_id_a, req.speaker_blend_embedding_id_b)):
        raise ValueError("This checkpoint conditions on raw reference audio and does not accept speaker embedding "
                         "files or embedding blends.")
    codes = reference_codes(state.engine, req.speaker_reference_codes)
    if codes is not None:
        if req.speaker_audio_base64 or req.speaker_embedding_id:
            raise ValueError("Provide speaker_reference_codes or reference audio, not both.")
        return codes
    if req.speaker_embedding_id:
        if req.speaker_audio_base64:
            raise ValueError("Provide either a reference audio file or a cached speaker id, not both.")
        voice = get_default_voice(state, req.speaker_embedding_id)
        if voice is not None:
            if voice.source_type != "audio":
                raise ValueError(f"Default voice '{voice.label}' is an embedding file; speaker_mode=reference needs "
                                 "reference audio.")
            if voice.codes_path is not None:
                return truncate_reference(torch.from_numpy(np.load(voice.codes_path).astype(np.int32)),
                                          state.engine.reference_max_frames)
            return await encode_reference(state, _voice_bytes(voice))
        voice = await session_voice(state, session_id, req.speaker_embedding_id)
        if not voice.audio_bytes:
            raise ValueError(f"Cached speaker '{voice.label}' has no reference audio.")
        return await encode_reference(state, voice.audio_bytes)
    if req.speaker_audio_base64 is None:
        return None
    return await encode_reference(state, decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64"))


async def resolve_speaker(state: ServerState, session_id: str | None, req, mode: str | None) -> Speaker:
    """The request's speaker inputs for its speaker mode (``zonos3.tts.plan.request_head``)."""
    if mode == "reference":
        return Speaker(mode, reference_codes=await resolve_reference(state, session_id, req))
    return Speaker(mode, embedding=await resolve_embedding(state, session_id, req))


# ----------------------------------------------------------------------------------------------------- routes


def router(state: ServerState) -> APIRouter:
    r = APIRouter()

    def session_of(header: str | None) -> str | None:
        return header.strip() or None if header is not None else None

    @r.post("/v2/speakers")
    async def create_speaker(req: SpeakerCreateRequest, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        try:
            session_id = session_of(x_tts_session_id)
            if session_id is None:
                raise ValueError("Cached speaker operations require the X-TTS-Session-ID header.")
            if req.speaker_audio_base64 and req.speaker_embedding_base64:
                raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")
            if req.speaker_embedding_base64:
                name = req.speaker_embedding_name
                embedding = load_embedding_vector(
                    decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64"),
                    expected_dim=state.config.speaker_embedding_dim, file_name=name)
                voice = Voice(f"spk_{uuid.uuid4().hex[:12]}", _label(req.label, name, "Speaker embedding"), "embedding",
                              time.time(), False, name, embedding=embedding)
            elif req.speaker_audio_base64:
                if state.engine.speaker_encoder is None and not state.engine.has_reference_encoder:
                    raise ValueError("This server has no speaker encoder or reference encoder for uploaded audio.")
                name = req.speaker_audio_name
                raw = decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64")
                voice = Voice(f"spk_{uuid.uuid4().hex[:12]}", _label(req.label, name, "Speaker audio"), "audio",
                              time.time(), False, name, audio_bytes=raw, preview_wav=_transcode_audio_bytes_to_wav(raw))
                if state.engine.speaker_encoder is not None:
                    voice.embedding = await embed_audio(state, raw)
            else:
                raise ValueError("Provide an audio file or saved embedding file to cache.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        async with state.sessions_lock:
            state.sessions.setdefault(session_id, {})[voice.speaker_id] = voice
        return serialize(state, voice)

    @r.get("/v2/speakers")
    async def list_speakers(x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        voices = list_default_voices(state)
        session_id = session_of(x_tts_session_id)
        if session_id is not None:
            async with state.sessions_lock:
                voices += sorted(state.sessions.get(session_id, {}).values(), key=lambda v: v.created_at, reverse=True)
        return {"speakers": [serialize(state, v) for v in voices]}

    @r.get("/v2/speakers/{speaker_id}/preview")
    async def preview(speaker_id: str, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        try:
            voice = await find_voice(state, session_of(x_tts_session_id), speaker_id)
            if voice.source_type != "audio":
                raise HTTPException(status_code=404, detail="This voice has no preview audio.")
            return Response(content=preview_wav(voice), media_type="audio/wav")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @r.get("/v2/speakers/{speaker_id}/embedding")
    async def embedding(speaker_id: str, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        try:
            voice = await find_voice(state, session_of(x_tts_session_id), speaker_id)
            if voice.source_type == "audio" and voice.embedding is None and state.engine.speaker_encoder is None:
                raise HTTPException(status_code=404, detail=f"No speaker embedding for {speaker_id!r}.")
            vec = await voice_embedding(state, voice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"id": speaker_id, "embedding": [round(float(x), 6) for x in vec.reshape(-1)]}

    return r


def _label(explicit: str | None, name: str | None, fallback: str) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if name and Path(name).stem.strip():
        return Path(name).stem.strip()
    return fallback
