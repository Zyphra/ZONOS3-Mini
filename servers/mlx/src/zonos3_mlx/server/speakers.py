"""Voices: default voices on disk, session voices (POST /v2/speakers) and a request's speaker resolution.

The sgl server's speaker rules (zonos3.server.voices: the default-voice scan, session ids, the /v2/speakers documents;
embedding files, slerp blends) with the ggml server's companions: embeddings come from a default voice's
``<stem>.embedding.npy`` or spk-encoder.gguf, reference codes from a default voice's ``<stem>.codes.npy`` or from the
MLX reference encoder (zac-encoder.gguf + ssl-teacher.gguf).
"""

from __future__ import annotations

import hashlib
import logging
import time

import numpy as np
import torch
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
from zonos3.generation import REFERENCE_AUDIO_MAX_SECONDS
from zonos3.reference.synth import decode_base64_blob, load_embedding_vector, slerp
from zonos3.server.schemas import SpeakerCreateRequest
from zonos3.server.voices import (
    DefaultVoice,
    SessionVoice,
    new_speaker_id,
    normalize_session_id,
    read_voice_file,
    require_session_id,
    scan_default_voices,
    serialize_default_voice,
    serialize_session_voice,
    speaker_label,
    voice_preview,
    voice_sidecar,
    voice_speaker_modes,
    warm_voice_embeddings,
)
from zonos3.tts.audio import _transcode_audio_bytes_to_wav
from zonos3.tts.plan import Speaker, reference_codes, truncate_reference
from zonos3.tts.steering import disabled_controls

from zonos3_mlx.audio_io import decode_pcm
from zonos3_mlx.server.state import ServerState

REFERENCE_CACHE = 64
NO_SPEAKER_ENCODER = (
    "This server has no speaker encoder (spk-encoder.gguf); send speaker_embedding_base64 or use "
    "speaker_mode=reference."
)
NO_REFERENCE_ENCODER = (
    "This server has no reference encoder (zac-encoder.gguf + ssl-teacher.gguf); send "
    "speaker_reference_codes or use speaker_mode=embedding."
)

logger = logging.getLogger("zonos3_mlx.server")

Voice = DefaultVoice | SessionVoice


# ----------------------------------------------------------------------------------------------------- modes


def voice_modes(state: ServerState, voice: Voice) -> list[str]:
    """The modes a voice supports given the loaded companions (ggml speakers.cpp ``voice_modes``): an audio voice's
    embedding needs the speaker encoder or a shipped ``.embedding.npy``, its reference prefix the reference encoder
    or shipped ``.codes.npy``."""
    engine = state.engine
    has_embedding = has_reference = True
    if voice.source_type == "audio":
        default = isinstance(voice, DefaultVoice)
        has_embedding = (
            engine.speaker_encoder is not None
            or voice.embedding is not None
            or (default and voice_sidecar(voice.path, ".embedding.npy") is not None)
        )
        has_reference = engine.has_reference_encoder or (
            default and voice_sidecar(voice.path, ".codes.npy") is not None
        )
    modes = voice_speaker_modes(engine.speaker_modes, voice.source_type, has_embedding)
    return [m for m in modes if m != "reference" or has_reference]


def serialize(state: ServerState, voice: Voice) -> dict:
    modes, dim = voice_modes(state, voice), state.config.speaker_embedding_dim
    disabled = disabled_controls(state.engine.steer_bundle, voice.speaker_id)
    if isinstance(voice, DefaultVoice):
        return serialize_default_voice(voice, modes=modes, embedding_dim=dim, disabled=disabled)
    return serialize_session_voice(
        speaker_id=voice.speaker_id,
        label=voice.label,
        source_type=voice.source_type,
        original_name=voice.original_name,
        created_at=voice.created_at,
        has_preview=bool(voice.audio_bytes),
        embedding_dim=dim,
        modes=modes,
        disabled=disabled,
    )


# ----------------------------------------------------------------------------------------------------- defaults


def list_default_voices(state: ServerState) -> list[DefaultVoice]:
    """Default voices under --tts-default-voices-dir (zonos3.server.voices.scan_default_voices)."""
    root = state.options.default_voices_dir
    if root is None or not root.is_dir():
        return []
    with state.default_lock:
        state.default_voices = scan_default_voices(root, state.default_voices)
        return sorted(state.default_voices.values(), key=lambda v: v.label.lower())


def get_default_voice(state: ServerState, speaker_id: str) -> DefaultVoice | None:
    return next((v for v in list_default_voices(state) if v.speaker_id == speaker_id), None)


def voice_audio(voice: Voice) -> bytes:
    """An audio voice's clip as uploaded / on disk."""
    return read_voice_file(voice) if isinstance(voice, DefaultVoice) else voice.audio_bytes


async def warm_default_voices(state: ServerState) -> None:
    """Every default voice's speaker embedding, once (the sgl server's warm-up with a steering bundle: the studio's Who
    map places them all)."""
    if "embedding" in state.engine.speaker_modes:
        await warm_voice_embeddings(list_default_voices(state), lambda voice: voice_embedding(state, voice), logger)


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
            voice.embedding = load_embedding_vector(
                read_voice_file(voice), expected_dim=dim, file_name=voice.original_name
            )
        else:
            shipped = voice_sidecar(voice.path, ".embedding.npy") if isinstance(voice, DefaultVoice) else None
            if shipped is not None:
                try:
                    voice.embedding = load_embedding_vector(
                        shipped.read_bytes(), expected_dim=dim, file_name=shipped.name
                    )
                except (OSError, ValueError) as exc:
                    logger.warning("cannot load shipped embedding %s (%s); embedding %s", shipped, exc, voice.path)
            if voice.embedding is None:
                voice.embedding = await embed_audio(state, voice_audio(voice))
    if int(voice.embedding.numel()) != dim:
        raise ValueError(
            f"Speaker '{voice.label}' has {int(voice.embedding.numel())} dimensions, the model expects {dim}."
        )
    return voice.embedding.detach().to(torch.float32).clone()


async def session_voice(state: ServerState, session_id: str | None, speaker_id: str) -> SessionVoice:
    session_id = require_session_id(session_id)
    async with state.sessions_lock:
        voice = state.sessions.get(session_id, {}).get(speaker_id)
    if voice is None:
        raise ValueError(f"Unknown cached speaker id '{speaker_id}'.")
    return voice


async def find_voice(state: ServerState, session_id: str | None, speaker_id: str) -> Voice:
    voice = get_default_voice(state, speaker_id)
    return voice if voice is not None else await session_voice(state, session_id, speaker_id)


def needs_encoder(state: ServerState, voice: Voice) -> bool:
    """An audio voice whose embedding only the (missing) speaker encoder could compute."""
    return (
        voice.source_type == "audio"
        and voice.embedding is None
        and state.engine.speaker_encoder is None
        and not (isinstance(voice, DefaultVoice) and voice_sidecar(voice.path, ".embedding.npy") is not None)
    )


async def resolve_embedding(state: ServerState, session_id: str | None, req) -> torch.Tensor | None:
    """speaker_mode=embedding: upload, embedding file, one voice id or a blend of two (sgl _resolve_speaker_embedding)."""
    direct = sum(bool(v) for v in (req.speaker_audio_base64, req.speaker_embedding_base64))
    if direct > 1:
        raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")
    cached, blended = (
        bool(req.speaker_embedding_id),
        bool(req.speaker_blend_embedding_id_a or req.speaker_blend_embedding_id_b),
    )
    selected = sum((direct == 1, cached, blended))
    if selected == 0:
        return None
    if selected > 1:
        raise ValueError(
            "Provide speaker conditioning via upload, one cached speaker id, or two cached ids for blending."
        )
    if blended:
        if not req.speaker_blend_embedding_id_a or not req.speaker_blend_embedding_id_b:
            raise ValueError("Provide both cached speaker ids to blend between two speakers.")
        a = await voice_embedding(state, await find_voice(state, session_id, req.speaker_blend_embedding_id_a))
        b = await voice_embedding(state, await find_voice(state, session_id, req.speaker_blend_embedding_id_b))
        return slerp(a, b, 0.5 if req.speaker_blend_t is None else req.speaker_blend_t)
    if cached:
        voice = await find_voice(state, session_id, req.speaker_embedding_id)
        if needs_encoder(state, voice):
            raise ValueError(NO_SPEAKER_ENCODER)
        return await voice_embedding(state, voice)
    dim = state.config.speaker_embedding_dim
    if req.speaker_embedding_base64:
        return load_embedding_vector(
            decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64"),
            expected_dim=dim,
            file_name=req.speaker_embedding_name,
        )
    return await embed_audio(state, decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64"))


# ----------------------------------------------------------------------------------------------------- reference


async def encode_reference(
    state: ServerState, audio_bytes: bytes, max_seconds: float = REFERENCE_AUDIO_MAX_SECONDS
) -> torch.Tensor:
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
    (sgl _resolve_speaker_reference_audio + the frontend's reference encode)."""
    if any((req.speaker_embedding_base64, req.speaker_blend_embedding_id_a, req.speaker_blend_embedding_id_b)):
        raise ValueError(
            "This checkpoint conditions on raw reference audio and does not accept speaker embedding "
            "files or embedding blends."
        )
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
                raise ValueError(
                    f"Default voice '{voice.label}' is an embedding file; speaker_mode=reference needs reference audio."
                )
            codes = voice_sidecar(voice.path, ".codes.npy")
            if codes is not None:
                return truncate_reference(
                    torch.from_numpy(np.load(codes).astype(np.int32)), state.engine.reference_max_frames
                )
            return await encode_reference(state, read_voice_file(voice))
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

    @r.post("/v2/speakers")
    async def create_speaker(
        req: SpeakerCreateRequest, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")
    ):
        try:
            session_id = require_session_id(x_tts_session_id)
            if req.speaker_audio_base64 and req.speaker_embedding_base64:
                raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")
            if req.speaker_embedding_base64:
                name = req.speaker_embedding_name
                embedding = load_embedding_vector(
                    decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64"),
                    expected_dim=state.config.speaker_embedding_dim,
                    file_name=name,
                )
                voice = SessionVoice(
                    new_speaker_id(),
                    speaker_label(req.label, name, fallback="Speaker embedding"),
                    "embedding",
                    time.time(),
                    name,
                    embedding=embedding,
                )
            elif req.speaker_audio_base64:
                if state.engine.speaker_encoder is None and not state.engine.has_reference_encoder:
                    raise ValueError("This server has no speaker encoder or reference encoder for uploaded audio.")
                name = req.speaker_audio_name
                raw = decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64")
                voice = SessionVoice(
                    new_speaker_id(),
                    speaker_label(req.label, name, fallback="Speaker audio"),
                    "audio",
                    time.time(),
                    name,
                    audio_bytes=raw,
                    preview_wav=_transcode_audio_bytes_to_wav(raw),
                )
                if state.engine.speaker_encoder is not None:  # computed at upload, as the sgl server does
                    voice.embedding = await embed_audio(state, raw)
            else:
                raise ValueError("Provide an audio file or saved embedding file to cache.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        async with state.sessions_lock:
            state.sessions.setdefault(session_id, {})[voice.speaker_id] = voice
        return serialize(state, voice)

    @r.get("/v2/speakers")
    async def list_speakers(x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        voices: list[Voice] = list(list_default_voices(state))
        session_id = normalize_session_id(x_tts_session_id)
        if session_id is not None:
            async with state.sessions_lock:
                voices += sorted(state.sessions.get(session_id, {}).values(), key=lambda v: v.created_at, reverse=True)
        return {"speakers": [serialize(state, v) for v in voices]}

    @r.get("/v2/speakers/{speaker_id}/preview")
    async def preview(speaker_id: str, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        try:
            voice = await find_voice(state, normalize_session_id(x_tts_session_id), speaker_id)
            if voice.source_type != "audio":
                raise HTTPException(status_code=404, detail="This voice has no preview audio.")
            wav = voice_preview(voice)[0] if isinstance(voice, DefaultVoice) else voice.preview_wav
            return Response(content=wav, media_type="audio/wav")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @r.get("/v2/speakers/{speaker_id}/embedding")
    async def embedding(speaker_id: str, x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID")):
        try:
            voice = await find_voice(state, normalize_session_id(x_tts_session_id), speaker_id)
            if needs_encoder(state, voice):
                raise HTTPException(status_code=404, detail=f"No speaker embedding for {speaker_id!r}.")
            vec = await voice_embedding(state, voice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"id": speaker_id, "embedding": [round(float(x), 6) for x in vec.reshape(-1)]}

    return r
