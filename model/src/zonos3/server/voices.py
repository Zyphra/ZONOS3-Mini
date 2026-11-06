"""Voices of the Python servers (sgl, mlx, the Space): the default voices on disk (--tts-default-voices-dir), session ids
(X-TTS-Session-ID) and the GET /v2/speakers serialization.

A default voice is an audio file or a ``.npy``/``.npz`` speaker embedding anywhere under the directory, its id
``default_<sha1[:16]>`` of its POSIX path relative to the directory; an audio voice's ``<stem>.txt`` is its transcript
and its ``<stem>.codes.npy`` / ``<stem>.embedding.npy`` sidecars (the GGUF servers' reference codes and speaker
embedding) are no voice of their own.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Literal

import torch

from zonos3.tts.audio import _transcode_audio_bytes_to_wav

AUDIO_EXTENSIONS = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"})
EMBEDDING_EXTENSIONS = frozenset({".npy", ".npz"})
SIDECARS = (".codes.npy", ".embedding.npy")

SourceType = Literal["audio", "embedding"]


@dataclass
class DefaultVoice:
    """A default voice and what was computed from it (its embedding, its transcoded WAV preview)."""

    speaker_id: str
    label: str
    source_type: SourceType
    path: Path
    mtime: float
    original_name: str
    embedding: torch.Tensor | None = None
    audio_bytes: bytes | None = None  # the preview: the file transcoded to WAV (voice_preview)
    audio_media_type: str | None = None
    transcript: str | None = None  # <voice>.txt next to the audio (studio edit examples)


@dataclass
class SessionVoice:
    """A session voice of the mlx server and the Space (POST /v2/speakers, per X-TTS-Session-ID): its upload and what
    was computed from it (the sgl server keeps its own CachedSpeakerReference)."""

    speaker_id: str
    label: str
    source_type: SourceType
    created_at: float
    original_name: str | None = None
    audio_bytes: bytes | None = None  # the uploaded clip
    preview_wav: bytes | None = None
    embedding: torch.Tensor | None = None


def voice_source_type(path: Path) -> SourceType | None:
    """``audio`` or ``embedding`` by extension; None for anything else and for an audio voice's sidecars."""
    name = path.name.lower()
    if name.endswith(SIDECARS):
        return None
    suffix = path.suffix.lower()
    if suffix in EMBEDDING_EXTENSIONS:
        return "embedding"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    return None


def voice_sidecar(path: Path, suffix: str) -> Path | None:
    """An audio voice's ``<stem><suffix>`` sidecar (one of SIDECARS), when present."""
    sidecar = path.with_name(path.stem + suffix)
    return sidecar if sidecar.is_file() else None


def default_voice_id(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return "default_" + hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]


def default_voice_label(path: Path) -> str:
    label = path.stem.replace("_", " ").replace("-", " ").strip()
    return label or path.name


def read_voice_transcript(path: Path) -> str | None:
    """Text of the ``<voice>.txt`` sidecar next to a default voice, when present."""
    try:
        return path.with_suffix(".txt").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def scan_default_voices(root: Path, previous: dict[str, DefaultVoice]) -> dict[str, DefaultVoice]:
    """The default voices under ``root`` by id; an entry of ``previous`` whose file is unchanged (path, mtime, kind) is
    kept with what was computed from it. Transcripts are read again."""
    root = root.resolve()
    current: dict[str, DefaultVoice] = {}
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        source_type = voice_source_type(path)
        if source_type is None:
            continue
        resolved = path.resolve()
        try:
            stat = resolved.stat()
        except OSError:
            continue
        speaker_id = default_voice_id(root, path)
        existing = previous.get(speaker_id)
        if (
            existing is not None
            and existing.path == resolved
            and existing.mtime == stat.st_mtime
            and existing.source_type == source_type
        ):
            existing.transcript = read_voice_transcript(existing.path)
            current[speaker_id] = existing
            continue
        current[speaker_id] = DefaultVoice(
            speaker_id=speaker_id,
            label=default_voice_label(resolved),
            source_type=source_type,
            path=resolved,
            mtime=stat.st_mtime,
            original_name=resolved.name,
            transcript=read_voice_transcript(resolved),
        )
    return current


def read_voice_file(voice: DefaultVoice) -> bytes:
    try:
        return voice.path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Failed to read default voice '{voice.label}'.") from exc


def voice_preview(voice: DefaultVoice) -> tuple[bytes, str]:
    """An audio voice's file transcoded to WAV (cached on the voice) and its media type."""
    if voice.source_type != "audio":
        raise ValueError("Default embedding speakers do not include preview audio.")
    if voice.audio_bytes is None:
        voice.audio_bytes = _transcode_audio_bytes_to_wav(read_voice_file(voice))
        voice.audio_media_type = "audio/wav"
    return voice.audio_bytes, voice.audio_media_type or "audio/wav"


async def warm_voice_embeddings(
    voices: Iterable[DefaultVoice], embed: Callable[[DefaultVoice], Awaitable], logger: logging.Logger
) -> None:
    """Compute every default voice's speaker embedding once (the studio's Who map places them all); a voice without
    one is logged."""
    for voice in voices:
        try:
            await embed(voice)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Default voice %s has no speaker embedding: %s", voice.label, exc)
        await asyncio.sleep(0)


def new_speaker_id() -> str:
    """The id of a session voice (POST /v2/speakers)."""
    return f"spk_{uuid.uuid4().hex[:12]}"


def speaker_label(explicit_label: str | None, file_name: str | None, *, fallback: str) -> str:
    """A session voice's label: the explicit label, else the uploaded file's stem, else ``fallback``."""
    if explicit_label and explicit_label.strip():
        return explicit_label.strip()
    if file_name:
        stem = Path(file_name).stem.strip()
        if stem:
            return stem
    return fallback


def voice_speaker_modes(modes: list[str], source_type: str, has_embedding: bool = True) -> list[str]:
    """The checkpoint's speaker ``modes`` a voice supports: an embedding voice only ``embedding``, an audio voice every
    mode (a clip without its embedding: not ``embedding``)."""
    if source_type == "embedding":
        return [m for m in modes if m == "embedding"]
    return list(modes) if has_embedding else [m for m in modes if m != "embedding"]


def serialize_default_voice(voice: DefaultVoice, *, modes: list[str], embedding_dim: int, disabled: list[str]) -> dict:
    """A default voice of GET /v2/speakers (``modes``: voice_speaker_modes; ``disabled``: its disabled controls)."""
    return {
        "id": voice.speaker_id,
        "label": voice.label,
        "source_type": voice.source_type,
        "scope": "default",
        "is_default": True,
        "original_name": voice.original_name,
        "created_at": voice.mtime,
        "has_preview": voice.source_type == "audio",
        "speaker_modes": modes,
        "embedding_dim": embedding_dim if "embedding" in modes else None,
        "transcript": voice.transcript,
        "disabled_controls": disabled,
    }


def serialize_session_voice(
    *,
    speaker_id: str,
    label: str,
    source_type: str,
    original_name: str | None,
    created_at: float,
    has_preview: bool,
    embedding_dim: int,
    modes: list[str],
    disabled: list[str],
) -> dict:
    """A session voice of GET /v2/speakers and POST /v2/speakers."""
    return {
        "id": speaker_id,
        "label": label,
        "source_type": source_type,
        "scope": "session",
        "is_default": False,
        "original_name": original_name,
        "created_at": created_at,
        "has_preview": has_preview,
        "speaker_modes": modes,
        "embedding_dim": embedding_dim if "embedding" in modes else None,
        "transcript": None,
        "disabled_controls": disabled,
    }


def normalize_session_id(session_id: str | None) -> str | None:
    """X-TTS-Session-ID, stripped; None when absent or blank."""
    if session_id is None:
        return None
    return session_id.strip() or None


def require_session_id(session_id: str | None) -> str:
    """X-TTS-Session-ID of a session-voice operation (ValueError without one: HTTP 400)."""
    normalized = normalize_session_id(session_id)
    if normalized is None:
        raise ValueError("Cached speaker operations require the X-TTS-Session-ID header.")
    return normalized
