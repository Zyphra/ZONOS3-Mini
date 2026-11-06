from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import io
import os
import secrets
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from zonos3_sgl.message import (
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchTTSFrontendMsg,
    TTSAudioReply,
    TTSCodesReply,
    TTSDecodeCodesMsg,
    TTSEncodeAudioMsg,
    TTSSamplingParams,
    TTSTokenizeMsg,
)
from zonos3 import languages
from zonos3.generation import DECODE_CONTEXT_FRAMES, SAMPLE_RATE, SAMPLING, TAG_CFG_MAX_SCALE, TAG_CFG_MIN_SCALE
from zonos3.tts import plan
from zonos3.tts.audio import (
    _decode_audio_bytes,
    _decode_wav_bytes,
    _transcode_audio_bytes_to_wav,
    pcm16_wav_bytes,
    write_pcm16_wav,
)
from zonos3.tts.directions import RAW_SPACE
from zonos3.tts.edit import edit_marker_rows
from zonos3.tts.edit_align import AsrWorker, FuseAlignWorker, fusealign_installed
from zonos3.tts.edit_runtime import EditModel, EditRuntime, FillJob, edit_capabilities
from zonos3.tts.steering import (
    SteeringBundle,
    check_controls,
    controls_document,
    disabled_controls,
    load_bundle,
    resolve_controls,
)
from zonos3.tts.markup import NONVERBAL_MARKUP_TAGS, trained_on_tags
from zonos3.server import plugins
from zonos3.server.edit import make_router as _edit_router
from zonos3.server.keepalive import KeepaliveMiddleware
from zonos3.server.projects import ProjectStore, make_router as _projects_router
from zonos3.server.schemas import (
    GenerateRequest,
    PhonemizeRequest,
    SpeakerCreateRequest,
    SpeechRequest,
)
from zonos3_sgl.utils.mp import ZmqAsyncPullQueue, ZmqAsyncPushQueue
from zonos3_sgl.utils.logger import init_logger

from .args import ServerArgs


logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None
_SPEAKER_EMBEDDERS = {}
_SPEAKER_EMBEDDER_LOCK = asyncio.Lock()
_SPEAKER_EMBEDDER_DEVICE = os.getenv("ZONOS3_SPEAKER_EMBEDDER_DEVICE", "cpu")
_SESSION_SPEAKER_CACHE: Dict[str, Dict[str, "CachedSpeakerReference"]] = {}
_SESSION_SPEAKER_CACHE_LOCK = asyncio.Lock()
_DEFAULT_SPEAKER_CACHE: Dict[Tuple[str, int], Dict[str, "DefaultSpeakerReference"]] = {}
_DEFAULT_SPEAKER_CACHE_LOCK = asyncio.Lock()
# The server's ASR worker (zonos3/workers/asr.py: Qwen3-ASR in .venv-asr; ZONOS3_ASR_PYTHON / ZONOS3_ASR_DEVICE):
# every speech recognition of speech editing and plugins. It starts on the first call.
_ASR = AsrWorker()
# The server's FuseAlign worker (zonos3/workers/fusealign.py in this interpreter; ZONOS3_FUSEALIGN_DEVICE picks its
# device): every forced alignment of speech editing and plugins. It starts on the first call.
_FUSEALIGN = FuseAlignWorker()
# Gold pronunciation lexicons, keyed by LMDB directory.
_PHONEME_LEXICON_CACHE: Dict[str, "PhonemeLexicon"] = {}
_DEFAULT_VOICE_AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
_DEFAULT_VOICE_EMBEDDING_EXTENSIONS = {".npy", ".npz"}


@dataclass
class CachedSpeakerReference:
    speaker_id: str
    label: str
    source_type: Literal["audio", "embedding"]
    embedding: torch.Tensor
    created_at: float
    original_name: str | None = None
    audio_bytes: bytes | None = None
    audio_media_type: str | None = None


@dataclass
class DefaultSpeakerReference:
    speaker_id: str
    label: str
    source_type: Literal["audio", "embedding"]
    path: Path
    mtime: float
    original_name: str
    embedding: torch.Tensor | None = None
    audio_bytes: bytes | None = None
    audio_media_type: str | None = None
    transcript: str | None = None  # <voice>.txt next to the audio (studio edit examples)


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


_PROJECT_STORE: ProjectStore | None = None


def _project_store() -> ProjectStore | None:
    """The studio's shared project store (zonos3.server.projects) under --studio-projects-dir; None when unset."""
    global _PROJECT_STORE
    config = get_global_state().config
    if getattr(config, "studio_projects_dir", None) is None:
        return None
    if _PROJECT_STORE is None:
        _PROJECT_STORE = ProjectStore(config.studio_projects_dir, config.studio_max_project_bytes,
                                      config.studio_max_media_bytes)
    return _PROJECT_STORE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[TTSAudioReply | TTSCodesReply]:
    if isinstance(msg, BatchTTSFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, (TTSAudioReply, TTSCodesReply))
            result.append(reply)
        return result
    if isinstance(msg, (TTSAudioReply, TTSCodesReply)):
        return [msg]
    raise TypeError(f"Unexpected frontend message type: {type(msg).__name__}")


def _model_supports_speaker_embedding(config: ServerArgs) -> bool:
    return bool(getattr(config.model_config, "speaker_enabled", False))


def _model_uses_speaker_reference_prefix(config: ServerArgs) -> bool:
    return bool(
        getattr(config.model_config, "speaker_reference_prefix_enabled", False)
    )


def _speaker_modes(config: ServerArgs) -> list[str]:
    modes = []
    if _model_uses_speaker_reference_prefix(config):
        modes.append("reference")
    if _model_supports_speaker_embedding(config):
        modes.append("embedding")
    return modes


# ----------------------------------------------------------------------------- condition rows (cond3 finetunes)
_CONDITION_LAYOUT = "[reference prefix | speaker row][loudness][fmax][speaking_rate] BOS text EOS"


def _condition_spec(config: ServerArgs) -> dict | None:
    """The checkpoint's condition rows (loudness / fmax / speaking-rate buckets): names, per-name bucket counts (the
    count is the unknown id), edges; None when the model has none."""
    model_config = config.model_config
    if not getattr(model_config, "condition_tokens_enabled", False):
        return None
    names = [str(item) for item in (getattr(model_config, "condition_names", None) or ("loudness", "fmax", "speaking_rate"))]
    count = int(getattr(model_config, "condition_count", len(names)) or len(names))
    counts = dict(zip(names[:count], (int(n) for n in model_config.condition_bucket_counts)))
    edges = {
        str(name): [float(x) for x in (values or ())]
        for name, values in (getattr(model_config, "condition_edges", None) or {}).items()
    }
    # flag rows (the encode-start finetunes): ids restricted to the flag's values, no edges
    flags = {str(name): dict(spec or {}) for name, spec in (getattr(model_config, "condition_flags", None) or {}).items()}
    return dict(
        enabled=True,
        names=names[:count],
        flags={name: flags[name] for name in names[:count] if name in flags},
        count=count,
        bucket_counts=counts,
        unknown_ids=dict(counts),
        edges=edges,
        units=dict(getattr(model_config, "condition_units", None) or {}),
        layout="[reference prefix | speaker row]" + "".join(f"[{name}]" for name in names[:count]) + " BOS text EOS",
    )


def _dual_codec(config: ServerArgs) -> dict | None:
    """The codec registry entry of the checkpoint's codec when it is a matched dual codec (the codecs with
    codec-matched silence frames), else None."""
    from zonos3.tokenizer.matched_codec import codec_names, entry

    codec = str(getattr(config.model_config, "audio_codec", "") or "")
    return entry(codec) if codec in codec_names() and entry(codec).get("family") == "dual" else None


def _silence_prefill_spec(config: ServerArgs) -> dict:
    """Whether codec-matched silence frames can be prefilled into the audio stream (2026-10-17), and the server
    default (zonos3.tts.plan.silence_prefill_default)."""
    codec = str(getattr(config.model_config, "audio_codec", "") or "")
    spec = _dual_codec(config)
    if spec is None:
        return dict(available=False, default_frames=None, codec=codec)
    hop, default = int(spec["hop"]), plan.silence_prefill_default()
    return dict(available=True, default_frames=default.frames, default_on=default.on,
                default_seconds=round(default.frames * hop / SAMPLE_RATE, 3), frame_seconds=hop / SAMPLE_RATE, codec=codec,
                note="codec-matched silence frames open the audio stream before the model starts (their audio is not emitted); "
                     "a request without silence_prefill_frames gets the server default, silence_prefill_frames=0 "
                     "turns it off and silence_prefill_frames=N sets the length")


def _model_supports_speaker(config: ServerArgs) -> bool:
    return _model_supports_speaker_embedding(
        config
    ) or _model_uses_speaker_reference_prefix(config)


def _model_speaker_dim(config: ServerArgs) -> int:
    return int(getattr(config.model_config, "speaker_embedding_dim", 128))


def _model_speaker_codec(config: ServerArgs) -> str | None:
    codec = getattr(config.model_config, "speaker_embedding_codec", None)
    return str(codec) if codec else None


_STREAM_DECODE_MODES = ("prefix", "whole")


def _model_prefix_stream_supported(config: ServerArgs) -> bool:
    """Progressive (prefix re-decode) streaming exists for the matched long-context codecs only."""
    from zonos3.tokenizer.matched_codec import codec_names

    return str(getattr(config.model_config, "audio_codec", "")) in codec_names()


def _default_stream_decode(config: ServerArgs) -> str | None:
    """The server default for a streaming response: progressive prefix re-decode whenever the codec supports it
    (first audio after ~15 frames instead of at EOS). ZONOS3_STREAM_DECODE_DEFAULT=whole restores the one exact
    decode at EOS for every request that does not ask for a mode."""
    wanted = os.environ.get("ZONOS3_STREAM_DECODE_DEFAULT", "prefix").strip().lower()
    return "prefix" if wanted == "prefix" and _model_prefix_stream_supported(config) else None


def _resolve_stream_decode(config: ServerArgs, value: str | None, streaming: bool = True) -> str | None:
    """GenerateRequest.stream_decode: None = the server default (prefix for streaming responses of the matched
    codecs, whole decode otherwise), "prefix", or "whole" (one exact decode at EOS). Non-streaming responses are
    always the whole decode."""
    if value is None:
        return _default_stream_decode(config) if streaming else None
    if value == "whole" or not streaming:
        return None
    if not _model_prefix_stream_supported(config):
        raise HTTPException(status_code=400, detail="stream_decode=prefix is only available for the matched long-context codecs")
    return value


def _adapter_accents(config: ServerArgs) -> list[str]:
    """The accents requests may name: every adapter but the speech-edit one."""
    if not config.accent_adapters:
        return []
    from zonos3.models.accent_format import edit_adapter, list_accent_adapters

    edit = edit_adapter(config.accent_adapters)
    return [name for name in list_accent_adapters(config.accent_adapters) if name != edit]


def _steering_sources(config: ServerArgs) -> tuple[int, ...]:
    """The Full AttnRes bank sources the engine steers, ascending (as Engine._init_steering): the steering bundle's
    hidden-space sources."""
    bundle = _steer_bundle(config)
    return tuple(sorted(bundle.hidden_sources)) if bundle is not None else ()


_STEER_BUNDLE_CACHE: Dict[str, SteeringBundle] = {}


def _steer_bundle(config: ServerArgs) -> SteeringBundle | None:
    """The --steer-bundle sets (loaded once)."""
    path = getattr(config, "steer_bundle", None)
    if not path:
        return None
    if path not in _STEER_BUNDLE_CACHE:
        bundle = load_bundle(path)
        _STEER_BUNDLE_CACHE[path] = bundle
        logger.info("Loaded steering bundle %s: %d directions in sets %s.", path, len(bundle.directions),
                    ", ".join(bundle.sets))
    return _STEER_BUNDLE_CACHE[path]


class _RulesContext:
    """zonos3.tts.plan.RulesContext over the server's arguments: the checkpoint's condition rows (``_condition_spec``)
    and speaker modes, its text tokenizer, the loaded accent adapters, steering bundle and LACI heads, the frame
    budget (``max_seq_len``), the reference prefix limit (the checkpoint's speaker_reference_max_seconds at its codec
    hop, the limit the tokenizer worker cuts inline reference codes to) and whether its codec has codec-matched
    silence frames."""

    steer_hint = "start it with --steer-bundle"

    def __init__(self, config: ServerArgs):
        self.config = config

    @functools.cached_property
    def _conditions(self) -> dict:
        return _condition_spec(self.config) or dict(names=[], bucket_counts={}, unknown_ids={}, edges={}, flags={})

    @property
    def n_codebooks(self) -> int:
        return int(self.config.model_config.n_codebooks)

    @property
    def codebook_size(self) -> int:
        return int(self.config.model_config.codebook_size)

    @property
    def hidden_size(self) -> int:
        return int(self.config.model_config.hidden_size)

    @property
    def condition_names(self) -> list[str]:
        return self._conditions["names"]

    @property
    def condition_bucket_counts(self) -> tuple[int, ...]:
        return tuple(self.config.model_config.condition_bucket_counts)

    @property
    def condition_edges(self) -> dict:
        return self._conditions["edges"]

    @property
    def condition_flags(self) -> dict:
        return self._conditions["flags"]

    @property
    def speaker_modes(self) -> list[str]:
        return _speaker_modes(self.config)

    @property
    def text_tokenizer_spec(self) -> dict | None:
        return _text_tokenizer_spec(self.config)

    @property
    def accent_names(self) -> list[str]:
        return _adapter_accents(self.config)

    @property
    def steer_bundle(self) -> SteeringBundle | None:
        return _steer_bundle(self.config)

    @property
    def steering_sources(self) -> tuple[int, ...]:
        return _steering_sources(self.config)

    @property
    def laci_languages(self) -> dict:
        return getattr(self.config, "laci_heads", None) or {}

    @property
    def max_frames(self) -> int:
        return int(self.config.max_seq_len)

    @property
    def reference_max_frames(self) -> int:
        from zonos3.tokenizer.matched_codec import entry

        hop = int(entry(str(self.config.tts_audio_codec))["hop"])
        return plan.reference_frame_limit(self.config.tts_speaker_reference_max_seconds, hop, SAMPLE_RATE)

    @property
    def silence_available(self) -> bool:
        return _dual_codec(self.config) is not None


def _apply_gain_pcm(audio_bytes: bytes, gain: float) -> bytes:
    """Scale float32 PCM bytes (the decoder's headroom: the iSTFT head is unbounded)."""
    if gain == 1.0 or not audio_bytes:
        return audio_bytes
    samples = np.frombuffer(audio_bytes, dtype=np.float32) * np.float32(gain)
    return samples.astype(np.float32).tobytes()


def _output_gain(config: ServerArgs, headroom_db: float | None) -> float:
    """Linear gain for a request: its headroom_db, else the matched decoder's registry default (0 dB otherwise)."""
    if headroom_db is None:
        codec = str(getattr(config.model_config, "audio_codec", ""))
        from zonos3.tokenizer.matched_codec import codec_names, decoder_headroom_db
        headroom_db = decoder_headroom_db(codec) if codec in codec_names() else 0.0
    return float(10.0 ** (-float(headroom_db) / 20.0))


def _apply_fade_out_pcm(
    audio_bytes: bytes, fade_out_ms: float, sample_rate: int = SAMPLE_RATE
) -> bytes:
    """Apply a cosine fade-out to the tail of float32 PCM bytes."""
    if fade_out_ms <= 0 or not audio_bytes:
        return audio_bytes
    samples = np.frombuffer(audio_bytes, dtype=np.float32).copy()
    n = min(len(samples), int(sample_rate * fade_out_ms / 1000.0))
    if n <= 0:
        return audio_bytes
    fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    samples[-n:] *= fade
    return samples.tobytes()


def _normalize_session_id(session_id: str | None) -> str | None:
    if session_id is None:
        return None
    normalized = session_id.strip()
    return normalized or None


def _require_session_id(session_id: str | None) -> str:
    normalized = _normalize_session_id(session_id)
    if normalized is None:
        raise ValueError("Cached speaker operations require the X-TTS-Session-ID header.")
    return normalized


def _default_speaker_label(
    explicit_label: str | None,
    file_name: str | None,
    *,
    fallback: str,
) -> str:
    if explicit_label and explicit_label.strip():
        return explicit_label.strip()
    if file_name:
        stem = Path(file_name).stem.strip()
        if stem:
            return stem
    return fallback


def _slerp_embeddings(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    v0 = v0.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    v1 = v1.detach().to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    if v0.shape != v1.shape:
        raise ValueError(
            f"Cannot blend speaker embeddings with different shapes {tuple(v0.shape)} and {tuple(v1.shape)}."
        )

    blend = float(t)
    if blend < 0.0 or blend > 1.0:
        raise ValueError("speaker_blend_t must be between 0.0 and 1.0.")

    v0_norm = torch.linalg.vector_norm(v0) + 1e-8
    v1_norm = torch.linalg.vector_norm(v1) + 1e-8
    v0_n = v0 / v0_norm
    v1_n = v1 / v1_norm
    omega = torch.acos(torch.clamp(torch.dot(v0_n, v1_n), -1.0, 1.0))
    if float(omega) < 1e-6:
        return ((1.0 - blend) * v0 + blend * v1).contiguous()

    sin_omega = torch.sin(omega)
    mixed = (
        (torch.sin((1.0 - blend) * omega) / sin_omega) * v0
        + (torch.sin(blend * omega) / sin_omega) * v1
    )
    return mixed.to(dtype=torch.float32, device="cpu").contiguous()


def _get_phoneme_lexicon(config: ServerArgs) -> "PhonemeLexicon | None":
    """Open (and cache) the configured gold pronunciation lexicon."""
    path = getattr(config, "tts_phoneme_lexicon", None)
    if not path:
        return None
    if path not in _PHONEME_LEXICON_CACHE:
        from zonos3.tts.phonemes import PhonemeLexicon

        _PHONEME_LEXICON_CACHE[path] = PhonemeLexicon(path)
        logger.info("Loaded pronunciation lexicon %s (%s).", path, ", ".join(_PHONEME_LEXICON_CACHE[path].languages))
    return _PHONEME_LEXICON_CACHE[path]


def _text_tokenizer_spec(config: ServerArgs) -> dict | None:
    """The served text tokenizer spec (EngineConfig.text_tokenizer_spec: the checkpoint's, with --phoneme-rows)."""
    return config.text_tokenizer_spec


@functools.lru_cache(maxsize=4)
def _model_trained_on_tags(model_path: str) -> bool:
    """zonos3.tts.markup.trained_on_tags, memoized per checkpoint path."""
    return trained_on_tags(model_path)


def _tag_cfg_capabilities(config: ServerArgs) -> dict:
    return {"available": True, "trained": _model_trained_on_tags(str(config.model_path)),
            "tags": list(NONVERBAL_MARKUP_TAGS), "min_scale": TAG_CFG_MIN_SCALE, "max_scale": TAG_CFG_MAX_SCALE,
            "default_scale": 1.0}


def _voice_speaker_modes(config: ServerArgs, source_type: str, has_embedding: bool = True) -> list[str]:
    """The checkpoint's speaker modes a voice supports: an embedding voice only ``embedding``, an audio voice every
    mode (a session clip cached without its embedding: ``reference`` only)."""
    modes = _speaker_modes(config)
    if source_type == "embedding":
        return [m for m in modes if m == "embedding"]
    return modes if has_embedding else [m for m in modes if m != "embedding"]


def _serialize_cached_speaker(config: ServerArgs, entry: CachedSpeakerReference) -> dict:
    dim = int(entry.embedding.numel())
    modes = _voice_speaker_modes(config, entry.source_type, has_embedding=dim > 0)
    return {
        "id": entry.speaker_id,
        "label": entry.label,
        "source_type": entry.source_type,
        "scope": "session",
        "is_default": False,
        "original_name": entry.original_name,
        "created_at": entry.created_at,
        "has_preview": bool(entry.audio_bytes),
        "speaker_modes": modes,
        "embedding_dim": dim if "embedding" in modes else None,
        "transcript": None,
        "disabled_controls": disabled_controls(_steer_bundle(config), entry.speaker_id),
    }


def _serialize_default_speaker(
    config: ServerArgs, entry: DefaultSpeakerReference, expected_dim: int
) -> dict:
    modes = _voice_speaker_modes(config, entry.source_type)
    return {
        "id": entry.speaker_id,
        "label": entry.label,
        "source_type": entry.source_type,
        "scope": "default",
        "is_default": True,
        "original_name": entry.original_name,
        "created_at": entry.mtime,
        "has_preview": entry.source_type == "audio",
        "speaker_modes": modes,
        "embedding_dim": expected_dim if "embedding" in modes else None,
        "transcript": entry.transcript,
        "disabled_controls": disabled_controls(_steer_bundle(config), entry.speaker_id),
    }


def _default_voice_root(config: ServerArgs) -> Path | None:
    raw = getattr(config, "tts_default_voices_dir", None)
    if not raw:
        return None
    return Path(str(raw)).expanduser()


def _default_voice_source_type(path: Path) -> Literal["audio", "embedding"] | None:
    name = path.name.lower()
    # An audio voice's sidecars for the GGUF servers (reference codes, speaker embedding) are no voice; this server
    # computes both from the clip with its own encoders.
    if name.endswith(".codes.npy") or name.endswith(".embedding.npy"):
        return None
    suffix = path.suffix.lower()
    if suffix in _DEFAULT_VOICE_EMBEDDING_EXTENSIONS:
        return "embedding"
    if suffix in _DEFAULT_VOICE_AUDIO_EXTENSIONS:
        return "audio"
    return None


def _default_voice_id(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
    return f"default_{digest}"


def _default_voice_label(path: Path) -> str:
    label = path.stem.replace("_", " ").replace("-", " ").strip()
    return label or path.name


def _read_voice_transcript(path: Path) -> str | None:
    """Text of the ``<voice>.txt`` sidecar next to a default voice, when present."""
    try:
        return path.with_suffix(".txt").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


async def _list_default_speakers(config: ServerArgs) -> list[DefaultSpeakerReference]:
    root = _default_voice_root(config)
    if root is None:
        return []
    if not root.exists() or not root.is_dir():
        logger.warning("TTS default voices directory does not exist or is not a directory: %s", root)
        return []

    root = root.resolve()
    expected_dim = _model_speaker_dim(config)
    cache_key = (str(root), expected_dim)

    async with _DEFAULT_SPEAKER_CACHE_LOCK:
        previous = _DEFAULT_SPEAKER_CACHE.get(cache_key, {})
        current: dict[str, DefaultSpeakerReference] = {}
        for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
            source_type = _default_voice_source_type(path)
            if source_type is None:
                continue

            resolved_path = path.resolve()
            try:
                stat = resolved_path.stat()
            except OSError:
                continue

            speaker_id = _default_voice_id(root, path)
            existing = previous.get(speaker_id)
            if (
                existing is not None
                and existing.path == resolved_path
                and existing.mtime == stat.st_mtime
                and existing.source_type == source_type
            ):
                existing.transcript = _read_voice_transcript(existing.path)
                current[speaker_id] = existing
                continue

            current[speaker_id] = DefaultSpeakerReference(
                speaker_id=speaker_id,
                label=_default_voice_label(resolved_path),
                source_type=source_type,
                path=resolved_path,
                mtime=stat.st_mtime,
                original_name=resolved_path.name,
                transcript=_read_voice_transcript(resolved_path),
            )

        _DEFAULT_SPEAKER_CACHE[cache_key] = current
        return sorted(current.values(), key=lambda entry: entry.label.lower())


async def _get_default_speaker(
    config: ServerArgs,
    speaker_id: str,
) -> DefaultSpeakerReference | None:
    speakers = await _list_default_speakers(config)
    return next((entry for entry in speakers if entry.speaker_id == speaker_id), None)


def _read_default_voice_file(entry: DefaultSpeakerReference) -> bytes:
    try:
        return entry.path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Failed to read default voice '{entry.label}'.") from exc


async def _default_speaker_preview(entry: DefaultSpeakerReference) -> tuple[bytes, str]:
    if entry.source_type != "audio":
        raise ValueError("Default embedding speakers do not include preview audio.")
    if entry.audio_bytes is None:
        entry.audio_bytes = _transcode_audio_bytes_to_wav(_read_default_voice_file(entry))
        entry.audio_media_type = "audio/wav"
    return entry.audio_bytes, entry.audio_media_type or "audio/wav"


async def _load_default_speaker_embedding(
    config: ServerArgs,
    entry: DefaultSpeakerReference,
) -> torch.Tensor:
    expected_dim = _model_speaker_dim(config)
    if entry.embedding is None:
        if entry.source_type == "embedding":
            entry.embedding = _load_embedding_vector(
                _read_default_voice_file(entry),
                expected_dim=expected_dim,
                file_name=entry.original_name,
            )
        else:
            wav_bytes, _ = await _default_speaker_preview(entry)
            wav, sample_rate = _decode_wav_bytes(wav_bytes)
            entry.embedding = await _compute_speaker_embedding_from_waveform(
                wav,
                sample_rate,
                expected_dim=expected_dim,
                speaker_codec=_model_speaker_codec(config),
            )

    return entry.embedding.detach().to(dtype=torch.float32, device="cpu").contiguous().clone()


async def _cache_speaker_reference(
    *,
    session_id: str,
    label: str,
    source_type: Literal["audio", "embedding"],
    embedding: torch.Tensor,
    original_name: str | None = None,
    audio_bytes: bytes | None = None,
    audio_media_type: str | None = None,
) -> CachedSpeakerReference:
    entry = CachedSpeakerReference(
        speaker_id=f"spk_{uuid.uuid4().hex[:12]}",
        label=label,
        source_type=source_type,
        embedding=embedding.detach().to(dtype=torch.float32, device="cpu").contiguous().clone(),
        created_at=time.time(),
        original_name=original_name,
        audio_bytes=audio_bytes,
        audio_media_type=audio_media_type,
    )
    async with _SESSION_SPEAKER_CACHE_LOCK:
        session_cache = _SESSION_SPEAKER_CACHE.setdefault(session_id, {})
        session_cache[entry.speaker_id] = entry
    return entry


async def _list_cached_speakers(session_id: str) -> list[CachedSpeakerReference]:
    async with _SESSION_SPEAKER_CACHE_LOCK:
        session_cache = _SESSION_SPEAKER_CACHE.get(session_id, {})
        return sorted(
            session_cache.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )


async def _get_cached_speaker(session_id: str, speaker_id: str) -> CachedSpeakerReference:
    async with _SESSION_SPEAKER_CACHE_LOCK:
        session_cache = _SESSION_SPEAKER_CACHE.get(session_id, {})
        entry = session_cache.get(speaker_id)
    if entry is None:
        raise ValueError(f"Unknown cached speaker id '{speaker_id}'.")
    return entry


def _decode_base64_blob(data: str, field_name: str) -> bytes:
    payload = data.strip()
    if payload.startswith("data:"):
        if "," not in payload:
            raise ValueError(f"Invalid data URL for {field_name}.")
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:  # pragma: no cover - branch depends on invalid user payload
        raise ValueError(f"{field_name} must be valid base64-encoded data.") from exc


def _load_embedding_vector(
    embedding_bytes: bytes,
    *,
    expected_dim: int,
    file_name: str | None = None,
) -> torch.Tensor:
    try:
        loaded = np.load(io.BytesIO(embedding_bytes), allow_pickle=False)
    except Exception as exc:
        file_desc = file_name or "Speaker embedding file"
        raise ValueError(
            f"{file_desc} must be a valid .npy or .npz file saved without pickle."
        ) from exc

    arr = None
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "emb" in loaded.files:
                arr = loaded["emb"]
            elif len(loaded.files) == 1:
                arr = loaded[loaded.files[0]]
            else:
                raise ValueError(
                    "Embedding archive must contain an 'emb' array or exactly one array."
                )
        finally:
            loaded.close()
    else:
        arr = loaded

    if isinstance(arr, np.ndarray) and arr.dtype.names:
        if "emb" in arr.dtype.names:
            arr = arr["emb"]
        elif len(arr.dtype.names) == 1:
            arr = arr[arr.dtype.names[0]]
        else:
            raise ValueError("Structured embedding arrays must contain an 'emb' field.")

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        vector = arr
    elif arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("Speaker embedding file is empty.")
        # Multi-row embedding files come from segmented exports; average them into one speaker vector.
        vector = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    else:
        raise ValueError(
            f"Unsupported embedding array shape {tuple(arr.shape)}. Expected (D,) or (N, D)."
        )

    vector = np.ascontiguousarray(vector, dtype=np.float32)
    if vector.shape[-1] != expected_dim:
        raise ValueError(
            f"Reference embedding dimension mismatch. Model expects {expected_dim}, "
            f"but loaded file has {vector.shape[-1]}."
        )

    return torch.from_numpy(vector).to(dtype=torch.float32, device="cpu")


async def _get_speaker_embedder(expected_dim: int, speaker_codec: str | None = None):
    global _SPEAKER_EMBEDDERS
    from zonos3.models.speaker_cloning import resolve_speaker_embedding_backend

    backend_key = resolve_speaker_embedding_backend(speaker_codec, expected_dim)
    if backend_key in _SPEAKER_EMBEDDERS:
        return _SPEAKER_EMBEDDERS[backend_key]

    async with _SPEAKER_EMBEDDER_LOCK:
        if backend_key in _SPEAKER_EMBEDDERS:
            return _SPEAKER_EMBEDDERS[backend_key]
        try:
            from zonos3.models.speaker_cloning import create_speaker_embedder

            embedder = create_speaker_embedder(
                speaker_codec,
                expected_dim,
                device=_SPEAKER_EMBEDDER_DEVICE,
            )
        except Exception as exc:  # pragma: no cover - depends on environment deps
            raise RuntimeError(
                "Speaker embedding backend failed to load. Ensure torchaudio, "
                "transformers, and huggingface_hub are installed and that the "
                f"{backend_key} speaker model is accessible. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        _SPEAKER_EMBEDDERS[backend_key] = embedder
    return _SPEAKER_EMBEDDERS[backend_key]


async def _compute_speaker_embedding_from_audio(
    speaker_audio_base64: str,
    expected_dim: int,
    speaker_codec: str | None = None,
) -> torch.Tensor:
    audio_bytes = _decode_base64_blob(speaker_audio_base64, "speaker_audio_base64")
    wav, sample_rate = _decode_audio_bytes(audio_bytes)
    return await _compute_speaker_embedding_from_waveform(
        wav,
        sample_rate,
        expected_dim=expected_dim,
        speaker_codec=speaker_codec,
    )


async def _compute_speaker_embedding_from_waveform(
    wav: torch.Tensor,
    sample_rate: int,
    *,
    expected_dim: int,
    speaker_codec: str | None = None,
) -> torch.Tensor:
    embedder = await _get_speaker_embedder(expected_dim, speaker_codec)
    with torch.inference_mode():
        output = embedder(wav, sample_rate)

    if isinstance(output, tuple):
        candidates = [
            tensor.squeeze(0).to(dtype=torch.float32, device="cpu")
            for tensor in output
        ]
    else:
        candidates = [output.squeeze(0).to(dtype=torch.float32, device="cpu")]

    for candidate in candidates:
        if candidate.numel() == expected_dim:
            return candidate

    raise ValueError(
        f"Reference embedding dimension mismatch. Model expects {expected_dim}, "
        f"but speaker encoder produced {', '.join(str(c.numel()) for c in candidates)}."
    )


def _compute_speaker_embedding_from_file(
    speaker_embedding_base64: str,
    expected_dim: int,
    speaker_embedding_name: str | None = None,
) -> torch.Tensor:
    embedding_bytes = _decode_base64_blob(
        speaker_embedding_base64,
        "speaker_embedding_base64",
    )
    return _load_embedding_vector(
        embedding_bytes,
        expected_dim=expected_dim,
        file_name=speaker_embedding_name,
    )


async def _resolve_cached_speaker_embedding(
    *,
    config: ServerArgs,
    session_id: str | None,
    speaker_id: str,
) -> torch.Tensor | None:
    if not _model_supports_speaker_embedding(config):
        logger.info(
            "Cached speaker reference provided, but current model does not support speaker embeddings; ignoring."
        )
        return None

    default_entry = await _get_default_speaker(config, speaker_id)
    if default_entry is not None:
        return await _load_default_speaker_embedding(config, default_entry)

    resolved_session_id = _require_session_id(session_id)
    entry = await _get_cached_speaker(resolved_session_id, speaker_id)
    expected_dim = _model_speaker_dim(config)
    actual_dim = int(entry.embedding.numel())
    if actual_dim == 0:
        raise ValueError(f"Cached speaker '{entry.label}' holds a reference clip only; it has no embedding for speaker_mode=embedding.")
    if actual_dim != expected_dim:
        raise ValueError(
            f"Cached speaker '{entry.label}' has {actual_dim} dimensions, but the current model expects {expected_dim}."
        )
    return entry.embedding.detach().to(dtype=torch.float32, device="cpu").contiguous().clone()


async def _resolve_speaker_embedding(
    *,
    config: ServerArgs,
    session_id: str | None = None,
    speaker_audio_base64: str | None = None,
    speaker_audio_name: str | None = None,
    speaker_embedding_base64: str | None = None,
    speaker_embedding_name: str | None = None,
    speaker_embedding_id: str | None = None,
    speaker_blend_embedding_id_a: str | None = None,
    speaker_blend_embedding_id_b: str | None = None,
    speaker_blend_t: float | None = None,
) -> torch.Tensor | None:
    direct_source_count = sum(
        bool(value) for value in (speaker_audio_base64, speaker_embedding_base64)
    )
    if direct_source_count > 1:
        raise ValueError(
            "Provide either a reference audio file or a saved embedding file, not both."
        )

    has_direct_source = direct_source_count == 1
    has_cached_source = bool(speaker_embedding_id)
    has_blended_source = bool(speaker_blend_embedding_id_a or speaker_blend_embedding_id_b)
    selected_modes = sum(bool(mode) for mode in (has_direct_source, has_cached_source, has_blended_source))
    if selected_modes == 0:
        return None
    if selected_modes > 1:
        raise ValueError(
            "Provide speaker conditioning via upload, one cached speaker id, or two cached ids for blending."
        )

    if has_blended_source:
        if not speaker_blend_embedding_id_a or not speaker_blend_embedding_id_b:
            raise ValueError("Provide both cached speaker ids to blend between two speakers.")
        speaker_a = await _resolve_cached_speaker_embedding(
            config=config,
            session_id=session_id,
            speaker_id=speaker_blend_embedding_id_a,
        )
        speaker_b = await _resolve_cached_speaker_embedding(
            config=config,
            session_id=session_id,
            speaker_id=speaker_blend_embedding_id_b,
        )
        if speaker_a is None or speaker_b is None:
            return None
        return _slerp_embeddings(
            speaker_a,
            speaker_b,
            0.5 if speaker_blend_t is None else speaker_blend_t,
        )

    if has_cached_source:
        assert speaker_embedding_id is not None
        return await _resolve_cached_speaker_embedding(
            config=config,
            session_id=session_id,
            speaker_id=speaker_embedding_id,
        )

    if not _model_supports_speaker_embedding(config):
        logger.info(
            "Speaker reference provided, but current model does not support speaker embeddings; ignoring."
        )
        return None

    expected_dim = _model_speaker_dim(config)
    if speaker_embedding_base64:
        return _compute_speaker_embedding_from_file(
            speaker_embedding_base64,
            expected_dim=expected_dim,
            speaker_embedding_name=speaker_embedding_name,
        )
    assert speaker_audio_base64 is not None
    return await _compute_speaker_embedding_from_audio(
        speaker_audio_base64,
        expected_dim=expected_dim,
        speaker_codec=_model_speaker_codec(config),
    )


async def _resolve_speaker_reference_audio(
    *,
    config: ServerArgs,
    session_id: str | None,
    speaker_audio_base64: str | None,
    speaker_embedding_base64: str | None,
    speaker_embedding_id: str | None,
    speaker_blend_embedding_id_a: str | None,
    speaker_blend_embedding_id_b: str | None,
) -> bytes | None:
    """Resolve the waveform required by raw-reference prefix checkpoints: an uploaded clip, a default voice
    (--tts-default-voices-dir) or a clip cached in this session through POST /v2/speakers."""
    if any((speaker_embedding_base64, speaker_blend_embedding_id_a, speaker_blend_embedding_id_b)):
        raise ValueError(
            "This checkpoint conditions on raw reference audio and does not accept "
            "speaker embedding files or embedding blends."
        )
    if speaker_embedding_id:
        if speaker_audio_base64:
            raise ValueError("Provide either a reference audio file or a cached speaker id, not both.")
        default_entry = await _get_default_speaker(config, speaker_embedding_id)
        if default_entry is not None:
            if default_entry.source_type != "audio":
                raise ValueError(
                    f"Default voice '{default_entry.label}' is an embedding file; this checkpoint needs reference audio."
                )
            audio_bytes, _ = await _default_speaker_preview(default_entry)   # transcoded WAV, cached on the entry
            return audio_bytes
        entry = await _get_cached_speaker(_require_session_id(session_id), speaker_embedding_id)
        if not entry.audio_bytes:
            raise ValueError(f"Cached speaker '{entry.label}' has no reference audio.")
        return entry.audio_bytes
    if speaker_audio_base64 is None:
        return None
    return _decode_base64_blob(speaker_audio_base64, "speaker_audio_base64")


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[TTSAudioReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    start_time_map: Dict[int, float] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        self.start_time_map[uid] = time.perf_counter()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    # Replies can keep arriving after a client abort; drop them.
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map.get(uid)
            if pending is None:
                # The request was aborted while we were waiting.
                return
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        self.start_time_map.pop(uid, None)
        logger.warning("Aborting request for user %s", uid)

    async def stream_tts_audio(self, uid: int, fade_out_ms: float = 0.0, gain: float = 1.0):
        """Stream TTS audio chunks as raw PCM bytes.

        When fade_out_ms > 0, the last fade_out_ms of audio is withheld until
        the stream finishes so a cosine fade-out can be applied to the tail.
        """
        t_start = self.start_time_map.pop(uid, None)
        first_byte = True
        total_audio_bytes = 0
        chunk_count = 0
        fade_bytes = int(SAMPLE_RATE * max(0.0, fade_out_ms) / 1000.0) * 4
        held = b""
        async for ack in self.wait_for_ack(uid):
            if isinstance(ack, TTSAudioReply):
                if ack.audio_data:
                    if first_byte and t_start is not None:
                        logger.info(
                            "uid=%d TTS TTFB=%.3fms", uid,
                            (time.perf_counter() - t_start) * 1000,
                        )
                        first_byte = False
                    total_audio_bytes += len(ack.audio_data)
                    chunk_count += 1
                    data = _apply_gain_pcm(ack.audio_data, gain)
                    if fade_bytes > 0:
                        held += data
                        if len(held) > fade_bytes:
                            yield held[:-fade_bytes]
                            held = held[-fade_bytes:]
                    else:
                        yield data
                if ack.finished:
                    break
        if held:
            yield _apply_fade_out_pcm(held, fade_out_ms)
        if t_start is not None:
            e2e_s = time.perf_counter() - t_start
            audio_s = total_audio_bytes / (SAMPLE_RATE * 4)  # float32 PCM
            logger.info(
                "uid=%d TTS completed: E2E=%.3fms, audio=%.2fs, chunks=%d, RTF=%.2fx",
                uid, e2e_s * 1000, audio_s, chunk_count,
                audio_s / e2e_s if e2e_s > 0 else 0,
            )

    async def wait_for_tts_audio(self, uid: int, fade_out_ms: float = 0.0, gain: float = 1.0):
        """Collect all TTS audio data and return as single bytes object."""
        return (await self.wait_for_tts_reply(uid, fade_out_ms, gain))[0]

    async def wait_for_tts_reply(self, uid: int, fade_out_ms: float = 0.0, gain: float = 1.0) -> tuple[bytes, dict | None]:
        """All TTS audio data as one bytes object, and the LACI record of a LACI request (else None)."""
        t_start = self.start_time_map.pop(uid, None)
        audio_data = b""
        laci = None
        async for ack in self.wait_for_ack(uid):
            if isinstance(ack, TTSAudioReply):
                if ack.audio_data:
                    audio_data += ack.audio_data
                if ack.finished:
                    laci = ack.laci
                    break
        # wait_for_ack stays suspended after the final reply: drop the request's queue here
        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)
        audio_data = _apply_fade_out_pcm(_apply_gain_pcm(audio_data, gain), fade_out_ms)
        if t_start is not None:
            e2e_s = time.perf_counter() - t_start
            audio_s = len(audio_data) / (SAMPLE_RATE * 4)
            logger.info(
                "uid=%d TTS completed (non-stream): E2E=%.3fms, audio=%.2fs, RTF=%.2fx",
                uid, e2e_s * 1000, audio_s,
                audio_s / e2e_s if e2e_s > 0 else 0,
            )
        return audio_data, laci

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


async def _warm_default_speaker_embeddings(config: ServerArgs) -> None:
    """Compute every default voice's speaker embedding once (the studio's Who map places them all)."""
    if not _model_supports_speaker_embedding(config):
        return
    for entry in await _list_default_speakers(config):
        try:
            await _load_default_speaker_embedding(config, entry)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Default voice %s has no speaker embedding: %s", entry.label, exc)
        await asyncio.sleep(0)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None and _steer_bundle(_GLOBAL_STATE.config) is not None:
        asyncio.get_running_loop().create_task(_warm_default_speaker_embeddings(_GLOBAL_STATE.config))
    yield
    # shutdown code here
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()
    _ASR.close()
    for p in _PLUGINS:
        p.close()
    _FUSEALIGN.close()


# The spec of this API is api/openapi.yaml; FastAPI's generated docs would describe a second one.
app = FastAPI(title="Zonos3 API Server", version="2.0.0", lifespan=lifespan,
              openapi_url=None, docs_url=None, redoc_url=None)

# Innermost, so CORS and the access check wrap its early status line (zonos3.server.keepalive). The /v2/edit
# prefix covers /v2/edit/transcribe and /v2/edit/align; plugins.mount adds each plugin's paths before serving.
_KEEPALIVE_PATHS = ["/v2/edit"]
app.add_middleware(KeepaliveMiddleware, paths=_KEEPALIVE_PATHS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # Studio clients on another origin (?server=) read these response headers.
    expose_headers=["X-Audio-Sample-Rate", "X-Audio-Channels", "X-Audio-Format", "X-Control-Span-Follow", "X-Seed"],
)


@app.middleware("http")
async def optional_private_access(request: Request, call_next):
    """Keep the inference UI and API private when ZONOS3_ACCESS_TOKEN is set.

    Clients send ``Authorization: Bearer <token>``; a browser opens
    ``/?access_token=<token>`` once and then carries an HttpOnly cookie.
    """
    token = os.environ.get("ZONOS3_ACCESS_TOKEN")
    if not token:
        return await call_next(request)
    candidate = request.headers.get("authorization", "").removeprefix("Bearer ")
    if not candidate:
        candidate = request.cookies.get("zonos3_access", "")
    if candidate and secrets.compare_digest(candidate, token):
        response = await call_next(request)
    elif request.method == "GET" and request.url.path == "/" and (
        candidate := request.query_params.get("access_token", "")
    ) and secrets.compare_digest(candidate, token):
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            "zonos3_access", token, httponly=True,
            secure=request.headers.get("x-forwarded-proto") == "https",
            samesite="strict",
        )
    else:
        response = Response("Access denied", status_code=403, media_type="text/plain")
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/v2/media/") and response.status_code in (200, 206, 304):
        # content-addressed and immutable: the browser may keep it (private: never a shared cache)
        response.headers["Cache-Control"] = "private, max-age=31536000, immutable"
    else:
        response.headers["Cache-Control"] = "no-store"
    return response


# Studio projects on the server (shared by every studio browser, kept across restarts): /v2/projects, /v2/media.
app.include_router(_projects_router(_project_store), prefix="/v2")


@app.get("/v2/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return {"object": "list", "data": [{"id": "zonos3", "object": "model", "created": 0, "owned_by": "zyphra",
                                        "root": str(state.config.model_path)}]}


# =============================================================================
# TTS Endpoints
# =============================================================================


async def _resolve_request_speaker_conditioning(
    config: ServerArgs,
    req: GenerateRequest,
    session_id: str | None,
    speaker_mode: str | None,
) -> tuple[bytes | None, torch.Tensor | None]:
    """(reference audio, speaker embedding) for a request.

    ``speaker_mode`` (zonos3.tts.plan.request_head) picks the conditioning
    path: "reference" takes a waveform (upload, default voice or session clip)
    and no embedding, "embedding" the reverse.
    """
    if speaker_mode == "reference":
        speaker_reference_audio = await _resolve_speaker_reference_audio(
            config=config,
            session_id=session_id,
            speaker_audio_base64=req.speaker_audio_base64,
            speaker_embedding_base64=req.speaker_embedding_base64,
            speaker_embedding_id=req.speaker_embedding_id,
            speaker_blend_embedding_id_a=req.speaker_blend_embedding_id_a,
            speaker_blend_embedding_id_b=req.speaker_blend_embedding_id_b,
        )
        speaker_embedding = None
    else:
        speaker_reference_audio = None
        speaker_embedding = await _resolve_speaker_embedding(
            config=config,
            session_id=session_id,
            speaker_audio_base64=req.speaker_audio_base64,
            speaker_audio_name=req.speaker_audio_name,
            speaker_embedding_base64=req.speaker_embedding_base64,
            speaker_embedding_name=req.speaker_embedding_name,
            speaker_embedding_id=req.speaker_embedding_id,
            speaker_blend_embedding_id_a=req.speaker_blend_embedding_id_a,
            speaker_blend_embedding_id_b=req.speaker_blend_embedding_id_b,
            speaker_blend_t=req.speaker_blend_t,
        )
    return speaker_reference_audio, speaker_embedding


# The speaker input fields a SpeechRequest `voice` cannot be combined with.
_SPEAKER_INPUT_FIELDS = (
    "speaker_audio_base64", "speaker_audio_name", "speaker_reference_codes", "speaker_embedding_base64",
    "speaker_embedding_name", "speaker_embedding_id", "speaker_blend_embedding_id_a", "speaker_blend_embedding_id_b",
    "speaker_blend_t",
)


@app.post("/v1/audio/speech")
async def create_speech(
    req: SpeechRequest,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """OpenAI-compatible speech endpoint: the /v2/generate request of ``input`` (pcm streams, wav is buffered), with
    ``voice`` (a speaker id of GET /v2/speakers) as its speaker_embedding_id."""
    logger.debug("Received TTS speech request: %s", req.input[:50])
    state = get_global_state()
    fields = req.model_dump(exclude={"model", "input", "voice", "response_format"})
    if req.voice is not None:
        if any(fields[name] is not None for name in _SPEAKER_INPUT_FIELDS):
            raise HTTPException(status_code=400, detail="voice cannot be combined with speaker_* fields")
        session_id = _normalize_session_id(x_tts_session_id)
        known = await _get_default_speaker(state.config, req.voice) is not None
        if not known and session_id is not None:
            async with _SESSION_SPEAKER_CACHE_LOCK:
                known = req.voice in _SESSION_SPEAKER_CACHE.get(session_id, {})
        if not known:
            raise HTTPException(status_code=400, detail=f"unknown voice '{req.voice}'")
        fields["speaker_embedding_id"] = req.voice
    request = GenerateRequest(text=req.input, stream=req.response_format == "pcm", format=req.response_format, **fields)
    return await _generate_response(state, request, x_tts_session_id)


async def _tts_generate_msg(
    state: FrontendManager,
    req: GenerateRequest,
    session_id: str | None,
    *,
    laci_target_frames: float | None = None,
    laci_adjust_rate: bool = True,
) -> TTSTokenizeMsg:
    """The tokenizer message of a /v2/generate request (HTTP 400/500 on an invalid one), with a new uid and the seed
    it samples with: the request rules of zonos3.tts.plan (``request_head``, the request's speaker inputs resolved
    for its speaker mode, ``request_rules``); the tokenizer worker builds the prompt rows.
    ``laci_target_frames`` / ``laci_adjust_rate``: a LACI take's duration target (zonos3.server.plugins.LaciTakeRequest;
    needs req.laci)."""
    ctx = _RulesContext(state.config)
    try:
        request = plan.parse_request(req.model_dump())
        head = plan.request_head(ctx, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        speaker_reference_audio, speaker_embedding = (
            await _resolve_request_speaker_conditioning(state.config, req, session_id, head.mode)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        reference_codes = plan.reference_codes(ctx, request.speaker_reference_codes)
        speaker = plan.Speaker(head.mode, speaker_embedding, reference_codes, speaker_reference_audio)
        rules = plan.request_rules(ctx, request, head, speaker, duration_target=laci_target_frames is not None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stream_decode = _resolve_stream_decode(state.config, req.stream_decode, streaming=bool(req.stream))
    # sgl's own LACI rule: a retake in another speaking-rate bucket changes the condition rows, which only the naive
    # cache never serves stale from a cached prompt prefix
    if laci_target_frames is not None and laci_adjust_rate and getattr(state.config, "cache_type", "naive") != "naive":
        raise HTTPException(status_code=400, detail="A LACI duration target that adjusts the speaking rate needs --cache-type naive")
    accent, accent_strength = rules.accent or (None, 1.0)
    return TTSTokenizeMsg(
        uid=state.new_user(),
        text=req.text,
        language=rules.language,
        text_normalization=req.text_normalization,
        silence_prefill_frames=req.silence_prefill_frames,
        decode_context=req.decode_context,
        decode_context_frames=int(req.decode_context_frames or DECODE_CONTEXT_FRAMES),
        sampling_params=TTSSamplingParams(
            temperature=req.temperature,
            topk=req.topk,
            top_p=req.top_p,
            min_p=req.min_p,
            max_tokens=rules.max_tokens,
            repetition_window=req.repetition_window,
            repetition_penalty=req.repetition_penalty,
            repetition_codebooks=req.repetition_codebooks,
            seed=rules.seed,
            controls_cfg_scale=req.controls_cfg_scale,
            speaker_cfg_scale=req.speaker_cfg_scale,
            tag_cfg_scale=req.tag_cfg_scale,
            stream_decode=stream_decode,
        ),
        speaker_embedding=rules.embedding,
        speaker_reference_audio=speaker_reference_audio,
        speaker_reference_codes=reference_codes,
        steering_delta=rules.hidden_delta,
        condition_buckets=rules.condition_ids,
        accent=accent,
        accent_strength=accent_strength,
        steer_schedule=rules.schedule,
        steer_follow_chars=rules.follow_chars,
        steer_follow_gate=rules.follow_gate,
        laci=req.laci,
        laci_target_frames=laci_target_frames,
        laci_adjust_rate=laci_adjust_rate,
    )


async def _generate_response(state: FrontendManager, req: GenerateRequest, session_id: str | None) -> Response:
    """Run a GenerateRequest: streamed or buffered float32 PCM, or a buffered 16-bit WAV."""
    if req.stream and req.format == "wav":
        raise HTTPException(status_code=400, detail="format=wav needs stream=false")
    msg = await _tts_generate_msg(state, req, session_id)
    uid = msg.uid
    await state.send_one(msg)

    async def _abort():
        await state.abort_user(uid)

    headers = {"X-Audio-Sample-Rate": str(SAMPLE_RATE), "X-Audio-Channels": "1",
               "X-Audio-Format": "pcm16" if req.format == "wav" else "float32",
               "X-Seed": str(msg.sampling_params.seed)}
    if req.control_spans:
        headers["X-Control-Span-Follow"] = "reading" if msg.steer_follow_chars is not None else "rate"
    gain = _output_gain(state.config, req.headroom_db)
    if req.stream:
        return StreamingResponse(
            state.stream_tts_audio(uid, fade_out_ms=req.fade_out_ms, gain=gain),
            media_type="audio/pcm",
            headers=headers,
            background=BackgroundTask(_abort),
        )
    audio_data = await state.wait_for_tts_audio(uid, fade_out_ms=req.fade_out_ms, gain=gain)
    if req.format == "wav":
        return Response(content=pcm16_wav_bytes(np.frombuffer(audio_data, np.float32), SAMPLE_RATE),
                        media_type="audio/wav", headers=headers)
    return Response(content=audio_data, media_type="audio/pcm", headers=headers)


@app.post("/v2/generate")
async def tts_generate(
    req: GenerateRequest,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """Synthesize speech (api/openapi.yaml GenerateRequest)."""
    logger.debug("Received TTS generate request: %s", req.text[:50])
    return await _generate_response(get_global_state(), req, x_tts_session_id)


@app.post("/v2/speakers")
async def tts_cache_speaker(
    req: SpeakerCreateRequest,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """Cache a speaker embedding for later selection in the current session."""
    state = get_global_state()
    try:
        session_id = _require_session_id(x_tts_session_id)
        if _model_uses_speaker_reference_prefix(state.config):
            # Raw-reference checkpoint: the cache holds the clip itself, plus its embedding when the checkpoint
            # also conditions on embeddings. An embedding file (e.g. a voice designed in the studio) is cached
            # alone on such dual-path checkpoints: speaker_mode=embedding only.
            raw_audio_base64 = req.speaker_audio_base64
            if req.speaker_embedding_base64:
                if not _model_supports_speaker_embedding(state.config):
                    raise ValueError(
                        "This checkpoint conditions on raw reference audio and does not accept speaker embedding files."
                    )
                if raw_audio_base64:
                    raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")
                embedding_name = req.speaker_embedding_name
                entry = await _cache_speaker_reference(
                    session_id=session_id,
                    label=_default_speaker_label(req.label, embedding_name, fallback="Speaker embedding"),
                    source_type="embedding",
                    embedding=_compute_speaker_embedding_from_file(
                        req.speaker_embedding_base64, expected_dim=_model_speaker_dim(state.config),
                        speaker_embedding_name=embedding_name),
                    original_name=embedding_name,
                )
                return _serialize_cached_speaker(state.config, entry)
            if not raw_audio_base64:
                raise ValueError("Provide an audio file to cache as the speaker reference.")
            raw_name = req.speaker_audio_name
            raw_bytes = _decode_base64_blob(raw_audio_base64, "speaker_audio_base64")
            raw_wav_bytes = _transcode_audio_bytes_to_wav(raw_bytes)
            raw_embedding = torch.empty(0)
            if _model_supports_speaker_embedding(state.config):     # both paths: keep the clip and its embedding
                raw_wav, raw_rate = _decode_wav_bytes(raw_wav_bytes)
                raw_embedding = await _compute_speaker_embedding_from_waveform(
                    raw_wav, raw_rate, expected_dim=_model_speaker_dim(state.config), speaker_codec=_model_speaker_codec(state.config))
            entry = await _cache_speaker_reference(
                session_id=session_id,
                label=_default_speaker_label(req.label, raw_name, fallback="Speaker audio"),
                source_type="audio",
                embedding=raw_embedding,
                original_name=raw_name,
                audio_bytes=raw_wav_bytes,
                audio_media_type="audio/wav",
            )
            return _serialize_cached_speaker(state.config, entry)
        if not _model_supports_speaker_embedding(state.config):
            raise ValueError("Current model does not support speaker embeddings.")

        speaker_audio_base64 = req.speaker_audio_base64
        direct_source_count = sum(
            bool(value)
            for value in (speaker_audio_base64, req.speaker_embedding_base64)
        )
        if direct_source_count == 0:
            raise ValueError("Provide an audio file or saved embedding file to cache.")
        if direct_source_count > 1:
            raise ValueError(
                "Provide either a reference audio file or a saved embedding file, not both."
            )

        expected_dim = _model_speaker_dim(state.config)
        if speaker_audio_base64:
            audio_name = req.speaker_audio_name
            audio_bytes = _decode_base64_blob(speaker_audio_base64, "speaker_audio_base64")
            preview_wav_bytes = _transcode_audio_bytes_to_wav(audio_bytes)
            wav, sample_rate = _decode_wav_bytes(preview_wav_bytes)
            embedding = await _compute_speaker_embedding_from_waveform(
                wav,
                sample_rate,
                expected_dim=expected_dim,
                speaker_codec=_model_speaker_codec(state.config),
            )
            entry = await _cache_speaker_reference(
                session_id=session_id,
                label=_default_speaker_label(req.label, audio_name, fallback="Speaker audio"),
                source_type="audio",
                embedding=embedding,
                original_name=audio_name,
                audio_bytes=preview_wav_bytes,
                audio_media_type="audio/wav",
            )
        else:
            embedding_name = req.speaker_embedding_name
            embedding = _compute_speaker_embedding_from_file(
                req.speaker_embedding_base64,
                expected_dim=expected_dim,
                speaker_embedding_name=embedding_name,
            )
            entry = await _cache_speaker_reference(
                session_id=session_id,
                label=_default_speaker_label(req.label, embedding_name, fallback="Speaker embedding"),
                source_type="embedding",
                embedding=embedding,
                original_name=embedding_name,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _serialize_cached_speaker(state.config, entry)


@app.get("/v2/speakers")
async def tts_list_speakers(
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """List default disk speakers and cached speakers for the current session."""
    state = get_global_state()
    expected_dim = _model_speaker_dim(state.config)
    default_entries = await _list_default_speakers(state.config)

    entries: list[CachedSpeakerReference] = []
    session_id = _normalize_session_id(x_tts_session_id)
    if session_id is not None:
        entries = await _list_cached_speakers(session_id)

    return {
        "speakers": [
            *[_serialize_default_speaker(state.config, entry, expected_dim) for entry in default_entries],
            *[_serialize_cached_speaker(state.config, entry) for entry in entries],
        ]
    }


@app.get("/v2/speakers/{speaker_id}/preview")
async def tts_preview_speaker(
    speaker_id: str,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """Return the cached reference audio for a speaker, when available."""
    state = get_global_state()
    try:
        default_entry = await _get_default_speaker(state.config, speaker_id)
        if default_entry is not None:
            audio_bytes, media_type = await _default_speaker_preview(default_entry)
            return Response(content=audio_bytes, media_type=media_type)

        session_id = _require_session_id(x_tts_session_id)
        entry = await _get_cached_speaker(session_id, speaker_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not entry.audio_bytes:
        raise HTTPException(status_code=404, detail="Cached speaker preview audio is not available.")

    return Response(
        content=entry.audio_bytes,
        media_type=entry.audio_media_type or "audio/wav",
    )


@app.get("/v2/speakers/{speaker_id}/embedding")
async def tts_speaker_embedding(
    speaker_id: str,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """The speaker embedding the server conditions on for a voice id (default or cached voice)."""
    state = get_global_state()
    try:
        embedding = await _resolve_speaker_embedding(
            config=state.config, session_id=x_tts_session_id, speaker_embedding_id=speaker_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if embedding is None:
        raise HTTPException(status_code=404, detail=f"No speaker embedding for {speaker_id!r}.")
    return {"id": speaker_id, "embedding": [round(float(x), 6) for x in embedding.reshape(-1)]}


@app.get("/v2/controls")
async def tts_controls():
    """The steering bundle's controls (--steer-bundle): sets, directions and the studio's UI metadata."""
    state = get_global_state()
    bundle = _steer_bundle(state.config)
    if bundle is None:
        raise HTTPException(status_code=404, detail="This server has no steering bundle (--steer-bundle).")
    return controls_document(
        bundle, speaker_embedding_dim=getattr(state.config.model_config, "speaker_embedding_dim", None)
    )



@app.get("/v2/capabilities")
async def tts_capabilities():
    """Return TTS feature flags for the currently loaded model."""
    from zonos3.tokenizer.textnorm import normalization_enabled

    state = get_global_state()
    return {
        "text_tokenizer_spec": _text_tokenizer_spec(state.config),
        "text_normalization_enabled": normalization_enabled(),
        "text_norm_languages": list(languages.CODES),
        "accents": _adapter_accents(state.config),
        "speaker_enabled": _model_supports_speaker(state.config),
        "speaker_conditioning_type": (
            "raw_reference_prefix"
            if _model_uses_speaker_reference_prefix(state.config)
            else "embedding"
            if _model_supports_speaker_embedding(state.config)
            else "none"
        ),
        "speaker_embedding_dim": _model_speaker_dim(state.config),
        "speaker_embedding_codec": _model_speaker_codec(state.config),
        "n_codebooks": int(getattr(state.config.model_config, "n_codebooks", 9)),
        "codebook_sizes": list(
            getattr(state.config.model_config, "codebook_sizes", ()) or ()
        ),
        "audio_codec": str(
            getattr(state.config.model_config, "audio_codec", "")
        ),
        "max_tokens": int(state.config.max_seq_len),
        "stream_decode_modes": list(_STREAM_DECODE_MODES) if _model_prefix_stream_supported(state.config) else ["whole"],
        "stream_decode_default": _default_stream_decode(state.config),
        "silence_prefill": _silence_prefill_spec(state.config),
        "tag_cfg": _tag_cfg_capabilities(state.config),
        "decode_context": {"available": True, "default_frames": DECODE_CONTEXT_FRAMES, "max_frames": 450,
                           "note": "decode the generated frames behind the last decode_context_frames of the raw-reference prompt "
                                   "(their audio dropped), so the first generated frame is not the decoder's position 0; "
                                   "no-op without a reference prompt"},
        "speaker_modes": _speaker_modes(state.config),
        "speaker_mode_default": (_speaker_modes(state.config) or [None])[0],
        "decoder": _decoder_capability(state.config),
        "condition_tokens": _condition_spec(state.config),
        "speaker_audio_upload": _model_supports_speaker(state.config),
        "speaker_cfg_supported": bool(_speaker_modes(state.config)),
        "speaker_cfg_modes": _speaker_modes(state.config),      # both paths: the twin drops the reference prefix or the speaker row
        "speaker_cfg_unconditional": "remove_reference_prefix_or_speaker_row_condition_rows_unknown_keep_text_and_generated_history",
        "speaker_cfg_formula": "uncond + scale * (cond - uncond)",
        # LACI (laci=true): languages with alignment heads (--laci-heads); needs speaker / controls CFG 1
        "laci_languages": sorted(getattr(state.config, "laci_heads", None) or {}),
        # server plugins (--plugins): plugin N serves /v2/N/* and /N
        "plugins": [p.name for p in _PLUGINS],
        "speaker_embedding_upload": _model_supports_speaker_embedding(state.config),
        "speaker_embedding_cache": _model_supports_speaker(state.config),   # cached clips for raw-reference models too
        "speaker_embedding_blend": _model_supports_speaker_embedding(state.config),
        "default_voices_enabled": bool(_default_voice_root(state.config)),
        # editing needs the checkpoint's span-end/marker rows and FuseAlign (fusealign_hub); Qwen3-ASR (.venv-asr) is
        # optional and enables transcription (edit_transcribe)
        **edit_capabilities(_EDIT_RUNTIME.model, fusealign_installed(), _ASR.installed),
        **_phoneme_capabilities(state.config),
        # server-side studio projects (/v2/projects, /v2/media; --studio-projects-dir)
        "projects": (store.capabilities() if (store := _project_store()) is not None else {"enabled": False}),
    }


def _phoneme_capabilities(config: ServerArgs) -> dict:
    """Pronunciation input: the checkpoint's phoneme tokens (if any) and the configured lexicon."""
    spec = _text_tokenizer_spec(config)
    phonemes = None if spec is None else spec.get("phonemes")
    lexicon = _get_phoneme_lexicon(config)
    return {
        # null: "(IPA)" in plain text (paren-trained checkpoints); else the U+E000 IPA U+E001 group syntax.
        "phoneme_tokens": None if phonemes is None else {
            "open_marker": phonemes["open_marker"],
            "close_marker": phonemes["close_marker"],
            "symbols": list(phonemes["symbols"]),
        },
        "phonemize_enabled": lexicon is not None,
        "phonemize_languages": [] if lexicon is None else lexicon.server_languages,
        "phonemize_guess_languages": [] if lexicon is None else lexicon.guess_languages,
    }


@app.post("/v2/phonemize")
async def tts_phonemize(request: PhonemizeRequest):
    """Gold-lexicon IPA for the words of a text, for the pronunciation input.

    With ``guess`` (and the ``phonemize`` dependency group), espeak-ng fills in
    ambiguous and unknown words of the languages in ``phonemize_guess_languages``. On a
    phoneme-token checkpoint, a pronunciation with a symbol outside its phone inventory is
    returned as ``status`` "unsupported" (no ``ipa``): the model has no token for it."""
    config = get_global_state().config
    lexicon = _get_phoneme_lexicon(config)
    if lexicon is None:
        raise HTTPException(status_code=404, detail="This server has no pronunciation lexicon.")
    try:
        return lexicon.phonemize_for(request.text, request.language, _text_tokenizer_spec(config), guess=request.guess)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _decoder_capability(config: ServerArgs) -> dict | None:
    """The audio decoder behind the matched codec (the causal iSTFT head), with its default headroom."""
    codec = str(getattr(config.model_config, "audio_codec", ""))
    from zonos3.tokenizer.matched_codec import codec_names, decoder_info
    if codec not in codec_names():
        return None
    try:
        return dict(decoder_info(codec), codec=codec)
    except Exception as exc:  # noqa: BLE001
        return dict(codec=codec, error=str(exc))


# =============================================================================
# Speech editing (zonos3.tts.edit_runtime: /v2/edit/transcribe, /v2/edit/align, /v2/edit)
# =============================================================================


def _edit_adapter(config: ServerArgs) -> str | None:
    """The speech-edit adapter (zonos3.models.accent_format.edit_adapter) edit fills run on; None without one."""
    from zonos3.models.accent_format import edit_adapter

    return edit_adapter(config.accent_adapters)


def _edit_max_spans(config: ServerArgs) -> int:
    """Edit marker rows: the extra text rows not claimed by phoneme tokens (0 without span_end_id: no editing)."""
    if getattr(config.model_config, "span_end_id", None) is None:
        return 0
    return edit_marker_rows(
        int(getattr(config.model_config, "text_extra_tokens", 0)), _text_tokenizer_spec(config)
    )


def _edit_frame_rate(config: ServerArgs) -> float:
    from zonos3_sgl.tokenizer.vocoder import TTSVocoderManager

    manager = TTSVocoderManager(
        n_codebooks=config.tts_n_codebooks,
        audio_pad_id=config.tts_audio_pad_id,
        audio_codec=config.tts_audio_codec,
    )
    return manager.sample_rate / manager.hop_length


async def _backend_reply(state: FrontendManager, make_msg: Callable[[int], BaseTokenizerMsg]):
    """Send one tokenizer message and return its finished reply."""
    uid = state.new_user()
    state.start_time_map.pop(uid, None)
    reply = None
    try:
        await state.send_one(make_msg(uid))
        async for ack in state.wait_for_ack(uid):
            reply = ack
    finally:
        state.ack_map.pop(uid, None)
        state.event_map.pop(uid, None)
    return reply


class _SglEditEngine:
    """zonos3.tts.edit_runtime.EditEngine over this server: uploads decode like the codec path, the speaker row is the
    speaker encoder's embedding, codes come from the tokenizer's codec encode / decode, and each region is one
    span-end-terminated edit generation on the scheduler (steered like /v2/generate: the engine never steers the
    prefilled prompt rows)."""

    @property
    def model(self) -> EditModel:
        config = get_global_state().config
        max_spans = _edit_max_spans(config)
        return EditModel(
            frame_rate=_edit_frame_rate(config) if max_spans > 0 else 0.0,
            n_codebooks=int(config.tts_n_codebooks),
            audio_pad_id=int(config.tts_audio_pad_id),
            codebook_sizes=tuple(int(size) for size in config.model_config.codebook_sizes),
            max_spans=max_spans,
            speaker_row=_model_supports_speaker_embedding(config),
        )

    async def decode_audio(self, audio: bytes) -> tuple[np.ndarray, int]:
        waveform, sample_rate = _decode_audio_bytes(audio)
        return waveform[0].numpy(), int(sample_rate)

    async def embed(self, audio: bytes) -> torch.Tensor:
        config = get_global_state().config
        waveform, sample_rate = _decode_audio_bytes(audio)
        try:
            return await _compute_speaker_embedding_from_waveform(
                waveform, sample_rate, expected_dim=_model_speaker_dim(config), speaker_codec=_model_speaker_codec(config)
            )
        except RuntimeError as exc:  # the speaker encoder refused the clip: the request's fault (HTTP 400)
            raise ValueError(str(exc)) from exc

    async def encode(self, audio: bytes, max_seconds: float) -> np.ndarray:
        reply = await _backend_reply(
            get_global_state(), lambda uid: TTSEncodeAudioMsg(uid=uid, audio=audio, max_seconds=max_seconds)
        )
        if reply.error is not None:
            raise ValueError(reply.error)
        return reply.codes.to(torch.int64).numpy()

    async def check(self, target_text: str, doses: list[dict[str, float]] | None, condition_buckets,
                    condition_values) -> set[str]:
        ctx = _RulesContext(get_global_state().config)
        plan.validate_phoneme_groups(ctx.text_tokenizer_spec, target_text)
        raw: set[str] = set()
        if doses is not None:
            bundle = ctx.steer_bundle
            if bundle is None:
                raise ValueError(f"This server has no steering bundle ({ctx.steer_hint}); `controls` are unavailable.")
            for region in doses:
                check_controls(bundle, region)
            raw = {did for region in doses for did in region if bundle.directions[did].space == RAW_SPACE}
        plan.resolve_condition_buckets(ctx, condition_buckets, condition_values)
        return raw

    async def fill(self, job: FillJob) -> tuple[np.ndarray, int]:
        state = get_global_state()
        ctx = _RulesContext(state.config)
        speaker_embedding, delta = job.speaker_embedding, None
        if job.controls:
            resolved = resolve_controls(
                ctx.steer_bundle, job.controls, None, text=job.controls_text,
                speaker_embedding=job.speaker_embedding, sources=ctx.steering_sources, hidden_size=ctx.hidden_size,
            )
            speaker_embedding, delta = resolved.speaker_embedding, resolved.hidden_delta
        sampling = TTSSamplingParams(
            **job.sampling, max_tokens=plan.max_tokens_budget(job.max_tokens, ctx.max_frames), seed=job.seed,
            end_token="span", min_frames=job.min_frames,
        )
        condition_ids = plan.resolve_condition_buckets(ctx, job.condition_buckets, job.condition_values)
        reply = await _backend_reply(
            state,
            lambda uid: TTSTokenizeMsg(
                uid=uid,
                text=job.text,
                sampling_params=sampling,
                language=job.language,
                text_normalization=job.text_normalization,
                edit_codes=torch.as_tensor(job.window_codes, dtype=torch.int64),
                edit_spans=job.spans,
                edit_fills=[torch.as_tensor(fill, dtype=torch.int32) for fill in job.fills],
                speaker_embedding=speaker_embedding,
                steering_delta=delta,
                condition_buckets=condition_ids,
                accent=_edit_adapter(state.config),
                accent_strength=1.0,
            ),
        )
        return np.asarray(reply.codes), -1 if reply.eos_frame is None else int(reply.eos_frame)

    async def decode(self, codes: np.ndarray) -> np.ndarray:
        reply = await _backend_reply(
            get_global_state(), lambda uid: TTSDecodeCodesMsg(uid=uid, codes=torch.as_tensor(codes, dtype=torch.int32))
        )
        return np.frombuffer(reply.audio_data, dtype=np.float32)


# The server's speech editing, over the ASR and FuseAlign workers it shares with plugins.
_EDIT_RUNTIME = EditRuntime(_SglEditEngine(), _ASR, _FUSEALIGN)
app.include_router(_edit_router(lambda: _EDIT_RUNTIME))


# ----------------------------------------------------------------------------- plugins (zonos3.server.plugins)
# --plugins: each plugin's routes over _SglHost (mounted by run_api_server). A LACI take is a /v2/generate request on
# this server's scheduler with laci=true and its duration target.
_PLUGINS: list = []
_LACI_TAKE_SLOTS: asyncio.Semaphore | None = None


def _speaking_rate_edges(config: ServerArgs) -> list[float]:
    spec = _condition_spec(config)
    return [] if spec is None else list(spec["edges"].get("speaking_rate") or [])


def _laci_take_rate_scale(config: ServerArgs, msg: TTSTokenizeMsg, take: plugins.LaciTakeRequest) -> None:
    """Put a LACI take on an extended speaking-rate scale (take.rate_edges, README "Extended speaking-rate buckets"): in
    a steered bucket the take's steering delta gains the bucket's tempo dose (on top of the take's own controls). The
    take carries the scale, its bucket on it and its delta in every bucket: LACI reports the bucket on the scale (a
    steered one's row id is only "unknown"), and a rate-adjustable take's retakes switch to another bucket's delta
    (row bucket <-> rate row unknown + tempo)."""
    if take.rate_edges is None:
        return
    bundle = _steer_bundle(config)
    if take.rate_doses and (bundle is None or bundle.rate_extension is None):
        raise ValueError("rate_doses need a steering bundle with a rate_extension")
    sources, hidden = _steering_sources(config), int(config.model_config.hidden_size)
    base = msg.steering_delta

    def delta(b: int) -> torch.Tensor | None:
        if b not in take.rate_doses:
            return base
        tempo = resolve_controls(bundle, {bundle.rate_extension.direction: take.rate_doses[b]}, None, text=msg.text,
                                 speaker_embedding=None, sources=sources, hidden_size=hidden).hidden_delta
        return tempo if base is None else base + tempo

    if take.rate_bucket is None:
        raise ValueError("a LACI take on an extended speaking-rate scale needs its rate_bucket")
    first, bucket = int(take.rate_first), int(take.rate_bucket)
    deltas = [delta(b) for b in range(first, first + len(take.rate_edges) + 1)]
    msg.steering_delta = deltas[bucket - first]
    msg.laci_rate_edges, msg.laci_rate_first, msg.laci_rate_bucket = list(take.rate_edges), first, bucket
    msg.laci_rate_deltas = deltas


class _SglHost:
    """zonos3.server.plugins.Host over this server."""

    asr = _ASR
    aligner = _FUSEALIGN

    def facts(self) -> plugins.HostFacts:
        config = get_global_state().config
        spec = _condition_spec(config)
        bundle = _steer_bundle(config)
        return plugins.HostFacts(
            model_name=Path(str(config.model_path)).name,
            rate_edges=tuple(_speaking_rate_edges(config)),
            rate_buckets=int(spec["bucket_counts"].get("speaking_rate", 0)) if spec else 0,
            laci_languages=frozenset(getattr(config, "laci_heads", None) or {}),
            speaker_modes=tuple(_speaker_modes(config)),
            raw_controls=frozenset(
                did for did, d in (bundle.directions.items() if bundle is not None else ()) if d.space == RAW_SPACE
            ),
            phoneme_tokens=_phoneme_capabilities(config)["phoneme_tokens"] is not None,
            rate_extension=bundle.rate_extension if bundle is not None else None,
        )

    async def synthesize_laci(
        self, takes: list[plugins.LaciTakeRequest], session_id: str | None
    ) -> list[plugins.TakeResult]:
        """Every take through the scheduler, at most twice the running-request cap in flight so other requests
        interleave. All takes are planned before the first is queued (HTTP 400 for a bad voice / control / text)."""
        global _LACI_TAKE_SLOTS
        state = get_global_state()
        if _LACI_TAKE_SLOTS is None:
            _LACI_TAKE_SLOTS = asyncio.Semaphore(max(1, 2 * int(getattr(state.config, "max_running_req", 1) or 1)))
        gain = _output_gain(state.config, None)
        msgs = []
        try:
            for take in takes:
                msg = await _tts_generate_msg(
                    state, take.request, session_id,
                    laci_target_frames=float(take.target_frames), laci_adjust_rate=bool(take.adjust_rate),
                )
                msgs.append(msg)
                _laci_take_rate_scale(state.config, msg, take)
        except BaseException:
            for msg in msgs:
                state.ack_map.pop(msg.uid, None)
                state.event_map.pop(msg.uid, None)
                state.start_time_map.pop(msg.uid, None)
            raise

        async def run(msg: TTSTokenizeMsg):
            async with _LACI_TAKE_SLOTS:
                await state.send_one(msg)
                data, record = await state.wait_for_tts_reply(msg.uid, gain=gain)
            if record is None:
                raise RuntimeError(f"LACI take uid={msg.uid}: the scheduler sent no LACI record")
            audio = np.frombuffer(data, dtype=np.float32).copy() if data else None
            return audio, record

        return list(await asyncio.gather(*(run(msg) for msg in msgs)))

    async def encode_reference(self, audio: np.ndarray, sample_rate: int, max_seconds: float) -> list[list[int]]:
        """Raw reference prefix codes of a mono clip (the tokenizer's codec encode)."""
        buf = io.BytesIO()
        sf.write(buf, audio, int(sample_rate), format="WAV", subtype="FLOAT")
        reply = await _backend_reply(
            get_global_state(),
            lambda uid: TTSEncodeAudioMsg(uid=uid, audio=buf.getvalue(), max_seconds=float(max_seconds)),
        )
        if reply.error is not None:
            raise ValueError(reply.error)
        return reply.codes.to(torch.int64).tolist()


def _studio_dir() -> Path:
    """--studio-dir: the production build of Zonos Voice Studio (studio/dist)."""
    state = _GLOBAL_STATE
    return Path(state.config.studio_dir if state is not None else ServerArgs.studio_dir).resolve()


# Registered last, so every API route matches first: the studio serving rule. GET|HEAD / is the studio's
# index.html (the chooser); GET|HEAD /<path> a file inside the studio dir, else the page <path>.html (/studio, /lite;
# never under /v1/ or /v2/); anything else is a 404. Pages (*.html) are no-store, so a rebuild's new hashed assets
# are picked up at once; every other file is immutable.
# No SPA fallback: the studio routes with the URL hash.
@app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
async def serve_studio(request: Request, path: str = ""):
    if request.method not in ("GET", "HEAD") or path.split("/", 1)[0] in ("v1", "v2"):
        raise HTTPException(status_code=404, detail="Not Found")
    root = _studio_dir()
    if not path:
        index = root / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="The studio is not built (pnpm --dir studio build).")
        return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-store, max-age=0"})
    for target in ((root / path).resolve(), (root / f"{path}.html").resolve()):
        if target.is_relative_to(root) and target.is_file():
            cache = "no-store, max-age=0" if target.name.endswith(".html") else "public, max-age=31536000, immutable"
            return FileResponse(target, headers={"Cache-Control": cache})
    raise HTTPException(status_code=404, detail="Not Found")


async def tts_shell():
    """Interactive TTS shell for generating and playing audio."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter

    commands = [
        "/exit",
        "/save",
        "/play",
        "/temperature",
        "/topk",
        "/repwindow",
        "/reppenalty",
        "/repcodebooks",
    ]
    completer = WordCompleter(commands)
    session = PromptSession("TTS> ", completer=completer)

    # TTS settings (the generation registry's sampling defaults)
    temperature = SAMPLING["temperature"]
    topk = SAMPLING["topk"]
    repetition_window = SAMPLING["repetition_window"]
    repetition_penalty = SAMPLING["repetition_penalty"]
    repetition_codebooks = SAMPLING["repetition_codebooks"]
    last_audio: bytes | None = None

    # Try to import sounddevice for audio playback
    try:
        import numpy as np
        import sounddevice as sd

        can_play_audio = True
    except ImportError:
        can_play_audio = False
        print("Note: Install 'sounddevice' to enable audio playback")

    print("TTS Shell - Enter text to generate speech")
    print(
        "Commands: /exit, /save <filename>, /play, /temperature <val>, "
        "/topk <val>, /repwindow <frames>, /reppenalty <val>, "
        "/repcodebooks <count>"
    )
    print()

    try:
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue

            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                elif cmd.startswith("/save"):
                    parts = cmd.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: /save <filename.wav>")
                        continue
                    filename = parts[1]
                    if last_audio is None:
                        print("No audio to save. Generate some speech first.")
                        continue
                    write_pcm16_wav(filename, np.frombuffer(last_audio, np.float32), SAMPLE_RATE)
                    print(f"Saved to {filename}")
                    continue
                elif cmd == "/play":
                    if not can_play_audio:
                        print("Audio playback not available. Install 'sounddevice'.")
                        continue
                    if last_audio is None:
                        print("No audio to play. Generate some speech first.")
                        continue
                    audio = np.frombuffer(last_audio, dtype=np.float32)
                    sd.play(audio, samplerate=SAMPLE_RATE)
                    sd.wait()
                    continue
                elif cmd.startswith("/temperature"):
                    parts = cmd.split()
                    if len(parts) < 2:
                        print(f"Current temperature: {temperature}")
                        continue
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                    continue
                elif cmd.startswith("/topk"):
                    parts = cmd.split()
                    if len(parts) < 2:
                        print(f"Current topk: {topk}")
                        continue
                    topk = int(parts[1])
                    print(f"Top-k set to {topk}")
                    continue
                elif cmd.startswith("/repwindow"):
                    parts = cmd.split()
                    if len(parts) < 2:
                        print(f"Current repetition window: {repetition_window}")
                        continue
                    repetition_window = max(int(parts[1]), 0)
                    print(f"Repetition window set to {repetition_window}")
                    continue
                elif cmd.startswith("/reppenalty"):
                    parts = cmd.split()
                    if len(parts) < 2:
                        print(f"Current repetition penalty: {repetition_penalty}")
                        continue
                    repetition_penalty = max(float(parts[1]), 1.0)
                    print(f"Repetition penalty set to {repetition_penalty}")
                    continue
                elif cmd.startswith("/repcodebooks"):
                    parts = cmd.split()
                    if len(parts) < 2:
                        print(f"Current repetition codebooks: {repetition_codebooks}")
                        continue
                    repetition_codebooks = int(parts[1])
                    print(f"Repetition codebooks set to {repetition_codebooks}")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue

            # Generate TTS
            state = get_global_state()
            uid = state.new_user()

            await state.send_one(
                TTSTokenizeMsg(
                    uid=uid,
                    text=cmd,
                    sampling_params=TTSSamplingParams(
                        temperature=temperature,
                        topk=topk,
                        repetition_window=repetition_window,
                        repetition_penalty=repetition_penalty,
                        repetition_codebooks=repetition_codebooks,
                    ),
                )
            )

            # Collect audio
            print("Generating...", end="", flush=True)
            audio_chunks = []
            async for ack in state.wait_for_ack(uid):
                if isinstance(ack, TTSAudioReply):
                    if ack.audio_data:
                        audio_chunks.append(ack.audio_data)
                        print(".", end="", flush=True)
                    if ack.finished:
                        break

            if audio_chunks:
                last_audio = b"".join(audio_chunks)
                duration = len(last_audio) / (SAMPLE_RATE * 4)  # float32 = 4 bytes
                print(f" Done! ({duration:.2f}s)")

                # Auto-play if sounddevice is available
                if can_play_audio:
                    audio = np.frombuffer(last_audio, dtype=np.float32)
                    sd.play(audio, samplerate=SAMPLE_RATE)
                    sd.wait()
            else:
                print(" No audio generated")

    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting TTS shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # Plugin routes go ahead of the studio catch-all; their keepalive paths join _KEEPALIVE_PATHS before serving.
    _PLUGINS[:] = plugins.load(config.plugins)
    plugins.mount(app, _PLUGINS, _SglHost(), _KEEPALIVE_PATHS)

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        # Request logs would record the access_token query of a private server.
        uvicorn.run(app, host=host, port=port, access_log=not os.environ.get("ZONOS3_ACCESS_TOKEN"))
    else:
        asyncio.run(tts_shell())

# CONDITION_ROWS_RUNTIME_20261016
# EMBEDDING_CFG_20261016

# CONDITION_FLAG_RUNTIME_20261017

# DECODE_CONTEXT_RUNTIME_20261017

# SILENCE_DEFAULT_RUNTIME_20261017
