from __future__ import annotations

import asyncio
import functools
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Tuple

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask
from zonos3 import languages
from zonos3.generation import DECODE_CONTEXT_FRAMES, SAMPLE_RATE, SAMPLING
from zonos3.reference.synth import decode_base64_blob, load_embedding_vector, slerp
from zonos3.server import plugins
from zonos3.server.capabilities import (
    CapabilityFacts,
    capabilities_document,
    condition_spec,
    decoder_capability,
    dual_codec_hop,
    models_document,
)
from zonos3.server.edit import make_router as _edit_router
from zonos3.server.host import host_facts, laci_slots, rate_scale
from zonos3.server.keepalive import KEEPALIVE_PATHS, KeepaliveMiddleware
from zonos3.server.middleware import access_log_enabled, install_cors, install_private_access
from zonos3.server.projects import ProjectStore
from zonos3.server.projects import make_router as _projects_router
from zonos3.server.schemas import (
    SPEAKER_INPUT_FIELDS,
    GenerateRequest,
    PhonemizeRequest,
    SpeakerCreateRequest,
    SpeechRequest,
)
from zonos3.server.speech import check_format, prefix_stream_supported, resolve_stream_decode, speech_headers
from zonos3.server.studio import install_studio
from zonos3.server.voices import (
    DefaultVoice,
    new_speaker_id,
    normalize_session_id,
    read_voice_file,
    require_session_id,
    scan_default_voices,
    serialize_default_voice,
    serialize_session_voice,
    speaker_label,
    voice_preview,
    voice_speaker_modes,
    warm_voice_embeddings,
)
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
from zonos3.tts.edit_runtime import EditModel, EditRuntime, FillJob
from zonos3.tts.markup import trained_on_tags
from zonos3.tts.steering import (
    SteeringBundle,
    check_controls,
    controls_document,
    disabled_controls,
    load_bundle,
    no_steering_message,
    resolve_controls,
)

from zonos3_sgl.message import (
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchTTSFrontendMsg,
    TTSAudioReply,
    TTSCodesReply,
    TTSDecodeCodesMsg,
    TTSEncodeAudioMsg,
    TTSNormalizeMsg,
    TTSSamplingParams,
    TTSTokenizeMsg,
)
from zonos3_sgl.utils.logger import init_logger
from zonos3_sgl.utils.mp import ZmqAsyncPullQueue, ZmqAsyncPushQueue

from .args import ServerArgs

if TYPE_CHECKING:
    from zonos3.tts.phonemes import PhonemeLexicon


logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None
_SPEAKER_EMBEDDERS = {}
_SPEAKER_EMBEDDER_LOCK = asyncio.Lock()
# Reference clips are decoded (ffmpeg transcode) on this many worker threads, off the event loop. The encoder forward
# itself is serialized: concurrent forwards from several threads hit "illegal memory access" in the CUDA front end.
_SPEAKER_EMBEDDER_THREADS = max(1, int(os.getenv("ZONOS3_SPEAKER_EMBEDDER_THREADS", "8")))
_SPEAKER_EMBEDDER_POOL: ThreadPoolExecutor | None = None
_SPEAKER_EMBEDDER_FORWARD_LOCK = threading.Lock()
_SESSION_SPEAKER_CACHE: Dict[str, Dict[str, "CachedSpeakerReference"]] = {}
_SESSION_SPEAKER_CACHE_LOCK = asyncio.Lock()
_DEFAULT_SPEAKER_CACHE: Dict[Tuple[str, int], Dict[str, DefaultVoice]] = {}
_DEFAULT_SPEAKER_CACHE_LOCK = asyncio.Lock()
# The server's ASR worker (zonos3/workers/asr.py: Qwen3-ASR in .venv-asr; ZONOS3_ASR_PYTHON / ZONOS3_ASR_DEVICE):
# every speech recognition of speech editing and plugins. It starts on the first call.
_ASR = AsrWorker()
# The server's FuseAlign worker (zonos3/workers/fusealign.py in this interpreter; ZONOS3_FUSEALIGN_DEVICE picks its
# device): every forced alignment of speech editing and plugins. It starts on the first call.
_FUSEALIGN = FuseAlignWorker()
# Gold pronunciation lexicons, keyed by LMDB directory.
_PHONEME_LEXICON_CACHE: Dict[str, "PhonemeLexicon"] = {}


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
        _PROJECT_STORE = ProjectStore(
            config.studio_projects_dir, config.studio_max_project_bytes, config.studio_max_media_bytes
        )
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
    return bool(getattr(config.model_config, "speaker_reference_prefix_enabled", False))


def _speaker_modes(config: ServerArgs) -> list[str]:
    modes = []
    if _model_uses_speaker_reference_prefix(config):
        modes.append("reference")
    if _model_supports_speaker_embedding(config):
        modes.append("embedding")
    return modes


# ----------------------------------------------------------------------------- condition rows (cond3 finetunes)
def _condition_spec(config: ServerArgs) -> dict | None:
    """The checkpoint's condition rows (zonos3.server.capabilities.condition_spec); None when the model has none."""
    model_config = config.model_config
    if not getattr(model_config, "condition_tokens_enabled", False):
        return None
    names = [
        str(item) for item in (getattr(model_config, "condition_names", None) or ("loudness", "fmax", "speaking_rate"))
    ]
    return condition_spec(
        names=names,
        count=int(getattr(model_config, "condition_count", len(names)) or len(names)),
        bucket_counts=model_config.condition_bucket_counts,
        edges=getattr(model_config, "condition_edges", None),
        flags=getattr(model_config, "condition_flags", None),
        units=getattr(model_config, "condition_units", None),
    )


def _audio_codec(config: ServerArgs) -> str:
    return str(getattr(config.model_config, "audio_codec", "") or "")


def _model_speaker_dim(config: ServerArgs) -> int:
    return int(getattr(config.model_config, "speaker_embedding_dim", 128))


def _model_speaker_codec(config: ServerArgs) -> str | None:
    codec = getattr(config.model_config, "speaker_embedding_codec", None)
    return str(codec) if codec else None


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
        logger.info(
            "Loaded steering bundle %s: %d directions in sets %s.", path, len(bundle.directions), ", ".join(bundle.sets)
        )
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
        return _condition_spec(self.config) or {
            "names": [],
            "bucket_counts": {},
            "unknown_ids": {},
            "edges": {},
            "flags": {},
        }

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
        return dual_codec_hop(_audio_codec(self.config)) is not None


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


def _apply_fade_out_pcm(audio_bytes: bytes, fade_out_ms: float, sample_rate: int = SAMPLE_RATE) -> bytes:
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


def _speaker_document(config: ServerArgs, entry: CachedSpeakerReference | DefaultVoice) -> dict:
    """A voice of GET /v2/speakers (zonos3.server.voices) with the checkpoint's speaker modes and dimension."""
    modes, disabled = _speaker_modes(config), disabled_controls(_steer_bundle(config), entry.speaker_id)
    if isinstance(entry, DefaultVoice):
        return serialize_default_voice(
            entry,
            modes=voice_speaker_modes(modes, entry.source_type),
            embedding_dim=_model_speaker_dim(config),
            disabled=disabled,
        )
    dim = int(entry.embedding.numel())
    return serialize_session_voice(
        speaker_id=entry.speaker_id,
        label=entry.label,
        source_type=entry.source_type,
        original_name=entry.original_name,
        created_at=entry.created_at,
        has_preview=bool(entry.audio_bytes),
        embedding_dim=dim,
        modes=voice_speaker_modes(modes, entry.source_type, has_embedding=dim > 0),
        disabled=disabled,
    )


def _default_voice_root(config: ServerArgs) -> Path | None:
    raw = getattr(config, "tts_default_voices_dir", None)
    if not raw:
        return None
    return Path(str(raw)).expanduser()


async def _list_default_speakers(config: ServerArgs) -> list[DefaultVoice]:
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
        current = scan_default_voices(root, _DEFAULT_SPEAKER_CACHE.get(cache_key, {}))
        _DEFAULT_SPEAKER_CACHE[cache_key] = current
        return sorted(current.values(), key=lambda entry: entry.label.lower())


async def _get_default_speaker(
    config: ServerArgs,
    speaker_id: str,
) -> DefaultVoice | None:
    speakers = await _list_default_speakers(config)
    return next((entry for entry in speakers if entry.speaker_id == speaker_id), None)


async def _load_default_speaker_embedding(
    config: ServerArgs,
    entry: DefaultVoice,
) -> torch.Tensor:
    expected_dim = _model_speaker_dim(config)
    if entry.embedding is None:
        if entry.source_type == "embedding":
            entry.embedding = load_embedding_vector(
                read_voice_file(entry),
                expected_dim=expected_dim,
                file_name=entry.original_name,
            )
        else:
            wav_bytes, _ = voice_preview(entry)
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
        speaker_id=new_speaker_id(),
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
            from zonos3.models.speaker_cloning import create_speaker_embedder, speaker_embedder_device

            embedder = create_speaker_embedder(
                speaker_codec,
                expected_dim,
                device=speaker_embedder_device(),
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


def _speaker_embedder_pool() -> ThreadPoolExecutor:
    """Worker threads for reference-clip decoding and speaker embedding, so concurrent uploads are decoded in parallel
    instead of one after another on the event loop."""
    global _SPEAKER_EMBEDDER_POOL
    if _SPEAKER_EMBEDDER_POOL is None:
        _SPEAKER_EMBEDDER_POOL = ThreadPoolExecutor(
            max_workers=_SPEAKER_EMBEDDER_THREADS, thread_name_prefix="speaker-embedder"
        )
    return _SPEAKER_EMBEDDER_POOL


def _embed_waveform_sync(embedder, wav: torch.Tensor, sample_rate: int) -> list[torch.Tensor]:
    """One encoder forward (one at a time across worker threads); returns the CPU candidates."""
    with _SPEAKER_EMBEDDER_FORWARD_LOCK, torch.inference_mode():
        output = embedder(wav, sample_rate)
        outputs = output if isinstance(output, tuple) else (output,)
        return [tensor.squeeze(0).to(dtype=torch.float32, device="cpu") for tensor in outputs]


async def _compute_speaker_embedding_from_audio(
    speaker_audio_base64: str,
    expected_dim: int,
    speaker_codec: str | None = None,
) -> torch.Tensor:
    audio_bytes = decode_base64_blob(speaker_audio_base64, "speaker_audio_base64")
    wav, sample_rate = await asyncio.get_running_loop().run_in_executor(
        _speaker_embedder_pool(), _decode_audio_bytes, audio_bytes
    )
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
    candidates = await asyncio.get_running_loop().run_in_executor(
        _speaker_embedder_pool(), _embed_waveform_sync, embedder, wav, sample_rate
    )

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
    embedding_bytes = decode_base64_blob(
        speaker_embedding_base64,
        "speaker_embedding_base64",
    )
    return load_embedding_vector(
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

    resolved_session_id = require_session_id(session_id)
    entry = await _get_cached_speaker(resolved_session_id, speaker_id)
    expected_dim = _model_speaker_dim(config)
    actual_dim = int(entry.embedding.numel())
    if actual_dim == 0:
        raise ValueError(
            f"Cached speaker '{entry.label}' holds a reference clip only; it has no embedding for speaker_mode=embedding."
        )
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
    direct_source_count = sum(bool(value) for value in (speaker_audio_base64, speaker_embedding_base64))
    if direct_source_count > 1:
        raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")

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
        return slerp(
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
        logger.info("Speaker reference provided, but current model does not support speaker embeddings; ignoring.")
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
            audio_bytes, _ = voice_preview(default_entry)  # transcoded WAV, cached on the entry
            return audio_bytes
        entry = await _get_cached_speaker(require_session_id(session_id), speaker_embedding_id)
        if not entry.audio_bytes:
            raise ValueError(f"Cached speaker '{entry.label}' has no reference audio.")
        return entry.audio_bytes
    if speaker_audio_base64 is None:
        return None
    return decode_base64_blob(speaker_audio_base64, "speaker_audio_base64")


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
                            "uid=%d TTS TTFB=%.3fms",
                            uid,
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
                uid,
                e2e_s * 1000,
                audio_s,
                chunk_count,
                audio_s / e2e_s if e2e_s > 0 else 0,
            )

    async def wait_for_tts_audio(self, uid: int, fade_out_ms: float = 0.0, gain: float = 1.0):
        """Collect all TTS audio data and return as single bytes object."""
        return (await self.wait_for_tts_reply(uid, fade_out_ms, gain))[0]

    async def wait_for_tts_reply(
        self, uid: int, fade_out_ms: float = 0.0, gain: float = 1.0
    ) -> tuple[bytes, dict | None]:
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
                uid,
                e2e_s * 1000,
                audio_s,
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
    await warm_voice_embeddings(
        await _list_default_speakers(config), lambda entry: _load_default_speaker_embedding(config, entry), logger
    )


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


# The spec of this API is servers/api/openapi.yaml; FastAPI's generated docs would describe a second one.
app = FastAPI(
    title="Zonos3 API Server", version="2.0.0", lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None
)

# Innermost, so CORS and the access check wrap its early status line (zonos3.server.keepalive); plugins.mount adds
# each plugin's paths before serving.
_KEEPALIVE = list(KEEPALIVE_PATHS)
app.add_middleware(KeepaliveMiddleware, paths=_KEEPALIVE)
install_cors(app)
install_private_access(app)

# Studio projects on the server (shared by every studio browser, kept across restarts): /v2/projects, /v2/media.
app.include_router(_projects_router(_project_store), prefix="/v2")


@app.get("/v2/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def available_models():
    return models_document(get_global_state().config.model_path)


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
        if any(fields[name] is not None for name in SPEAKER_INPUT_FIELDS):
            raise HTTPException(status_code=400, detail="voice cannot be combined with speaker_* fields")
        session_id = normalize_session_id(x_tts_session_id)
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
    # The tokenizer normalizes the text (the longest step before the scheduler, ~0.2 ms per English character) while
    # this coroutine resolves the request; the TTSTokenizeMsg then finds it memoized. An invalid language is
    # rejected below.
    if req.text_normalization:
        try:
            language = languages.normalize(req.language)
        except ValueError:
            language = None
        if language is not None:
            await state.send_one(TTSNormalizeMsg(text=req.text, language=language))
    ctx = _RulesContext(state.config)
    try:
        request = plan.parse_request(req.model_dump())
        head = plan.request_head(ctx, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        speaker_reference_audio, speaker_embedding = await _resolve_request_speaker_conditioning(
            state.config, req, session_id, head.mode
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
    try:
        stream_decode = resolve_stream_decode(
            req.stream_decode,
            streaming=bool(req.stream),
            prefix_supported=prefix_stream_supported(_audio_codec(state.config)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # sgl's own LACI rule: a retake in another speaking-rate bucket changes the condition rows, which only the naive
    # cache never serves stale from a cached prompt prefix
    if laci_target_frames is not None and laci_adjust_rate and getattr(state.config, "cache_type", "naive") != "naive":
        raise HTTPException(
            status_code=400, detail="A LACI duration target that adjusts the speaking rate needs --cache-type naive"
        )
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
            stream_codebooks=req.stream_codebooks,
            stream_codebooks_scope=req.stream_codebooks_scope,
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
    try:
        check_format(bool(req.stream), req.format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    msg = await _tts_generate_msg(state, req, session_id)
    uid = msg.uid
    await state.send_one(msg)

    async def _abort():
        await state.abort_user(uid)

    headers = speech_headers(
        fmt=req.format,
        seed=msg.sampling_params.seed,
        control_spans=bool(req.control_spans),
        follows_reading=msg.steer_follow_chars is not None,
    )
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
        return Response(
            content=pcm16_wav_bytes(np.frombuffer(audio_data, np.float32), SAMPLE_RATE),
            media_type="audio/wav",
            headers=headers,
        )
    return Response(content=audio_data, media_type="audio/pcm", headers=headers)


@app.post("/v2/generate")
async def tts_generate(
    req: GenerateRequest,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """Synthesize speech (servers/api/openapi.yaml GenerateRequest)."""
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
        session_id = require_session_id(x_tts_session_id)
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
                    label=speaker_label(req.label, embedding_name, fallback="Speaker embedding"),
                    source_type="embedding",
                    embedding=_compute_speaker_embedding_from_file(
                        req.speaker_embedding_base64,
                        expected_dim=_model_speaker_dim(state.config),
                        speaker_embedding_name=embedding_name,
                    ),
                    original_name=embedding_name,
                )
                return _speaker_document(state.config, entry)
            if not raw_audio_base64:
                raise ValueError("Provide an audio file to cache as the speaker reference.")
            raw_name = req.speaker_audio_name
            raw_bytes = decode_base64_blob(raw_audio_base64, "speaker_audio_base64")
            raw_wav_bytes = _transcode_audio_bytes_to_wav(raw_bytes)
            raw_embedding = torch.empty(0)
            if _model_supports_speaker_embedding(state.config):  # both paths: keep the clip and its embedding
                raw_wav, raw_rate = _decode_wav_bytes(raw_wav_bytes)
                raw_embedding = await _compute_speaker_embedding_from_waveform(
                    raw_wav,
                    raw_rate,
                    expected_dim=_model_speaker_dim(state.config),
                    speaker_codec=_model_speaker_codec(state.config),
                )
            entry = await _cache_speaker_reference(
                session_id=session_id,
                label=speaker_label(req.label, raw_name, fallback="Speaker audio"),
                source_type="audio",
                embedding=raw_embedding,
                original_name=raw_name,
                audio_bytes=raw_wav_bytes,
                audio_media_type="audio/wav",
            )
            return _speaker_document(state.config, entry)
        if not _model_supports_speaker_embedding(state.config):
            raise ValueError("Current model does not support speaker embeddings.")

        speaker_audio_base64 = req.speaker_audio_base64
        direct_source_count = sum(bool(value) for value in (speaker_audio_base64, req.speaker_embedding_base64))
        if direct_source_count == 0:
            raise ValueError("Provide an audio file or saved embedding file to cache.")
        if direct_source_count > 1:
            raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")

        expected_dim = _model_speaker_dim(state.config)
        if speaker_audio_base64:
            audio_name = req.speaker_audio_name
            audio_bytes = decode_base64_blob(speaker_audio_base64, "speaker_audio_base64")
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
                label=speaker_label(req.label, audio_name, fallback="Speaker audio"),
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
                label=speaker_label(req.label, embedding_name, fallback="Speaker embedding"),
                source_type="embedding",
                embedding=embedding,
                original_name=embedding_name,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _speaker_document(state.config, entry)


@app.get("/v2/speakers")
async def tts_list_speakers(
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    """List default disk speakers and cached speakers for the current session."""
    state = get_global_state()
    default_entries = await _list_default_speakers(state.config)

    entries: list[CachedSpeakerReference] = []
    session_id = normalize_session_id(x_tts_session_id)
    if session_id is not None:
        entries = await _list_cached_speakers(session_id)

    return {"speakers": [_speaker_document(state.config, entry) for entry in (*default_entries, *entries)]}


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
            audio_bytes, media_type = voice_preview(default_entry)
            return Response(content=audio_bytes, media_type=media_type)

        session_id = require_session_id(x_tts_session_id)
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
        raise HTTPException(status_code=404, detail=no_steering_message("--steer-bundle"))
    return controls_document(
        bundle, speaker_embedding_dim=getattr(state.config.model_config, "speaker_embedding_dim", None)
    )


def _capability_facts(config: ServerArgs) -> CapabilityFacts:
    """This server's facts for GET /v2/capabilities (zonos3.server.capabilities)."""
    from zonos3.tokenizer.textnorm import normalization_enabled

    modes, codec = _speaker_modes(config), _audio_codec(config)
    store = _project_store()
    return CapabilityFacts(
        text_tokenizer_spec=_text_tokenizer_spec(config),
        text_normalization_enabled=normalization_enabled(),
        accents=_adapter_accents(config),
        speaker_modes=modes,
        speaker_embedding_dim=_model_speaker_dim(config),
        speaker_embedding_codec=_model_speaker_codec(config),
        n_codebooks=int(getattr(config.model_config, "n_codebooks", 9)),
        codebook_sizes=list(getattr(config.model_config, "codebook_sizes", ()) or ()),
        audio_codec=codec,
        max_tokens=int(config.max_seq_len),
        prefix_stream=prefix_stream_supported(codec),
        silence_hop=dual_codec_hop(codec),
        tag_cfg_available=True,
        tag_cfg_trained=_model_trained_on_tags(str(config.model_path)),
        decoder=decoder_capability(codec),
        condition_tokens=_condition_spec(config),
        speaker_audio_upload=bool(modes),
        speaker_cfg_available=bool(modes),
        laci_languages=getattr(config, "laci_heads", None) or {},
        plugins=[p.name for p in _PLUGINS],
        default_voices_enabled=bool(_default_voice_root(config)),
        edit_model=_EDIT_RUNTIME.model,
        edit_available=fusealign_installed(),
        edit_transcribe=_ASR.installed,
        lexicon=_get_phoneme_lexicon(config),
        projects=store.capabilities() if store is not None else {"enabled": False},
    )


@app.get("/v2/capabilities")
async def tts_capabilities():
    """Return TTS feature flags for the currently loaded model."""
    return capabilities_document(_capability_facts(get_global_state().config))


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
        return await asyncio.to_thread(
            lexicon.phonemize_for, request.text, request.language, _text_tokenizer_spec(config), guess=request.guess
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    return edit_marker_rows(int(getattr(config.model_config, "text_extra_tokens", 0)), _text_tokenizer_spec(config))


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
        from zonos3.tokenizer.matched_codec import entry

        config = get_global_state().config
        return EditModel.from_config(
            config.model_config, hop=int(entry(str(config.tts_audio_codec))["hop"]), max_spans=_edit_max_spans(config)
        )

    async def decode_audio(self, audio: bytes) -> tuple[np.ndarray, int]:
        waveform, sample_rate = _decode_audio_bytes(audio)
        return waveform[0].numpy(), int(sample_rate)

    async def embed(self, audio: bytes) -> torch.Tensor:
        config = get_global_state().config
        waveform, sample_rate = _decode_audio_bytes(audio)
        try:
            return await _compute_speaker_embedding_from_waveform(
                waveform,
                sample_rate,
                expected_dim=_model_speaker_dim(config),
                speaker_codec=_model_speaker_codec(config),
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

    async def check(
        self, target_text: str, doses: list[dict[str, float]] | None, condition_buckets, condition_values
    ) -> set[str]:
        ctx = _RulesContext(get_global_state().config)
        plan.validate_phoneme_groups(ctx.text_tokenizer_spec, target_text)
        raw: set[str] = set()
        if doses is not None:
            bundle = ctx.steer_bundle
            if bundle is None:
                raise ValueError(no_steering_message(ctx.steer_hint, field="controls"))
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
                ctx.steer_bundle,
                job.controls,
                None,
                text=job.controls_text,
                speaker_embedding=job.speaker_embedding,
                sources=ctx.steering_sources,
                hidden_size=ctx.hidden_size,
            )
            speaker_embedding, delta = resolved.speaker_embedding, resolved.hidden_delta
        sampling = TTSSamplingParams(
            **job.sampling,
            max_tokens=plan.max_tokens_budget(job.max_tokens, ctx.max_frames),
            seed=job.seed,
            end_token="span",
            min_frames=job.min_frames,
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


class _SglHost:
    """zonos3.server.plugins.Host over this server."""

    asr = _ASR
    aligner = _FUSEALIGN

    def facts(self) -> plugins.HostFacts:
        config = get_global_state().config
        return host_facts(
            model_path=config.model_path,
            condition_tokens=_condition_spec(config),
            speaker_modes=_speaker_modes(config),
            bundle=_steer_bundle(config),
            laci_languages=getattr(config, "laci_heads", None) or {},
            text_tokenizer_spec=_text_tokenizer_spec(config),
        )

    async def synthesize_laci(
        self, takes: list[plugins.LaciTakeRequest], session_id: str | None
    ) -> list[plugins.TakeResult]:
        """Every take through the scheduler, at most twice the running-request cap in flight so other requests
        interleave. All takes are planned before the first is queued (HTTP 400 for a bad voice / control / text)."""
        global _LACI_TAKE_SLOTS
        state = get_global_state()
        if _LACI_TAKE_SLOTS is None:
            _LACI_TAKE_SLOTS = laci_slots(getattr(state.config, "max_running_req", 1))
        gain = _output_gain(state.config, None)
        msgs = []
        try:
            for take in takes:
                msg = await _tts_generate_msg(
                    state,
                    take.request,
                    session_id,
                    laci_target_frames=float(take.target_frames),
                    laci_adjust_rate=bool(take.adjust_rate),
                )
                msgs.append(msg)
                scale = rate_scale(
                    take,
                    _steer_bundle(state.config),
                    text=msg.text,
                    base_delta=msg.steering_delta,
                    sources=_steering_sources(state.config),
                    hidden_size=int(state.config.model_config.hidden_size),
                )
                if scale is not None:
                    msg.steering_delta = scale.delta
                    msg.laci_rate_edges, msg.laci_rate_first, msg.laci_rate_bucket = (
                        list(scale.edges),
                        scale.first,
                        scale.bucket,
                    )
                    msg.laci_rate_deltas = scale.deltas
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


# Registered last, so every API route matches first (zonos3.server.studio).
install_studio(app, lambda: _studio_dir())


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

    # Plugin routes go ahead of the studio catch-all; their keepalive paths join _KEEPALIVE before serving.
    _PLUGINS[:] = plugins.load(config.plugins)
    plugins.mount(app, _PLUGINS, _SglHost(), _KEEPALIVE)

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        # Request logs would record the access_token query of a private server (access_log_enabled).
        uvicorn.run(app, host=host, port=port, access_log=access_log_enabled())
    else:
        asyncio.run(tts_shell())
