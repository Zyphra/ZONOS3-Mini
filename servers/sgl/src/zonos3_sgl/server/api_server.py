from __future__ import annotations

import asyncio
import base64
import bisect
import functools
import math
import hashlib
import io
import os
import random
import re
import secrets
import subprocess
import tempfile
import time
import uuid
import wave
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
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
from zonos3.tts.audio import (
    _decode_audio_bytes,
    _decode_wav_bytes,
    _transcode_audio_bytes_to_wav,
)
from zonos3.tts.conditioning import _model_tts_max_tokens, _resolve_tts_max_tokens
from zonos3.tts.edit import edit_marker_rows, extract_fill, splice_edit, validate_edit_spans
from zonos3.tts import textmatch
from zonos3.tts.edit_align import (
    GROUP_OPEN,
    AsrWorker,
    FuseAlignWorker,
    WorkerError,
    canonical_markup,
    diff_spans,
    edit_aligned_words,
    edit_language,
    edit_words,
    expected_fill_seconds,
    fill_groups,
    fusealign_installed,
    plan_edit_windows,
    seconds_to_spans,
    target_tokens,
    transcribe_audio,
    whole_clip_window,
    wildcard_verify,
)
from zonos3.tts.emotion import (
    HIDDEN_SPACES,
    EmotionCalibration,
    EmotionDirections,
    apply_emotion,
    emotion_hidden_delta,
)
from zonos3.tts.steering import (
    RAW_SPACE,
    SteeringBundle,
    check_controls,
    controls_document,
    load_bundle,
    resolve_controls,
)
from zonos3.tts.markup import NONVERBAL_MARKUP_TAGS, has_nonverbal_markup, strip_nonverbal_markup, trained_on_tags
from zonos3.server.keepalive import KeepaliveMiddleware
from zonos3.server.projects import ProjectStore, make_router as _projects_router
from zonos3.server.schemas import (
    DubAnalyzeRequest,
    DubGenerateRequest,
    DubRegenerateRequest,
    DubTranscribeRequest,
    EditAlignRequest,
    EditRequest,
    EditTranscribeRequest,
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
# Loaded emotion directions, keyed by directory. Value is None when no directions
# are available so we only attempt to load each directory once.
_EMOTION_DIRECTIONS_CACHE: Dict[str, "EmotionDirections | None"] = {}
# Loaded per-speaker emotion strength calibration, keyed by directory.
_EMOTION_CALIBRATION_CACHE: Dict[str, "EmotionCalibration | None"] = {}
# Speech editing: the longest clip transcribed (Qwen3-ASR) and aligned (FuseAlign) per request, and the clip limit.
_EDIT_ALIGN_MAX_SECONDS = 150.0
# A clip longer than max_context_seconds (at most 20 s) is edited on windows planned over its aligned
# word timings, so the alignment limit bounds every edit.
_EDIT_MAX_SECONDS = _EDIT_ALIGN_MAX_SECONDS
# The server's ASR worker (zonos3/workers/asr.py: Qwen3-ASR in .venv-asr; ZONOS3_ASR_PYTHON / ZONOS3_ASR_DEVICE):
# every speech recognition of speech editing and dubbing. It starts on the first call.
_ASR = AsrWorker()
# The server's FuseAlign worker (zonos3/workers/fusealign.py in this interpreter; ZONOS3_FUSEALIGN_DEVICE picks its
# device): every forced alignment of speech editing and dubbing. It starts on the first call.
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


def _resolve_speaker_mode(config: ServerArgs, value: str | None) -> str | None:
    """Which conditioning path a request uses: raw reference prefix (default when the model has it) or the reserved
    speaker-embedding row; explicit requests (the schema's enum) must name a path the checkpoint supports."""
    modes = _speaker_modes(config)
    if value is None:
        return modes[0] if modes else None
    if value not in modes:
        raise HTTPException(status_code=400, detail=f"speaker_mode={value!r} is not available for this checkpoint (supported: {modes})")
    return value


# ----------------------------------------------------------------------------- condition rows (cond3 finetunes)
_CONDITION_LAYOUT = "[reference prefix | speaker row][loudness][fmax][speaking_rate] BOS text EOS"


def _condition_spec(config: ServerArgs) -> dict | None:
    """The checkpoint's condition rows (loudness / fmax / speaking-rate buckets): names, bucket count, edges; None
    when the model has none."""
    model_config = config.model_config
    if not getattr(model_config, "condition_tokens_enabled", False):
        return None
    names = [str(item) for item in (getattr(model_config, "condition_names", None) or ("loudness", "fmax", "speaking_rate"))]
    count = int(getattr(model_config, "condition_count", len(names)) or len(names))
    n_buckets = int(getattr(model_config, "condition_n_buckets", 10) or 10)
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
        n_buckets=n_buckets,
        unknown_index=n_buckets,
        edges=edges,
        units=dict(getattr(model_config, "condition_units", None) or {}),
        layout="[reference prefix | speaker row]" + "".join(f"[{name}]" for name in names[:count]) + " BOS text EOS",
    )


def _silence_prefill_spec(config: ServerArgs) -> dict:
    """Whether codec-matched silence frames can be prefilled into the audio stream (2026-10-17)."""
    codec = str(getattr(config.model_config, "audio_codec", "") or "")
    try:
        from zonos3.tokenizer.matched_codec import codec_names, entry
        if codec in codec_names() and entry(codec).get("family") == "dual":
            hop = int(entry(codec)["hop"])
            try:
                default_frames = max(0, int(os.environ.get("ZONOS3_SILENCE_PREFILL_FRAMES", "3")))
            except ValueError:
                default_frames = 3
            default_on = os.environ.get("ZONOS3_SILENCE_PREFILL_DEFAULT", "1").strip().lower() not in ("0", "false", "no", "off") and default_frames > 0
            return dict(available=True, default_frames=default_frames, default_on=default_on, default_seconds=round(default_frames * hop / 44100.0, 3),
                        frame_seconds=hop / 44100.0, codec=codec,
                        note="codec-matched silence frames open the audio stream before the model starts (their audio is not emitted); "
                             "a request without silence_prefill_frames gets the server default, silence_prefill_frames=0 "
                             "turns it off and silence_prefill_frames=N sets the length")
    except Exception:  # noqa: BLE001
        pass
    return dict(available=False, default_frames=None, codec=codec)


def _condition_bucket_of(edges: list[float], value: float) -> int:
    # bucket i = [edge_{i-1}, edge_i), exactly the training loader's bisect_right over the edges
    return int(bisect.bisect_right(edges, float(value)))


def _resolve_condition_buckets(config: ServerArgs, buckets, values) -> list[int] | None:
    """Per-request bucket ids for the condition rows: explicit ids win, raw values map through the edges, anything
    else is unknown. None for checkpoints without condition rows (a request that names them is refused)."""
    spec = _condition_spec(config)
    if spec is None:
        if buckets not in (None, {}, []) or values not in (None, {}, []):
            raise HTTPException(status_code=400, detail="This checkpoint has no condition rows (loudness / fmax / speaking rate)")
        return None
    names, n_buckets = spec["names"], spec["n_buckets"]

    def as_map(raw, what):
        if raw is None:
            return {}
        if isinstance(raw, Mapping):
            unknown = [str(k) for k in raw if str(k) not in names]
            if unknown:
                raise HTTPException(status_code=400, detail=f"{what}: unknown condition name(s) {unknown}; expected {names}")
            return {str(k): v for k, v in raw.items()}
        raw = list(raw)
        if len(raw) != len(names):
            raise HTTPException(status_code=400, detail=f"{what}: expected {len(names)} entries in the order {names}")
        return dict(zip(names, raw))

    ids_by_name = as_map(buckets, "condition_buckets")
    values_by_name = as_map(values, "condition_values")
    out = []
    for name in names:
        if ids_by_name.get(name) is not None:
            try:
                bucket = int(ids_by_name[name])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"condition_buckets[{name}] must be an integer") from None
            flag = spec["flags"].get(name)
            if flag is not None and bucket != n_buckets and bucket not in [int(v) for v in (flag.get("values") or (0, 1))]:
                raise HTTPException(status_code=400, detail=f"condition_buckets[{name}] is a flag: one of {flag.get('values') or [0, 1]} or {n_buckets} (= unknown)")
            if not 0 <= bucket <= n_buckets:
                raise HTTPException(status_code=400, detail=f"condition_buckets[{name}] must be in [0, {n_buckets - 1}] or {n_buckets} (= unknown)")
        elif values_by_name.get(name) is not None:
            try:
                value = float(values_by_name[name])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"condition_values[{name}] must be a number") from None
            if not math.isfinite(value):
                raise HTTPException(status_code=400, detail=f"condition_values[{name}] must be finite")
            if name in spec["flags"]:
                bucket = 1 if value >= 0.5 else 0          # a flag row: any truthy value means "yes"
            else:
                edges = spec["edges"].get(name)
                if not edges:
                    raise HTTPException(status_code=400, detail=f"the checkpoint carries no edges for {name}; send condition_buckets[{name}] instead")
                bucket = _condition_bucket_of(edges, value)
        else:
            bucket = n_buckets
        out.append(bucket)
    return out


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


def _normalize_tts_request_language(language: str) -> str:
    from zonos3.tokenizer.textnorm import SERVER_TO_NEMO_LANG

    normalized = str(language or "").strip().lower().replace("-", "_")
    if normalized not in SERVER_TO_NEMO_LANG:
        supported = ", ".join(SERVER_TO_NEMO_LANG)
        raise ValueError(
            f"Unsupported language code: {language!r}. Supported: {supported}."
        )
    return normalized


def _adapter_accents(config: ServerArgs) -> list[str]:
    if not config.accent_adapters:
        return []
    from zonos3.models.accent_format import list_accent_adapters

    return list_accent_adapters(config.accent_adapters)


_ACCENT_DIRECTIONS_CACHE: Dict[str, "EmotionDirections"] = {}
_ACCENT_CALIBRATION_CACHE: Dict[str, "EmotionCalibration | None"] = {}


def _accent_directions(config: ServerArgs) -> "EmotionDirections | None":
    """The --accent-directions set (loaded once): hidden-space (``attnres_source``)
    steering, or speaker-embedding (``raw``) directions for speaker_mode=embedding."""
    root = getattr(config, "accent_directions", None)
    if not root:
        return None
    if root not in _ACCENT_DIRECTIONS_CACHE:
        directions = EmotionDirections.load(root)
        if directions is None or directions.space not in ("attnres_source", "raw"):
            raise ValueError(f"--accent-directions {root}: needs an attnres_source or raw manifest.json")
        if directions.space == "raw" and not _model_supports_speaker_embedding(config):
            raise ValueError(f"--accent-directions {root}: raw (speaker-embedding) accents need a checkpoint "
                             "that conditions on speaker embeddings")
        clash = sorted(set(_accent_names(directions)) & set(_adapter_accents(config)))
        if clash:
            raise ValueError(f"accent names in both --accent-adapters and --accent-directions: {clash}")
        _ACCENT_DIRECTIONS_CACHE[root] = directions
        _ACCENT_CALIBRATION_CACHE[root] = EmotionCalibration.load(root)
        logger.info("Loaded %s accent directions from %s (%s).", directions.space, root,
                    ", ".join(_accent_names(directions)))
    return _ACCENT_DIRECTIONS_CACHE[root]


def _accent_names(directions: "EmotionDirections") -> list[str]:
    """Every direction of an accent set is a request accent, named or axis (axes take signed strengths)."""
    return directions.emotion_names + directions.axis_names


def _server_accents(config: ServerArgs) -> list[str]:
    steering = _accent_directions(config)
    return _adapter_accents(config) + (_accent_names(steering) if steering is not None else [])


def _resolve_tts_request_accent(
    config: ServerArgs, accent: str | None, strength: float
) -> tuple[str | None, float]:
    """A request's accent (None = base model; an adapter or a steering direction) and its strength."""
    if accent is None or accent == "" or accent == "none":
        return None, 1.0
    accents = _server_accents(config)
    if accent not in accents:
        loaded = ", ".join(accents) or "none (start the server with --accent-adapters or --accent-directions)"
        raise ValueError(f"Unknown accent {accent!r}; loaded accents: {loaded}.")
    if not math.isfinite(strength):
        raise ValueError(f"accent_strength must be finite, got {strength!r}.")
    return accent, float(strength)


def _steering_sources(config: ServerArgs) -> tuple[int, ...]:
    """The Full AttnRes bank sources the engine steers, ascending (as Engine._init_steering)."""
    sources = set()
    emotion = _emotion_directions_for_model(config)
    if emotion is not None and emotion.space == "attnres_source":
        sources.add(int(emotion.source_index))
    accent = _accent_directions(config)
    if accent is not None and accent.space == "attnres_source":
        sources.add(int(accent.source_index))
    bundle = _steer_bundle(config)
    if bundle is not None:
        sources.update(bundle.hidden_sources)
    return tuple(sorted(sources))


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


def _apply_steering_accent(
    config: ServerArgs,
    accent: str | None,
    strength: float,
    speaker_id: str | None,
    speaker_embedding: torch.Tensor | None,
    emotion_delta: torch.Tensor | None,
    use_calibration: bool = True,
) -> tuple[str | None, torch.Tensor | None, torch.Tensor | None]:
    """Fold a steering accent into the request: (adapter accent or None, speaker embedding, delta).

    The accent shift is ``strength * dose * direction`` with the voice's calibrated dose
    (calibration.json of the accent directions; 1.0 without one, or the raw dose when
    ``use_calibration`` is off, as for emotion sliders). A ``raw`` set moves the speaker
    embedding (norm kept, as raw emotion directions), so it needs speaker_mode=embedding.
    An ``attnres_source`` set goes into the hidden delta: with one steered bank source the
    emotion and accent deltas sum; when the two sets steer different sources the result is
    one hidden-size slot per source (``_steering_sources``), flattened, each delta in its
    set's slot. Adapter accents pass through untouched."""
    steering = _accent_directions(config)
    if accent is None or steering is None or accent not in _accent_names(steering):
        return accent, speaker_embedding, _layout_steering_delta(config, emotion_delta, None)
    cal = _ACCENT_CALIBRATION_CACHE.get(str(config.accent_directions))
    dose = cal.strength(speaker_id, accent) if (cal is not None and use_calibration) else 1.0
    if not float(strength) or not float(dose):
        return None, speaker_embedding, _layout_steering_delta(config, emotion_delta, None)
    if steering.space == "raw":
        if speaker_embedding is None:
            raise ValueError(f"Accent {accent!r} is a speaker-embedding direction and needs speaker_mode=embedding.")
        speaker_embedding = apply_emotion(
            speaker_embedding, directions=steering, sliders={accent: float(strength) * float(dose)}, strict=True
        )
        return None, speaker_embedding, _layout_steering_delta(config, emotion_delta, None)
    delta = (float(strength) * float(dose)) * (steering.named.get(accent, steering.axes.get(accent)))
    return None, speaker_embedding, _layout_steering_delta(config, emotion_delta, delta)


def _layout_steering_delta(
    config: ServerArgs, emotion_delta: torch.Tensor | None, accent_delta: torch.Tensor | None
) -> torch.Tensor | None:
    if emotion_delta is None and accent_delta is None:
        return None
    sources = _steering_sources(config)
    if len(sources) <= 1:
        # One steered source (or a speaker-row emotion delta and no accent steering).
        if emotion_delta is None:
            return accent_delta
        return emotion_delta if accent_delta is None else emotion_delta + accent_delta
    hidden = (emotion_delta if emotion_delta is not None else accent_delta).numel()
    out = torch.zeros(len(sources), hidden, dtype=torch.float32)
    if emotion_delta is not None:
        out[sources.index(int(_emotion_directions_for_model(config).source_index))] += emotion_delta
    if accent_delta is not None:
        out[sources.index(int(_accent_directions(config).source_index))] += accent_delta
    return out.flatten()


def _apply_request_controls(
    config: ServerArgs,
    req: GenerateRequest,
    speaker_embedding: torch.Tensor | None,
    steering_delta: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[tuple[int, int]] | None]:
    """Fold a request's steering-bundle controls in: (speaker embedding, hidden delta, span schedule, each schedule
    row's character range). The hidden delta adds to the emotion / accent steering delta (same per-source layout)."""
    if not req.controls and not req.control_spans:
        return speaker_embedding, steering_delta, None, None
    bundle = _steer_bundle(config)
    if bundle is None:
        raise ValueError("This server has no steering bundle (start it with --steer-bundle); `controls` are unavailable.")
    if req.emotion_enabled and (req.emotion_sliders or req.emotion_valence or req.emotion_arousal):
        raise ValueError("`controls` and the emotion_* fields cannot be combined; send the emotion as controls.")
    resolved = resolve_controls(
        bundle, req.controls, [s.model_dump() for s in req.control_spans or []], text=req.text,
        speaker_embedding=speaker_embedding, sources=_steering_sources(config),
        hidden_size=int(config.model_config.hidden_size), frames_per_char=req.control_span_rate,
    )
    delta = steering_delta
    if resolved.hidden_delta is not None:
        delta = resolved.hidden_delta if delta is None else delta + resolved.hidden_delta
    return resolved.speaker_embedding, delta, resolved.schedule, resolved.span_chars


def _span_follow(
    config: ServerArgs, req: GenerateRequest, language: str, span_chars: list[tuple[int, int]] | None
) -> tuple[list[list[int]] | None, list[float] | None]:
    """(character range of every span-schedule row, gate [lead, lag]) when the request's spans follow the reading
    position (control_span_follow "reading" and LACI heads for its language; zonos3.tts.span_follow), else (None,
    None): the character-rate schedule."""
    from zonos3.tts.span_follow import FollowConfig

    if req.control_span_follow != "reading" or not span_chars:
        return None, None
    if language not in (getattr(config, "laci_heads", None) or {}):
        return None, None
    default = FollowConfig()
    gate = [default.lead if req.control_span_lead is None else float(req.control_span_lead),
            default.lag if req.control_span_lag is None else float(req.control_span_lag)]
    return [[int(a), int(b)] for a, b in span_chars], gate


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
    audio_bytes: bytes, fade_out_ms: float, sample_rate: int = 44100
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


def _emotion_directions_root(config: ServerArgs) -> Path | None:
    raw = getattr(config, "tts_emotion_directions_dir", None)
    if not raw:
        return None
    return Path(str(raw)).expanduser()


def _get_emotion_directions(config: ServerArgs) -> "EmotionDirections | None":
    """Load (and cache) the emotion directions for the configured directory."""
    root = _emotion_directions_root(config)
    if root is None:
        return None
    key = str(root)
    if key not in _EMOTION_DIRECTIONS_CACHE:
        try:
            directions = EmotionDirections.load(root)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load emotion directions from %s: %s", root, exc)
            directions = None
        if directions is None:
            logger.info("No emotion directions found at %s; emotion control disabled.", root)
        else:
            logger.info(
                "Loaded emotion directions from %s (emotions=%s, axes=%s).",
                root, directions.emotion_names, directions.axis_names,
            )
        _EMOTION_DIRECTIONS_CACHE[key] = directions
    return _EMOTION_DIRECTIONS_CACHE[key]


def _get_emotion_calibration(config: ServerArgs) -> "EmotionCalibration | None":
    """Load (and cache) the per-speaker strength calibration, if present."""
    root = _emotion_directions_root(config)
    if root is None:
        return None
    key = str(root)
    if key not in _EMOTION_CALIBRATION_CACHE:
        try:
            cal = EmotionCalibration.load(root)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load emotion calibration from %s: %s", root, exc)
            cal = None
        if cal is not None:
            logger.info("Loaded emotion strength calibration for %d speakers from %s.",
                        len(cal.by_speaker), root)
        _EMOTION_CALIBRATION_CACHE[key] = cal
    return _EMOTION_CALIBRATION_CACHE[key]


def _emotion_directions_for_model(config: ServerArgs) -> "EmotionDirections | None":
    """Directions the loaded model can consume: embedding/proj spaces need a
    speaker embedding; ``attnres_source`` steers the Full AttnRes bank and works
    for raw-reference checkpoints too."""
    directions = _get_emotion_directions(config)
    if directions is None or directions.is_empty():
        return None
    if directions.space == "attnres_source":
        return directions if getattr(config.model_config, "residual_type", None) == "full_attnres" else None
    return directions if _model_supports_speaker_embedding(config) else None


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


@functools.lru_cache(maxsize=4)
def _text_tokenizer_spec(model_path: str) -> dict | None:
    from zonos3.tts.superbpe import load_spec

    return load_spec(model_path)


def _validate_phoneme_groups(config: ServerArgs, text: str) -> None:
    """Reject malformed pronunciation groups here, before the tokenizer worker sees them."""
    from zonos3.tts.superbpe import PHONEME_CLOSE, PHONEME_OPEN, payload_ids

    spec = _text_tokenizer_spec(str(config.model_path))
    phonemes = None if spec is None else spec.get("phonemes")
    open_, close = (PHONEME_OPEN, PHONEME_CLOSE) if phonemes is None else (phonemes["open_marker"], phonemes["close_marker"])
    if open_ not in text and close not in text:
        return
    if phonemes is None:
        raise HTTPException(status_code=400, detail="This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups")
    try:
        payload_ids(text, spec)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Pronunciation groups: {exc}") from exc


_TAG_CFG_MIN_SCALE, _TAG_CFG_MAX_SCALE = 1.0, 4.0


@functools.lru_cache(maxsize=4)
def _model_trained_on_tags(model_path: str) -> bool:
    """zonos3.tts.markup.trained_on_tags, memoized per checkpoint path."""
    return trained_on_tags(model_path)


def _tag_cfg_capabilities(config: ServerArgs) -> dict:
    return {"available": True, "trained": _model_trained_on_tags(str(config.model_path)),
            "tags": list(NONVERBAL_MARKUP_TAGS), "min_scale": _TAG_CFG_MIN_SCALE, "max_scale": _TAG_CFG_MAX_SCALE,
            "default_scale": 1.0}


def _validate_tag_cfg(
    config: ServerArgs, text: str, tag_cfg_scale: float, *, speaker_cfg_scale: float, emotion_cfg_scale: float
) -> None:
    """Tag CFG (tagged-vs-plain guidance on nonverbal markup) request rules; raises HTTP 400 when broken.

    1.0 is the unguided path (the schema bounds the scale to [1, 4]). Above it the text must carry recognized markup
    whose removal changes it, and the guidance cannot be stacked with speaker or emotion CFG (one twin per request)."""
    def reject(detail: str) -> None:
        raise HTTPException(status_code=400, detail=detail)

    if float(tag_cfg_scale) == 1.0:
        return
    if not has_nonverbal_markup(text) or strip_nonverbal_markup(text) == text:
        reject(
            "tag_cfg_scale above 1.0 requires recognized nonverbal markup in the text: point tags "
            + ", ".join(f"<{tag}/>" for tag in NONVERBAL_MARKUP_TAGS)
            + " or the same names as span tags (<shouting>...</shouting>)."
        )
    if float(speaker_cfg_scale) != 1.0:
        reject("tag_cfg_scale above 1.0 cannot be combined with speaker_cfg_scale; set speaker_cfg_scale to 1.0.")
    if float(emotion_cfg_scale) != 1.0:
        reject("tag_cfg_scale above 1.0 cannot be combined with emotion_cfg_scale; set emotion_cfg_scale to 1.0.")



def _apply_request_emotion(
    config: ServerArgs,
    speaker_embedding: torch.Tensor | None,
    *,
    speaker_id: str | None = None,
    emotion_enabled: bool,
    sliders: Dict[str, float] | None,
    valence: float,
    arousal: float,
    strength: float,
    use_calibration: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve emotion control into a (speaker_embedding, hidden_delta) pair.

    For ``raw``/``lda`` directions the embedding itself is nudged and the hidden
    delta is ``None``. For ``proj`` directions the embedding is left unchanged
    and the returned hidden delta is added to the projected speaker vector
    inside the model. For ``attnres_source`` directions no embedding is needed:
    the hidden delta is added to the steered Full AttnRes bank source on every
    generated frame. No-op when emotion is disabled / nothing requested.
    ``use_calibration=False`` skips the per-speaker strength fold so
    ``strength`` is the raw injection dose. Raises ValueError on bad input
    (mapped to HTTP 400).
    """
    if not emotion_enabled:
        return speaker_embedding, None
    has_request = bool(sliders) or float(valence) != 0.0 or float(arousal) != 0.0
    if not has_request:
        return speaker_embedding, None
    directions = _emotion_directions_for_model(config)
    if directions is None:
        raise ValueError(
            "Emotion control requested but no emotion directions are configured for this model. "
            "Start the server with --tts-emotion-directions-dir (this repo ships "
            "steering/emotion_directions_mect60k_axes)."
        )
    if speaker_embedding is None and directions.space != "attnres_source":
        return speaker_embedding, None
    # Fold the per-speaker, per-emotion calibrated strength into the weights so a
    # single global `strength` (user multiplier, default 1.0) drives all emotions
    # at their voice-specific optimum. Skipped when the request opts out, in
    # which case `strength` is the raw dose.
    cal = _get_emotion_calibration(config)
    if cal is not None and use_calibration:
        sliders = {e: float(w) * cal.pole_strength(speaker_id, e, float(w))
                   for e, w in (sliders or {}).items()}
        valence = float(valence) * cal.pole_strength(speaker_id, "valence", float(valence))
        arousal = float(arousal) * cal.pole_strength(speaker_id, "arousal", float(arousal))

    if directions.space in HIDDEN_SPACES:
        delta = emotion_hidden_delta(
            directions, sliders=sliders, valence=valence, arousal=arousal,
            strength=strength, strict=True,
        )
        return speaker_embedding, delta
    emb = apply_emotion(
        speaker_embedding,
        directions=directions,
        sliders=sliders,
        valence=valence,
        arousal=arousal,
        strength=strength,
        strict=True,
    )
    return emb, None


def _speaker_emotion_disabled(config: ServerArgs, speaker_id: str) -> list[str]:
    """Controls calibrated to dose 0 for this speaker (UI greys them out)."""
    directions = _emotion_directions_for_model(config)
    cal = _get_emotion_calibration(config) if directions is not None else None
    if cal is None:
        return []
    return cal.disabled(speaker_id, directions.emotion_names + directions.axis_names)


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
        "emotion_disabled": bool(_speaker_emotion_disabled(config, entry.speaker_id)),
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
        "emotion_disabled": bool(_speaker_emotion_disabled(config, entry.speaker_id)),
    }


def _default_voice_root(config: ServerArgs) -> Path | None:
    raw = getattr(config, "tts_default_voices_dir", None)
    if not raw:
        return None
    return Path(str(raw)).expanduser()


def _default_voice_source_type(path: Path) -> Literal["audio", "embedding"] | None:
    if path.name.lower().endswith(".codes.npy"):  # an audio voice's reference codes (the ggml server's)
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


def _reference_codes_tensor(codes) -> "torch.Tensor | None":
    """Validate caller-supplied reference prefix codes ([frames, n_codebooks] ints)."""
    if codes is None:
        return None
    import torch

    tensor = torch.as_tensor(codes, dtype=torch.int32)
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[0] > 16384:
        raise ValueError("speaker_reference_codes must be a non-empty [frames, codebooks] list")
    if bool((tensor < 0).any()):
        raise ValueError("speaker_reference_codes must be non-negative")
    return tensor.contiguous()


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


def _check_laci_request(config, laci: bool, language: str, speaker_cfg_scale: float, emotion_cfg_scale: float,
                        duration_target: bool = False, adjust_rate: bool = True, tag_cfg_scale: float = 1.0) -> None:
    """HTTP 400 unless a laci=true request meets LACI's constraints (zonos3.tts.laci): heads for its language, no CFG,
    and for a duration target that moves the speaking-rate bucket the naive cache (the scheduler refuses the rest)."""
    if not laci:
        if duration_target:
            raise HTTPException(status_code=400, detail="A LACI duration target needs laci=true")
        return
    heads = getattr(config, "laci_heads", None) or {}
    if not heads:
        raise HTTPException(status_code=400, detail="LACI is not enabled on this server (start it with --laci-heads LANG=PATH)")
    if language not in heads:
        raise HTTPException(status_code=400, detail=f"No LACI alignment heads for language {language!r} (configured: {', '.join(sorted(heads))})")
    if float(speaker_cfg_scale) != 1.0 or float(emotion_cfg_scale) != 1.0:
        raise HTTPException(status_code=400, detail="LACI needs speaker_cfg_scale = emotion_cfg_scale = 1")
    if float(tag_cfg_scale) != 1.0:
        raise HTTPException(status_code=400, detail="LACI needs tag_cfg_scale = 1")
    if duration_target and adjust_rate and getattr(config, "cache_type", "naive") != "naive":
        raise HTTPException(status_code=400, detail="A LACI duration target that adjusts the speaking rate needs --cache-type naive")


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
        fade_bytes = int(44100 * max(0.0, fade_out_ms) / 1000.0) * 4
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
            audio_s = total_audio_bytes / (44100 * 4)  # float32 @ 44.1kHz
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
            audio_s = len(audio_data) / (44100 * 4)
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
    if _DUB_RUNTIME is not None:
        _DUB_RUNTIME.close()
    _FUSEALIGN.close()


# The spec of this API is api/openapi.yaml; FastAPI's generated docs would describe a second one.
app = FastAPI(title="Zonos3 API Server", version="2.0.0", lifespan=lifespan,
              openapi_url=None, docs_url=None, redoc_url=None)

# Innermost, so CORS and the access check wrap its early status line (zonos3.server.keepalive). The /v2/edit
# prefix covers /v2/edit/transcribe and /v2/edit/align.
app.add_middleware(
    KeepaliveMiddleware,
    paths=("/v2/edit", "/v2/dub/ref", "/v2/dub/lips", "/v2/dub/transcribe", "/v2/dub/analyze", "/v2/dub/generate",
           "/v2/dub/regenerate"),
)

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
) -> tuple[bytes | None, torch.Tensor | None, torch.Tensor | None]:
    """(reference audio, speaker embedding, hidden emotion delta) for a request.

    ``speaker_mode`` (from ``_resolve_speaker_mode``) picks the conditioning
    path: "reference" takes a waveform (upload, default voice or session clip)
    and no embedding, "embedding" the reverse. Emotion control is resolved for
    both: hidden-space (``attnres_source``) directions steer the model directly,
    embedding-space directions need the embedding.
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
    speaker_embedding, speaker_emotion_delta = _apply_request_emotion(
        config,
        speaker_embedding,
        speaker_id=req.speaker_embedding_id,
        emotion_enabled=req.emotion_enabled,
        sliders=req.emotion_sliders,
        valence=req.emotion_valence,
        arousal=req.emotion_arousal,
        strength=req.emotion_strength,
        use_calibration=req.emotion_calibrated,
    )
    return speaker_reference_audio, speaker_embedding, speaker_emotion_delta


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
    it samples with (the request's, else drawn uniformly in 0..2^31-1).
    ``laci_target_frames`` / ``laci_adjust_rate``: the LACI duration target of a dub take (needs req.laci)."""
    _validate_tag_cfg(
        state.config, req.text, req.tag_cfg_scale,
        speaker_cfg_scale=req.speaker_cfg_scale, emotion_cfg_scale=req.emotion_cfg_scale,
    )
    speaker_mode = _resolve_speaker_mode(state.config, req.speaker_mode)
    condition_ids = _resolve_condition_buckets(state.config, req.condition_buckets, req.condition_values)
    try:
        speaker_reference_audio, speaker_embedding, speaker_emotion_delta = (
            await _resolve_request_speaker_conditioning(state.config, req, session_id, speaker_mode)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        language = _normalize_tts_request_language(req.language)
        reference_codes = _reference_codes_tensor(req.speaker_reference_codes)
        max_tokens = _resolve_tts_max_tokens(state.config, req.max_tokens)
        accent, accent_strength = _resolve_tts_request_accent(
            state.config, req.accent, req.accent_strength
        )
        accent, speaker_embedding, speaker_emotion_delta = _apply_steering_accent(
            state.config, accent, accent_strength, req.speaker_embedding_id, speaker_embedding,
            speaker_emotion_delta, use_calibration=req.emotion_calibrated,
        )
        speaker_embedding, speaker_emotion_delta, steer_schedule, span_chars = _apply_request_controls(
            state.config, req, speaker_embedding, speaker_emotion_delta
        )
        follow_chars, follow_gate = _span_follow(state.config, req, language, span_chars)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stream_decode = _resolve_stream_decode(state.config, req.stream_decode, streaming=bool(req.stream))
    if req.speaker_cfg_scale != 1.0:
        has_reference = speaker_mode == "reference" and (speaker_reference_audio is not None or reference_codes is not None)
        has_embedding = speaker_mode == "embedding" and speaker_embedding is not None
        if not (has_reference or has_embedding):
            raise HTTPException(status_code=400, detail="Speaker CFG needs a speaker prompt: speaker_mode=reference with reference audio or codes, or speaker_mode=embedding with a clip or an embedding")
        if req.emotion_cfg_scale != 1.0:
            raise HTTPException(status_code=400, detail="Speaker and emotion CFG cannot be combined")
    _validate_phoneme_groups(state.config, req.text)
    _check_laci_request(state.config, req.laci, language, req.speaker_cfg_scale, req.emotion_cfg_scale,
                        duration_target=laci_target_frames is not None, adjust_rate=laci_adjust_rate,
                        tag_cfg_scale=req.tag_cfg_scale)
    return TTSTokenizeMsg(
        uid=state.new_user(),
        text=req.text,
        language=language,
        text_normalization=req.text_normalization,
        silence_prefill_frames=req.silence_prefill_frames,
        decode_context=req.decode_context,
        decode_context_frames=int(req.decode_context_frames or 30),
        sampling_params=TTSSamplingParams(
            temperature=req.temperature,
            topk=req.topk,
            top_p=req.top_p,
            min_p=req.min_p,
            max_tokens=max_tokens,
            repetition_window=req.repetition_window,
            repetition_penalty=req.repetition_penalty,
            repetition_codebooks=req.repetition_codebooks,
            seed=req.seed if req.seed is not None else secrets.randbelow(2**31),
            emotion_cfg_scale=req.emotion_cfg_scale,
            speaker_cfg_scale=req.speaker_cfg_scale,
            tag_cfg_scale=req.tag_cfg_scale,
            stream_decode=stream_decode,
        ),
        speaker_embedding=speaker_embedding,
        speaker_reference_audio=speaker_reference_audio,
        speaker_reference_codes=reference_codes,
        speaker_emotion_delta=speaker_emotion_delta,
        condition_buckets=condition_ids,
        accent=accent,
        accent_strength=accent_strength,
        steer_schedule=steer_schedule,
        steer_follow_chars=follow_chars,
        steer_follow_gate=follow_gate,
        laci=req.laci,
        laci_target_frames=laci_target_frames,
        laci_adjust_rate=laci_adjust_rate,
    )


def _pcm16_wav(audio_bytes: bytes, sample_rate: int = 44100) -> bytes:
    """float32 PCM bytes as a 16-bit mono WAV file, samples clipped to [-1, 1]."""
    samples = np.clip(np.frombuffer(audio_bytes, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes((samples * 32767).astype("<i2").tobytes())
    return buf.getvalue()


async def _generate_response(state: FrontendManager, req: GenerateRequest, session_id: str | None) -> Response:
    """Run a GenerateRequest: streamed or buffered float32 PCM, or a buffered 16-bit WAV."""
    if req.stream and req.format == "wav":
        raise HTTPException(status_code=400, detail="format=wav needs stream=false")
    msg = await _tts_generate_msg(state, req, session_id)
    uid = msg.uid
    await state.send_one(msg)

    async def _abort():
        await state.abort_user(uid)

    headers = {"X-Audio-Sample-Rate": "44100", "X-Audio-Channels": "1",
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
        return Response(content=_pcm16_wav(audio_data), media_type="audio/wav", headers=headers)
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
    from zonos3.tokenizer.textnorm import SERVER_TO_NEMO_LANG, normalization_enabled

    state = get_global_state()
    return {
        "text_tokenizer_spec": _text_tokenizer_spec(str(state.config.model_path)),
        "text_normalization_enabled": normalization_enabled(),
        "text_norm_languages": list(SERVER_TO_NEMO_LANG),
        "accents": _server_accents(state.config),
        # the subset served as hidden-space steering directions (--accent-directions)
        "accent_steering": (_accent_names(_accent_directions(state.config))
                            if _accent_directions(state.config) is not None else []),
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
        "max_tokens": _model_tts_max_tokens(state.config),
        "stream_decode_modes": list(_STREAM_DECODE_MODES) if _model_prefix_stream_supported(state.config) else ["whole"],
        "stream_decode_default": _default_stream_decode(state.config),
        "silence_prefill": _silence_prefill_spec(state.config),
        "tag_cfg": _tag_cfg_capabilities(state.config),
        "decode_context": {"available": True, "default_frames": 30, "max_frames": 450,
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
        # LACI (laci=true): languages with alignment heads (--laci-heads); needs speaker / emotion CFG 1
        "laci_languages": sorted(getattr(state.config, "laci_heads", None) or {}),
        # the studio's Dub screen (/v2/dub/*, /v2/dub/capabilities): LACI heads and a speaking-rate condition row
        "dub": _dub_available(state.config),
        "speaker_embedding_upload": _model_supports_speaker_embedding(state.config),
        "speaker_embedding_cache": _model_supports_speaker(state.config),   # cached clips for raw-reference models too
        "speaker_embedding_blend": _model_supports_speaker_embedding(state.config),
        "default_voices_enabled": bool(_default_voice_root(state.config)),
        # editing needs the checkpoint's span-end/marker rows, Qwen3-ASR (.venv-asr) and FuseAlign (fusealign_hub)
        "speech_editing": _model_supports_editing(state.config) and _ASR.installed and fusealign_installed(),
        "edit_max_spans": (
            _edit_max_spans(state.config) if _model_supports_editing(state.config) else 0
        ),
        "edit_frame_rate": (
            _edit_frame_rate(state.config) if _model_supports_editing(state.config) else None
        ),
        "edit_max_seconds": _EDIT_MAX_SECONDS,
        "edit_align_max_seconds": _EDIT_ALIGN_MAX_SECONDS,
        **_emotion_capabilities(state.config),
        **_phoneme_capabilities(state.config),
        # server-side studio projects (/v2/projects, /v2/media; --studio-projects-dir)
        "projects": (store.capabilities() if (store := _project_store()) is not None else {"enabled": False}),
    }


def _phoneme_capabilities(config: ServerArgs) -> dict:
    """Pronunciation input: the checkpoint's phoneme tokens (if any) and the configured lexicon."""
    spec = _text_tokenizer_spec(str(config.model_path))
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
        result = lexicon.phonemize(request.text, request.language, guess=request.guess)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    spec = _text_tokenizer_spec(str(config.model_path))
    if spec is not None and spec.get("phonemes") is not None:
        symbols = set(spec["phonemes"]["symbols"])
        for segment in result["segments"]:
            if segment["ipa"] and not symbols.issuperset(segment["ipa"]):
                segment.update(ipa=None, status="unsupported")
    return result


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


def _emotion_capabilities(config: ServerArgs) -> dict:
    """Report which emotion sliders/axes/presets the loaded model exposes."""
    directions = _emotion_directions_for_model(config)
    if directions is None:
        return {
            "emotion_enabled": False,
            "emotion_names": [],
            "emotion_axes": [],
            "emotion_tiers": {},
            "emotion_presets": {},
            "emotion_space": None,
            "emotion_calibrated": False,
        }
    return {
        "emotion_enabled": True,
        "emotion_names": directions.emotion_names,
        "emotion_axes": directions.axis_names,
        "emotion_tiers": directions.tiers,
        "emotion_presets": directions.presets,
        "emotion_space": directions.space,
        "emotion_calibrated": _get_emotion_calibration(config) is not None,
    }


# =============================================================================
# Speech editing
# =============================================================================


def _edit_max_spans(config: ServerArgs) -> int:
    """Edit marker rows: the extra text rows not claimed by phoneme tokens."""
    return edit_marker_rows(
        int(getattr(config.model_config, "text_extra_tokens", 0)), _text_tokenizer_spec(str(config.model_path))
    )


def _model_supports_editing(config: ServerArgs) -> bool:
    model_config = config.model_config
    return (
        getattr(model_config, "span_end_id", None) is not None
        and _edit_max_spans(config) > 0
    )


def _require_editing(config: ServerArgs) -> None:
    if not _model_supports_editing(config):
        raise HTTPException(
            status_code=400,
            detail="The loaded checkpoint does not support speech editing (needs span_end_id and marker rows).",
        )


async def _edit_worker(fn, *args):
    """A blocking ASR / FuseAlign call of zonos3.tts.edit_align in a thread: HTTP 503 when a worker is unavailable
    (``WorkerError``), 400 when the input is at fault (``ValueError``); the messages are the workers' clients'."""
    try:
        return await asyncio.to_thread(fn, *args)
    except WorkerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _edit_language(language: str) -> tuple[str, str]:
    """``(Whisper-style code, Qwen3-ASR name)`` of a request's ``language`` (edit_align.edit_language over the ASR
    worker's languages; HTTP 400 for one editing does not serve)."""
    try:
        normalized = _normalize_tts_request_language(language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    supported = await _edit_worker(_ASR.languages)
    try:
        return edit_language(normalized, supported)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _edit_transcribe(audio_path: str, language: tuple[str, str]):
    """Qwen3-ASR transcript of a clip in the edit's language (edit_align.transcribe_audio: chunk-wise past 30 s)."""
    return await _edit_worker(transcribe_audio, _ASR, _FUSEALIGN, audio_path, language[1])


async def _edit_aligned_words(audio_path: str, text: str, language: tuple[str, str], utterances=None):
    """The words of ``text`` in the clip with their edit tokens (edit_align.edit_aligned_words; ``utterances``: the
    long-form anchors of a transcript of the same clip)."""
    return await _edit_worker(edit_aligned_words, _ASR, _FUSEALIGN, audio_path, text, language, utterances)


def _edit_frame_rate(config: ServerArgs) -> float:
    from zonos3_sgl.tokenizer.vocoder import TTSVocoderManager

    manager = TTSVocoderManager(
        n_codebooks=config.tts_n_codebooks,
        audio_pad_id=config.tts_audio_pad_id,
        audio_codec=config.tts_audio_codec,
    )
    return manager.sample_rate / manager.hop_length


def _write_edit_audio(audio_bytes: bytes, directory: str, max_seconds: float) -> tuple[str, float]:
    """Decode uploaded audio like the codec path does and store it as WAV for the ASR and FuseAlign workers."""
    waveform, sample_rate = _decode_audio_bytes(audio_bytes)
    duration = waveform.shape[-1] / sample_rate
    if duration > max_seconds:
        raise ValueError(f"Audio is {duration:.1f} s long; at most {max_seconds:.0f} s are supported here.")
    path = os.path.join(directory, "original.wav")
    sf.write(path, waveform[0].numpy(), sample_rate, subtype="FLOAT")
    return path, duration


def _public_words(words) -> list[dict]:
    return [
        {
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "confidence": w.confidence,
            "tokens": w.tokens,
        }
        for w in words
    ]


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


def _edit_region_doses(config: ServerArgs, req: EditRequest) -> list[dict[str, float]] | None:
    """Steering doses of each edited region: ``controls`` plus ``region_controls[k]``, summed per
    id (zero doses dropped), or None without controls. Checks the ids and doses against the
    steering bundle, and that speaker-embedding (raw) controls have a speaker row to move."""
    if req.region_controls is not None:
        if req.spans is None:
            raise ValueError("region_controls needs explicit spans (one entry per span).")
        if len(req.region_controls) != len(req.spans):
            raise ValueError(
                f"region_controls has {len(req.region_controls)} entries for {len(req.spans)} spans; "
                "send one per span (null for none)."
            )
    regions = len(req.spans) if req.spans is not None else 1
    doses = []
    for k in range(regions):
        merged = {did: float(v) for did, v in (req.controls or {}).items()}
        extra = req.region_controls[k] if req.region_controls is not None else None
        for did, v in (extra or {}).items():
            merged[did] = merged.get(did, 0.0) + float(v)
        doses.append({did: v for did, v in merged.items() if v != 0.0})
    if not any(doses):
        return None
    bundle = _steer_bundle(config)
    if bundle is None:
        raise ValueError("This server has no steering bundle (start it with --steer-bundle); `controls` are unavailable.")
    for region in doses:
        check_controls(bundle, region)
    raw = sorted({did for region in doses for did in region if bundle.directions[did].space == RAW_SPACE})
    if raw and not _edit_uses_speaker_row(config, req):
        raise ValueError(
            f"Controls {', '.join(raw)} move the speaker embedding and need the speaker row "
            "(speaker_embedding true or unset, on a speaker-embedding checkpoint)."
        )
    return doses


def _edit_uses_speaker_row(config: ServerArgs, req: EditRequest) -> bool:
    return _model_supports_speaker_embedding(config) if req.speaker_embedding is None else bool(req.speaker_embedding)


@app.post("/v2/edit/transcribe")
async def tts_edit_transcribe(req: EditTranscribeRequest):
    """Qwen3-ASR transcript of the audio to edit (chunk-wise past 30 s) and its words' FuseAlign timings."""
    state = get_global_state()
    _require_editing(state.config)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            audio = _decode_base64_blob(req.audio_base64, "audio_base64")
            path, duration = _write_edit_audio(audio, tmp, _EDIT_ALIGN_MAX_SECONDS)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        language = await _edit_language(req.language)
        transcript = await _edit_transcribe(path, language)
        words = await _edit_aligned_words(path, transcript.text, language, transcript.utterances)
    return {"text": transcript.text, "duration": duration, "words": _public_words(words)}


@app.post("/v2/edit/align")
async def tts_edit_align(req: EditAlignRequest):
    """FuseAlign forced alignment of the audio to ``text``: word timings, presences (``confidence``) and edit tokens."""
    state = get_global_state()
    _require_editing(state.config)
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty.")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            audio = _decode_base64_blob(req.audio_base64, "audio_base64")
            path, duration = _write_edit_audio(audio, tmp, _EDIT_ALIGN_MAX_SECONDS)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        words = await _edit_aligned_words(path, text, await _edit_language(req.language))
    return {"text": text, "duration": duration, "words": _public_words(words)}


@app.post("/v2/edit")
async def tts_edit(req: EditRequest):
    """Regenerate the edited regions of recorded speech; unedited codec frames are kept exactly.

    Each region is one span-end-terminated generation over [speaker row][condition rows][text]
    [unmasked context][earlier fills][marker]. A clip longer than ``max_context_seconds`` is cut
    into word-snapped windows around the regions (``plan_edit_windows``): each window's prompts
    hold its codes and its stretch of the target text, and every fill splices into the full
    clip, which is decoded once. Region k's steering (``controls`` + ``region_controls[k]``)
    moves its speaker row and adds its hidden delta on the frames it generates; like
    /v2/generate, the engine never steers the prefilled prompt rows.
    """
    state = get_global_state()
    config = state.config
    _require_editing(config)
    if req.tag_cfg_scale != 1.0:
        raise HTTPException(
            status_code=400,
            detail="tag_cfg_scale is not supported by /v2/edit (speech-edit requests run without tag CFG); use 1.0.",
        )
    max_spans = _edit_max_spans(config)
    n_codebooks = int(config.tts_n_codebooks)
    frame_rate = _edit_frame_rate(config)
    # Nonverbal markup is text for the model only: markers (canonical, no inner whitespace) ride in the prompt
    # text and never reach the ASR or FuseAlign (word diff, alignment, verify; target_tokens).
    target_text = canonical_markup(req.target_text.strip())
    plain_target_text = strip_nonverbal_markup(target_text)
    edit_lang = await _edit_language(req.language)  # every ASR and FuseAlign call of the edit
    with tempfile.TemporaryDirectory() as tmp:
        try:
            if not target_text:
                raise ValueError("target_text is empty.")
            _validate_phoneme_groups(config, target_text)
            if req.original_words is not None and not (req.original_text or "").strip():
                raise ValueError("original_words needs the original_text they spell.")
            region_doses = _edit_region_doses(config, req)
            language = _normalize_tts_request_language(req.language)
            use_speaker_row = _edit_uses_speaker_row(config, req)
            if use_speaker_row and not _model_supports_speaker_embedding(config):
                raise ValueError("speaker_embedding needs a checkpoint with a speaker-embedding row.")
            condition_ids = _resolve_condition_buckets(config, req.condition_buckets, req.condition_values)
            audio = _decode_base64_blob(req.audio_base64, "audio_base64")
            original_path, duration = _write_edit_audio(audio, tmp, _EDIT_MAX_SECONDS)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        speaker_embedding = None
        if use_speaker_row:
            waveform, sample_rate = _decode_audio_bytes(audio)
            try:
                speaker_embedding = await _compute_speaker_embedding_from_waveform(
                    waveform,
                    sample_rate,
                    expected_dim=_model_speaker_dim(config),
                    speaker_codec=_model_speaker_codec(config),
                )
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        encoded = await _backend_reply(
            state, lambda uid: TTSEncodeAudioMsg(uid=uid, audio=audio, max_seconds=_EDIT_MAX_SECONDS)
        )
        if encoded.error is not None:
            raise HTTPException(status_code=400, detail=encoded.error)
        codes = encoded.codes
        # Longer clips are windowed and every fill gets a minimum length; both read word timings
        # of the original (the caller's original_words, else a FuseAlign alignment) and the target
        # words' edit tokens. Without original_text, a short clip with explicit spans is not
        # transcribed: its regions' own lengths are expected.
        windowed = len(codes) / frame_rate > req.max_context_seconds
        target_words = target_text.split()
        # [target tokens, tokens of each target word]; a pronunciation group is one target token (target_tokens).
        normalized_target = target_tokens(target_text)
        original_text, words = req.original_text, None
        if original_text:
            original_text = strip_nonverbal_markup(original_text)  # what the workers read (and the response reports)
        if req.original_words is not None:
            given = [w.model_dump() for w in req.original_words]
            for w in given:
                if w["tokens"] is None:
                    w["tokens"] = textmatch.tokens(w["word"])
            try:
                words = edit_words(given, [w["tokens"] for w in given], textmatch.tokens(original_text),
                                   duration=duration)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        elif req.spans is None or windowed or (req.min_fill_ratio > 0 and original_text):
            utterances = None
            if not original_text:
                transcript = await _edit_transcribe(original_path, edit_lang)
                original_text, utterances = transcript.text, transcript.utterances
            words = await _edit_aligned_words(original_path, original_text, edit_lang, utterances)
        # verify with pronunciation groups: the reference's tokens, each group a wildcard word
        verify_reference = normalized_target[0] if req.verify and GROUP_OPEN in target_text else None
        try:
            if req.spans is not None:
                spans = seconds_to_spans(req.spans, len(codes), frame_rate)
            else:
                spans = diff_spans(
                    words,
                    normalized_target[0],
                    total_frames=len(codes),
                    frame_rate=frame_rate,
                    margin_seconds=req.margin_seconds,
                    max_spans=max_spans,
                ).spans
            validate_edit_spans(spans, len(codes), max_spans)
            windows = [whole_clip_window(spans, len(codes), target_text)]
            if windowed:
                windows = plan_edit_windows(
                    spans,
                    words,
                    target_words,
                    normalized_target[1:],
                    total_frames=len(codes),
                    frame_rate=frame_rate,
                    context_seconds=req.context_seconds,
                    target_text=target_text,
                )
            if words is not None:
                expected = expected_fill_seconds(
                    spans, words, target_words, normalized_target[1:], frame_rate=frame_rate
                )
            else:
                expected = [length / frame_rate for _, length in spans]
            # The span-end token is masked until a fill reaches min_fill_ratio of its expected length.
            min_frames = [
                min(int(req.min_fill_ratio * seconds * frame_rate), int(req.max_span_seconds * frame_rate))
                for seconds in expected
            ]
            max_tokens = _resolve_tts_max_tokens(
                config, int(req.max_span_seconds * frame_rate) + n_codebooks + 1
            )
            # Per region: (speaker-row embedding, hidden steering delta). Without explicit spans
            # the request's controls apply to every diffed region alike.
            steering = [(speaker_embedding, None)] * len(spans)
            if region_doses is not None:
                bundle = _steer_bundle(config)
                steering = []
                for k in range(len(spans)):
                    resolved = resolve_controls(
                        bundle, region_doses[k if req.spans is not None else 0], None, text=target_text,
                        speaker_embedding=speaker_embedding, sources=_steering_sources(config),
                        hidden_size=int(config.model_config.hidden_size),
                    )
                    steering.append((resolved.speaker_embedding, resolved.hidden_delta))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        base_seed = req.seed if req.seed is not None else random.randrange(2**31 - 8)
        base_sampling = TTSSamplingParams(
            temperature=req.temperature,
            topk=req.topk,
            top_p=req.top_p,
            min_p=req.min_p,
            max_tokens=max_tokens,
            repetition_window=req.repetition_window,
            repetition_penalty=req.repetition_penalty,
            repetition_codebooks=req.repetition_codebooks,
            seed=base_seed,
            end_token="span",
        )

        async def _edit_take(k: int) -> dict:
            sampling_params = replace(base_sampling, seed=base_seed + k)
            fills, stopped = [None] * len(spans), [None] * len(spans)
            for window in windows:
                window_codes = codes[window.start: window.end]
                window_spans = [[int(start), int(length)] for start, length in window.spans(spans)]
                window_fills = []
                for region in window.regions:
                    region_embedding, region_delta = steering[region]
                    reply = await _backend_reply(
                        state,
                        lambda uid: TTSTokenizeMsg(
                            uid=uid,
                            text=window.text,
                            sampling_params=replace(sampling_params, min_frames=min_frames[region]),
                            language=language,
                            text_normalization=req.text_normalization,
                            edit_codes=window_codes,
                            edit_spans=window_spans,
                            edit_fills=list(window_fills),
                            speaker_embedding=region_embedding,
                            speaker_emotion_delta=region_delta,
                            condition_buckets=condition_ids,
                        ),
                    )
                    fill, ended = extract_fill(
                        reply.codes,
                        reply.eos_frame,
                        audio_pad_id=config.tts_audio_pad_id,
                        codebook_sizes=config.model_config.codebook_sizes,
                    )
                    window_fills.append(fill.to(torch.int32))
                    fills[region], stopped[region] = window_fills[-1], ended
            spliced = splice_edit(codes, spans, fills).to(torch.int32)
            decoded = await _backend_reply(state, lambda uid: TTSDecodeCodesMsg(uid=uid, codes=spliced))
            edited = np.frombuffer(decoded.audio_data, dtype=np.float32)
            verify, take_words = None, None
            if req.verify:
                edited_path = os.path.join(tmp, f"take{k}.wav")
                sf.write(edited_path, edited, 44100, subtype="FLOAT")
                # The take's transcript (Qwen3-ASR) and its anchors for the alignment of the take's words.
                transcript = await _edit_transcribe(edited_path, edit_lang)
                if verify_reference is None:
                    scored = textmatch.verify(plain_target_text, transcript.text)
                    verify = {
                        "asr_text": transcript.text,
                        "wer": scored["word_errors"] / max(1, scored["ref_words"]),
                        "cer": scored["cer"],
                    }
                    take_words = _public_words(await _edit_aligned_words(
                        edited_path, plain_target_text, edit_lang, transcript.utterances))
                else:
                    # Groups are wildcard words: any one recognized word matches; the take's words
                    # come from aligning the target with each group replaced by the word heard.
                    scored = wildcard_verify(verify_reference, textmatch.tokens(transcript.text))
                    verify = {
                        "asr_text": transcript.text,
                        "wer": scored["word_errors"] / max(1, scored["ref_words"]),
                        "cer": scored["cer"],
                        "groups": scored["groups"],
                    }
                    heard_text = fill_groups(plain_target_text, [g["heard"] for g in scored["groups"]])
                    if heard_text:
                        take_words = _public_words(await _edit_aligned_words(
                            edited_path, heard_text, edit_lang, transcript.utterances))
            wav = io.BytesIO()
            sf.write(wav, edited, 44100, subtype="PCM_16", format="WAV")
            edited_regions, shift = [], 0
            for (start, length), fill in zip(spans, fills, strict=True):
                edited_regions.append(
                    [(start + shift) / frame_rate, (start + shift + len(fill)) / frame_rate]
                )
                shift += len(fill) - length
            return {
                "seed": base_seed + k,
                "audio_base64": base64.b64encode(wav.getvalue()).decode("ascii"),
                "duration": len(edited) / 44100,
                "edited_regions": edited_regions,
                "fill_seconds": [len(fill) / frame_rate for fill in fills],
                "stopped": stopped,
                "verify": verify,
                "words": take_words,
            }

        takes = await asyncio.gather(*(_edit_take(k) for k in range(req.num_takes)))
    return {
        "sample_rate": 44100,
        "original_text": original_text,
        "target_text": target_text,
        "words": None if words is None else _public_words(words),
        "spans": [[start / frame_rate, (start + length) / frame_rate] for start, length in spans],
        # The audio the model saw per window (full-clip seconds); null when the clip was edited whole.
        "contexts": [[w.start / frame_rate, w.end / frame_rate] for w in windows] if windowed else None,
        # Each region's expected fill length (seconds); its fills last at least min_fill_ratio of it.
        "expected_seconds": expected,
        "takes": list(takes),
    }


# ----------------------------------------------------------------------------- dubbing (zonos3.dub, docs/laci_dub.md)
# The studio's Dub screen: references, the dubbing worker (.venv-asr; ZONOS3_ASR_PYTHON) and the dub plan live in
# zonos3.dub.runtime, which shares the server's ASR and FuseAlign workers with speech editing; every take is a /v2/generate request on this server's scheduler with laci=true and the take's
# LACI duration target, so dub takes use studio voices, steering controls and pronunciation groups.
_DUB_RUNTIME = None
_DUB_TAKE_SLOTS: asyncio.Semaphore | None = None
_DUB_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


def _dub_rate_edges(config: ServerArgs) -> list[float]:
    spec = _condition_spec(config)
    return [] if spec is None else list(spec["edges"].get("speaking_rate") or [])


def _dub_available(config: ServerArgs) -> bool:
    """A dub needs LACI alignment heads (--laci-heads) and the speaking-rate condition row its takes are planned in."""
    return bool(getattr(config, "laci_heads", None)) and bool(_dub_rate_edges(config))


def _dub_rate_scales(config: ServerArgs, bundle) -> dict:
    """Voice mode ("studio" / "source") -> TTS language -> its speaking-rate scale with the tempo-steered buckets of the
    steering bundle's rate_extension (README "Extended speaking-rate buckets"; source voices, in reference mode, only
    on its reference_sides); empty without one, or when it was measured against other row edges."""
    from zonos3.dub.plan import RateScale

    ext = None if bundle is None else bundle.rate_extension
    if ext is None:
        return {}
    edges = tuple(_dub_rate_edges(config))
    if len(ext.row_edges) != len(edges) or not np.allclose(ext.row_edges, edges, rtol=0, atol=1e-6):
        logger.warning("steering bundle rate_extension was measured on speaking-rate edges %s, the checkpoint has %s: "
                       "dubs plan in the row buckets only", list(ext.row_edges), list(edges))
        return {}
    source = {lang: {b: d for b, d in doses.items() if ("slower" if b < 0 else "faster") in ext.reference_sides}
              for lang, doses in ext.doses.items()}
    return {mode: {lang: RateScale.extended(edges, ext.below, ext.above, doses) for lang, doses in by_lang.items() if doses}
            for mode, by_lang in (("studio", ext.doses), ("source", source))}


def _dub_runtime():
    """The server's DubRuntime (created on first use; its alignment worker starts with the first call that needs it)."""
    global _DUB_RUNTIME
    if _DUB_RUNTIME is None:
        from zonos3.dub.runtime import DubModel, DubRuntime

        config = get_global_state().config
        spec = _condition_spec(config)
        bundle = _steer_bundle(config)
        _DUB_RUNTIME = DubRuntime(
            DubModel(
                name=Path(str(config.model_path)).name,
                rate_edges=tuple(_dub_rate_edges(config)),
                n_buckets=int(spec["n_buckets"]) if spec else 0,
                laci_languages=frozenset(getattr(config, "laci_heads", None) or {}),
                studio_voices=_model_supports_speaker_embedding(config),
                source_voice=_model_uses_speaker_reference_prefix(config),
                raw_controls=frozenset(
                    did for did, d in (bundle.directions.items() if bundle is not None else ()) if d.space == RAW_SPACE
                ),
                phoneme_groups=_phoneme_capabilities(config)["phoneme_tokens"] is not None,
                rate_scales=_dub_rate_scales(config, bundle),
            ),
            _dub_synthesize,
            _dub_encode_voice,
            _ASR,
            _FUSEALIGN,
        )
    return _DUB_RUNTIME


def _dub_ready():
    config = get_global_state().config
    if not _dub_available(config):
        raise HTTPException(
            status_code=400,
            detail="Dubbing is not enabled on this server: it needs --laci-heads (DUB=1 in servers/sgl/scripts/serve.sh) and a "
                   "checkpoint with a speaking-rate condition row",
        )
    return _dub_runtime()


async def _dub_call(awaitable):
    """Run a DubRuntime call, mapping its errors to HTTP statuses."""
    from zonos3.dub.runtime import AlignWorkerError, UnknownId

    try:
        return await awaitable
    except UnknownId:
        raise HTTPException(status_code=404, detail="unknown ref_id/dub_id; upload or generate again") from None
    except AlignWorkerError as exc:
        raise HTTPException(status_code=503, detail=f"alignment worker: {exc}") from exc
    except WorkerError as exc:  # the ASR / FuseAlign worker is unavailable (the message says which)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _dub_take_request(job: dict) -> GenerateRequest:
    """The /v2/generate request of one dub take: the release sampling defaults, the take's speaking-rate bucket in the
    condition rows (the others unknown; a tempo-steered bucket leaves the rate row unknown too, see
    _dub_take_rate_steering), LACI, and its voice: the reference clone (raw reference prefix codes) or a studio voice
    (speaker embedding) with the take's steering controls."""
    voice, bucket = job["voice"], int(job["bucket"])
    scale = job.get("rate_scale")
    row = None if scale is not None and bucket in scale.doses else bucket
    fields = dict(
        text=job["text"], language=job["language"], seed=int(job["seed"]), max_tokens=int(job["max_tokens"]),
        stream=False, laci=True, speaker_cfg_scale=1.0, condition_buckets={"speaking_rate": row},
        controls=job["controls"],
    )
    if voice["mode"] == "source":
        fields.update(speaker_mode="reference", speaker_reference_codes=job["ref_codes"])
    else:
        fields.update(speaker_mode="embedding", speaker_embedding_id=voice["speaker_embedding_id"])
    return GenerateRequest(**fields)


def _dub_take_rate_steering(config: ServerArgs, msg: TTSTokenizeMsg, job: dict) -> None:
    """Put a dub take on its language's extended speaking-rate scale (job["rate_scale"], README "Extended
    speaking-rate buckets"): in a steered bucket the take's steering delta gains the bucket's tempo dose (on top of the
    take's own controls). The take carries the scale, its bucket on it and its delta in every bucket: LACI reports the
    bucket on the scale (a steered one's row id is only "unknown"), and a rate-adjustable take's retakes switch to
    another bucket's delta (row bucket <-> rate row unknown + tempo)."""
    scale = job.get("rate_scale")
    if scale is None:
        return
    bundle = _steer_bundle(config)
    sources, hidden = _steering_sources(config), int(config.model_config.hidden_size)
    base = msg.speaker_emotion_delta

    def delta(b: int) -> torch.Tensor | None:
        if b not in scale.doses:
            return base
        tempo = resolve_controls(bundle, {bundle.rate_extension.direction: scale.doses[b]}, None, text=msg.text,
                                 speaker_embedding=None, sources=sources, hidden_size=hidden).hidden_delta
        return tempo if base is None else base + tempo

    bucket = int(job["bucket"])
    deltas = [delta(b) for b in range(scale.first, scale.last + 1)]
    msg.speaker_emotion_delta = deltas[bucket - scale.first]
    msg.laci_rate_edges, msg.laci_rate_first, msg.laci_rate_bucket = list(scale.edges), scale.first, bucket
    msg.laci_rate_deltas = deltas


async def _dub_synthesize(jobs: list[dict], session_id: str | None) -> list[tuple[np.ndarray | None, dict]]:
    """Every dub take through the scheduler, at most twice the running-request cap in flight so studio requests
    interleave. All requests are validated before the first is queued (HTTP 400 for a bad voice / control / text)."""
    global _DUB_TAKE_SLOTS
    state = get_global_state()
    if _DUB_TAKE_SLOTS is None:
        _DUB_TAKE_SLOTS = asyncio.Semaphore(max(1, 2 * int(getattr(state.config, "max_running_req", 1) or 1)))
    gain = _output_gain(state.config, None)
    msgs = []
    try:
        for job in jobs:
            msg = await _tts_generate_msg(
                state, _dub_take_request(job), session_id,
                laci_target_frames=float(job["target_frames"]), laci_adjust_rate=bool(job["adjust_rate"]),
            )
            msgs.append(msg)
            _dub_take_rate_steering(state.config, msg, job)
    except BaseException:
        for msg in msgs:
            state.ack_map.pop(msg.uid, None)
            state.event_map.pop(msg.uid, None)
            state.start_time_map.pop(msg.uid, None)
        raise

    async def run(msg: TTSTokenizeMsg):
        async with _DUB_TAKE_SLOTS:
            await state.send_one(msg)
            data, record = await state.wait_for_tts_reply(msg.uid, gain=gain)
        if record is None:
            raise RuntimeError(f"dub take uid={msg.uid}: the scheduler sent no LACI record")
        audio = np.frombuffer(data, dtype=np.float32).copy() if data else None
        return audio, record

    return list(await asyncio.gather(*(run(msg) for msg in msgs)))


async def _dub_encode_voice(audio: np.ndarray, sample_rate: int) -> list[list[int]]:
    """Raw reference prefix codes of the source voice's prompt clip (the tokenizer's codec encode)."""
    from zonos3.dub.plan import VOICE_PROMPT_MAX_S

    buf = io.BytesIO()
    sf.write(buf, audio, int(sample_rate), format="WAV", subtype="FLOAT")
    reply = await _backend_reply(
        get_global_state(), lambda uid: TTSEncodeAudioMsg(uid=uid, audio=buf.getvalue(), max_seconds=VOICE_PROMPT_MAX_S)
    )
    if reply.error is not None:
        raise ValueError(reply.error)
    return reply.codes.to(torch.int64).tolist()


async def _dub_upload(request: Request, name: str, directory: str) -> Path:
    """Stream a request body (a file named ``name``) into ``directory``; 413 past the dub's upload cap."""
    from zonos3.dub.plan import MAX_VIDEO_BYTES

    declared = request.headers.get("content-length")
    if declared is not None and int(declared) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")
    suffix = Path(name).suffix.lower()
    path = Path(directory) / f"input{suffix if suffix in _DUB_VIDEO_SUFFIXES | {'.wav', '.flac', '.mp3', '.ogg'} else '.bin'}"
    size = 0
    with open(path, "wb") as fh:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                raise HTTPException(status_code=413, detail="upload too large")
            fh.write(chunk)
    return path


@app.get("/v2/dub/capabilities")
async def dub_capabilities():
    """Target languages, rate buckets, lip-sync, voice modes and limits of the dub (zonos3.dub.runtime)."""
    config = get_global_state().config
    return {**(await asyncio.to_thread(_dub_runtime().capabilities)), "enabled": _dub_available(config)}


@app.post("/v2/dub/ref")
async def dub_ref(request: Request, name: str = "audio"):
    """A dub reference: raw audio bytes (any sound file), or a video whose audio track is taken."""
    runtime = _dub_ready()
    with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
        path = await _dub_upload(request, name, tmp)
        return await _dub_call(asyncio.to_thread(runtime.add_ref, str(path)))


@app.post("/v2/dub/lips/{ref_id}")
async def dub_lips(ref_id: str, request: Request, name: str = "video"):
    """Face track + lip embeddings of the reference's video, for generate / regenerate with lipsync (AV-sync)."""
    runtime = _dub_ready()

    async def run():
        runtime.refs.get(ref_id)  # 404 before reading the body
        with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
            video = await _dub_upload(request, name, tmp)
            return await asyncio.to_thread(runtime.prepare_lips, ref_id, str(video))

    return await _dub_call(run())


@app.post("/v2/dub/transcribe")
async def dub_transcribe(req: DubTranscribeRequest):
    runtime = _dub_ready()
    return await _dub_call(runtime.transcribe(req.ref_id, req.ref_lang))


@app.post("/v2/dub/analyze")
async def dub_analyze(req: DubAnalyzeRequest):
    runtime = _dub_ready()
    return await _dub_call(runtime.analyze(req.model_dump()))


@app.post("/v2/dub/generate")
async def dub_generate(
    req: DubGenerateRequest,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    runtime = _dub_ready()
    return await _dub_call(runtime.generate(req.model_dump(), x_tts_session_id))


@app.post("/v2/dub/regenerate")
async def dub_regenerate(
    req: DubRegenerateRequest,
    x_tts_session_id: str | None = Header(default=None, alias="X-TTS-Session-ID"),
):
    runtime = _dub_ready()
    return await _dub_call(runtime.regenerate(req.model_dump(), x_tts_session_id))


@app.post("/v2/dub/mux/{dub_id}")
async def dub_mux(dub_id: str, request: Request, name: str = "video"):
    """The video with the dub as its audio track (raw video bytes in, MP4 out)."""
    runtime = _dub_ready()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "video"

    async def run():
        runtime.dubs.get(dub_id)  # 404 before reading the body
        with tempfile.TemporaryDirectory(prefix="dub-") as tmp:
            video, out = await _dub_upload(request, name, tmp), Path(tmp) / "out.mp4"
            await asyncio.to_thread(runtime.mux, dub_id, str(video), str(out))
            return out.read_bytes()

    data = await _dub_call(run())
    return Response(data, media_type="video/mp4",
                    headers={"Content-Disposition": f'attachment; filename="{stem}_dub.mp4"'})


def _studio_dir() -> Path:
    """--studio-dir: the production build of Zonos Voice Studio (studio/dist)."""
    state = _GLOBAL_STATE
    return Path(state.config.studio_dir if state is not None else ServerArgs.studio_dir).resolve()


# Registered last, so every API route matches first: the studio serving rule. GET|HEAD / is the studio's
# index.html; GET|HEAD /<path> a file inside the studio dir (never under /v1/ or /v2/); anything else is a 404.
# Pages (index.html and every other *.html) are no-store, so a rebuild's new hashed assets are picked up at once;
# every other file is immutable.
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
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Not Found")
    cache = "no-store, max-age=0" if target.name.endswith(".html") else "public, max-age=31536000, immutable"
    return FileResponse(target, headers={"Cache-Control": cache})


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

    # TTS settings
    temperature = 0.75
    topk = 80
    repetition_window = 96
    repetition_penalty = 2.4
    repetition_codebooks = 6
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
                    _save_wav(last_audio, filename)
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
                    sd.play(audio, samplerate=44100)
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
                duration = len(last_audio) / (44100 * 4)  # float32 = 4 bytes
                print(f" Done! ({duration:.2f}s)")

                # Auto-play if sounddevice is available
                if can_play_audio:
                    audio = np.frombuffer(last_audio, dtype=np.float32)
                    sd.play(audio, samplerate=44100)
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


def _save_wav(audio_bytes: bytes, path: str, sample_rate: int = 44100) -> None:
    """Save float32 PCM bytes as a 16-bit WAV file."""
    Path(path).write_bytes(_pcm16_wav(audio_bytes, sample_rate))


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
