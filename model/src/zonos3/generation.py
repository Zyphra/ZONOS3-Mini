"""The generation registry (``registries/generation.json``): the request defaults and limits of every server.

``SAMPLING`` holds the v2 sampling defaults (``servers/api/openapi.yaml`` repeats them), the rest the request limits the
request rules (``zonos3.tts.plan``) and the servers' upload paths enforce. The ggml server embeds the same file at
build time; the in-browser server imports it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

REGISTRY = Path(__file__).resolve().parent / "registries" / "generation.json"

_data = json.loads(REGISTRY.read_text(encoding="utf-8"))
SAMPLE_RATE: int = int(_data["sample_rate"])
# temperature, topk, top_p, min_p, repetition_penalty, repetition_window, repetition_codebooks
SAMPLING = MappingProxyType(dict(_data["sampling"]))
TAG_CFG_MIN_SCALE: float = float(_data["tag_cfg_scale"]["min"])
TAG_CFG_MAX_SCALE: float = float(_data["tag_cfg_scale"]["max"])
SILENCE_PREFILL_DEFAULT_FRAMES: int = int(_data["silence_prefill_frames"]["default"])
SILENCE_PREFILL_MAX_FRAMES: int = int(_data["silence_prefill_frames"]["max"])
DECODE_CONTEXT_FRAMES: int = int(_data["decode_context_frames"])
REFERENCE_MAX_FRAMES: int = int(_data["reference_max_frames"])
MAX_FRAMES: int = int(_data["max_frames"])
REFERENCE_AUDIO_MAX_SECONDS: float = float(_data["audio_max_seconds"]["reference"])
UPLOAD_MAX_SECONDS: float = float(_data["audio_max_seconds"]["upload"])
EDIT_MAX_SECONDS: float = float(_data["audio_max_seconds"]["edit"])
del _data
