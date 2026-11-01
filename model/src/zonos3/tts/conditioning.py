"""Quality-token and max-token conditioning shared by the HTTP server (``api_server``) and the offline
:class:`~zonos3_sgl.tts.llm.TTSLLM`.

Legacy quality-conditioned checkpoints read one quality token per configured feature; every request gets the
former request default, ``_DEFAULT_QUALITY_BUCKETS`` (trailing silence 0.25-0.5 s). The max-token helpers read only
``config.max_seq_len`` (via attribute access), so any object exposing it works -- there is no FastAPI / server
dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3}


def default_quality_buckets(features: Sequence[str], bucket_counts: Sequence[int]) -> list[int | None] | None:
    """The quality buckets of every request in the checkpoint's feature order (None per feature without a default);
    None when the checkpoint has no quality tokens."""
    if not features or sum(int(count) for count in bucket_counts) <= 0:
        return None
    return [_DEFAULT_QUALITY_BUCKETS.get(str(feature)) for feature in features]


def _model_tts_max_tokens(config: Any) -> int:
    return max(1, int(config.max_seq_len))


def _resolve_tts_max_tokens(config: Any, requested: int | None) -> int:
    model_max = _model_tts_max_tokens(config)
    if requested is None:
        return model_max
    requested = int(requested)
    if requested <= 0:
        raise ValueError("max_tokens must be positive.")
    return min(requested, model_max)
