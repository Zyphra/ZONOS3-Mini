"""Quality-token conditioning shared by the HTTP server (``api_server``) and the offline
:class:`~zonos3_sgl.tts.llm.TTSLLM`.

Legacy quality-conditioned checkpoints read one quality token per configured feature; every request gets the
former request default, ``_DEFAULT_QUALITY_BUCKETS`` (trailing silence 0.25-0.5 s). The request's max_tokens budget is
``zonos3.tts.plan.max_tokens_budget``.
"""

from __future__ import annotations

from collections.abc import Sequence

_DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3}


def default_quality_buckets(features: Sequence[str], bucket_counts: Sequence[int]) -> list[int | None] | None:
    """The quality buckets of every request in the checkpoint's feature order (None per feature without a default);
    None when the checkpoint has no quality tokens."""
    if not features or sum(int(count) for count in bucket_counts) <= 0:
        return None
    return [_DEFAULT_QUALITY_BUCKETS.get(str(feature)) for feature in features]
