"""The reference sampler (``zonos3.reference.synth.sample_row`` -> ``zonos3.tts.sampler.sample_tts``) on the MLX device.

``torch.multinomial(p, 1, generator=g)`` over a 2-D ``[n_codebooks, width]`` tensor draws one exponential per element,
``q = torch.empty_like(p).exponential_(generator=g)``, and returns ``argmax(p / q)``. The host draws ``q``
(``noise_for``: independent of the logits, the reference's random stream) and the device takes the argmax, so a row
equals the reference sampler's up to float rounding (which may flip a near-tie draw) and the generator ends in the same
state. The scheduler can then queue the next forward on the lazy sample before reading it back.
"""

from __future__ import annotations

import functools
import math

import mlx.core as mx
import numpy as np
import torch
from zonos3.tts.sampler import ClampedSampling, active_repetition_window


def noise_for(generator: torch.Generator, n_codebooks: int, width: int) -> np.ndarray:
    """The exponential draws of one reference ``torch.multinomial`` call over ``[n_codebooks, width]`` probabilities."""
    return torch.empty((n_codebooks, width), dtype=torch.float32).exponential_(generator=generator).numpy()


def guided_logits_mx(conditional: mx.array, unconditional: mx.array, scale: float) -> mx.array:
    """``zonos3.speaker_cfg.guided_logits`` on the device: the raw logits mixed in FP32."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Speaker CFG scale must be finite and nonnegative")
    if conditional.shape != unconditional.shape:
        raise ValueError("CFG branches must have identical logit shapes")
    if scale == 1.0:
        return conditional.astype(mx.float32)
    uncond = unconditional.astype(mx.float32)
    if scale == 0.0:
        return uncond
    return uncond + scale * (conditional.astype(mx.float32) - uncond)


def repetition_ids(
    s: ClampedSampling, history, available: int, *, n_codebooks: int, codebook_size: int
) -> np.ndarray | None:
    """The repetition-penalty ids of ``sample_row``: the last active-window rows of ``history`` per codebook
    ``[n_codebooks, active]``, -1 where the id is no ordinary code or the codebook is past ``repetition_codebooks``;
    None without a penalty."""
    active = active_repetition_window(s, available)
    if active <= 0:
        return None
    recent = np.asarray(history[len(history) - active :, :n_codebooks], dtype=np.int64).T
    valid = (recent >= 0) & (recent < codebook_size)
    valid[s.repetition_codebooks :] = False
    return np.where(valid, recent, -1)


@functools.lru_cache(maxsize=8)
def _invalid_mask(n_cb: int, vocab: int, codebook_size: int, sizes: tuple, audio_pad_id: int, stop_id: int) -> mx.array:
    """``sample_tts``'s never-sampled ids ``[n_cb, vocab]``: padding, special ids other than the stop id, and ordinary
    ids past each codebook's size (``sizes``; empty = the shared codebook size)."""
    ids = np.arange(vocab)
    invalid = np.broadcast_to((ids == audio_pad_id) | ((ids >= codebook_size) & (ids != stop_id)), (n_cb, vocab))
    if sizes:
        invalid = invalid | ((ids >= np.asarray(sizes)[:, None]) & (ids < codebook_size))
    return mx.array(invalid)


def sample_row_mx(
    logits: mx.array,
    noise: mx.array,
    s: ClampedSampling,
    repetition: np.ndarray | None,
    *,
    codebook_size: int,
    codebook_sizes,
    audio_pad_id: int,
    stop_id: int,
    stop_blocked: np.ndarray | None,
    total_generated: int,
    has_audio_prefix: bool,
    text_vocab: int,
    stop_bias: float = 0.0,
) -> mx.array:
    """One sampled row ``[n_codebooks + 1]`` int32 (lazy): the reference ``sample_row`` on ``logits``
    ``[n_codebooks, V]`` with the draws ``noise`` (``noise_for``, width ``max(stop_id, audio_pad_id) + 1``), the cold
    start delay applied and the text placeholder appended. ``repetition``: ``repetition_ids``; ``stop_blocked``: the
    codebooks whose stop id is masked (min_frames), None for none; ``stop_bias``: added to the stop id's logit in every
    codebook before the filters (LACI's stop boost)."""
    n_cb, vocab = logits.shape
    invalid = _invalid_mask(n_cb, vocab, codebook_size, tuple(codebook_sizes or ()), audio_pad_id, stop_id)
    if stop_blocked is not None:
        blocked = np.zeros((n_cb, vocab), dtype=bool)
        blocked[:, stop_id] = np.asarray(stop_blocked, dtype=bool)
        invalid = invalid | mx.array(blocked)
    x = mx.where(invalid, -mx.inf, logits.astype(mx.float32))
    if stop_bias:
        x = mx.where(mx.arange(vocab)[None, :] == stop_id, x + float(stop_bias), x)
    if repetition is not None:
        repeated = np.zeros((n_cb, vocab), dtype=bool)
        rows, cols = np.nonzero(repetition >= 0)
        repeated[rows, repetition[rows, cols]] = True
        penalty = s.repetition_penalty
        x = mx.where(mx.array(repeated), mx.where(x > 0, x / penalty, x * penalty), x)
    x = x / s.temperature  # clamp_sampling keeps it >= 1e-6
    if 0 < s.topk < vocab:
        kth = mx.min(mx.topk(x, s.topk, axis=-1), axis=-1, keepdims=True)
        x = mx.where(x < kth, -mx.inf, x)
    probs = mx.softmax(x, axis=-1)
    if 0.0 < s.top_p < 1.0:
        order = mx.argsort(-probs, axis=-1)
        ranked = mx.take_along_axis(probs, order, axis=-1)
        ranked = mx.where(mx.cumsum(ranked, axis=-1) - ranked > s.top_p, 0.0, ranked)
        probs = mx.put_along_axis(mx.zeros_like(probs), order, ranked, axis=-1)
        probs = probs / mx.maximum(probs.sum(axis=-1, keepdims=True), 1e-8)
    if s.min_p > 0.0:
        probs = mx.where(probs < s.min_p * probs.max(axis=-1, keepdims=True), 0.0, probs)
        probs = probs / mx.maximum(probs.sum(axis=-1, keepdims=True), 1e-8)
    # a row the filters emptied falls back to greedy over the filtered logits
    greedy = mx.arange(vocab)[None, :] == mx.argmax(x, axis=-1, keepdims=True)
    probs = mx.where(probs.sum(axis=-1, keepdims=True) <= 0, greedy.astype(mx.float32), probs)
    width = max(int(stop_id), int(audio_pad_id)) + 1
    token = mx.argmax(probs[:, :width] / noise, axis=-1).astype(mx.int32)
    if not has_audio_prefix and total_generated < n_cb - 1:
        token = mx.where(mx.arange(n_cb) > total_generated, int(audio_pad_id), token)
    return mx.concatenate([token, mx.array([int(text_vocab)], dtype=mx.int32)])
