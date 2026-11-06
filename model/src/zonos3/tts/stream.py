"""Streaming decode of the generated rows: one request's TTSVocoderManager (zonos3_sgl.tokenizer.vocoder).

Rows arrive delayed (codebook j of frame f in row f + j). The buffer opens with the silence-prefill rows (decoded,
never emitted). ``whole`` mode decodes every pre-EOS frame once at end of stream, re-decoding the last
``overlap_frames`` already-counted rows as left context; ``prefix`` mode re-decodes the growing prefix (the causal
decoder) every ``chunk`` frames (the first chunk after ``first_chunk``, for an earlier first audio) behind a hold-back,
with at most ``context`` frames of left context, and the exact whole decode supplies the tail at EOS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from zonos3.tts.delay import undelay_window

PREFIX_ERROR = (
    "prefix stream settings: chunk_frames >= 1, first_chunk_frames >= 1, hold_back_frames >= 0, "
    "context_frames >= chunk_frames"
)


@dataclass(frozen=True)
class StreamSettings:
    """Prefix streaming: a chunk once ``chunk`` new frames are decodable behind ``hold_back`` frames (``first_chunk``
    for the stream's first), each decoding the prefix with at most ``context`` frames of left context. The defaults
    are the arena's measured settings (1 s chunks, 3 frames held back, 30 s of left context, -48 dB vs whole)."""

    chunk: int = 15
    first_chunk: int = 4
    hold_back: int = 3
    context: int = 450

    def __post_init__(self):
        if self.chunk < 1 or self.first_chunk < 1 or self.hold_back < 0 or self.context < self.chunk:
            raise ValueError(PREFIX_ERROR)

    @classmethod
    def from_env(cls) -> StreamSettings:
        """ZONOS3_PREFIX_STREAM_{CHUNK,FIRST_CHUNK,HOLD_BACK,CONTEXT}_FRAMES over the defaults; ValueError if invalid."""
        default = cls()
        env = os.environ.get
        try:
            return cls(
                chunk=int(env("ZONOS3_PREFIX_STREAM_CHUNK_FRAMES", default.chunk)),
                first_chunk=int(env("ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES", default.first_chunk)),
                hold_back=int(env("ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES", default.hold_back)),
                context=int(env("ZONOS3_PREFIX_STREAM_CONTEXT_FRAMES", default.context)),
            )
        except (TypeError, ValueError):
            raise ValueError(PREFIX_ERROR) from None


def prefix_window(complete: int, emitted: int, prefill: int, s: StreamSettings) -> tuple[int, int] | None:
    """The next prefix chunk: ``(start, decodable)`` = decode output frames [start, complete) and emit the samples of
    frames [emitted, decodable), or None while too few frames are decodable.

    decodable = complete frames minus the held-back tail: the held-back frames are decoded as right context but not
    emitted (the last decoded frame of the iSTFT head sits ~-22 dB from the whole decode; one frame of right context
    already reaches the bf16 floor near -48 dB). The first chunk (nothing past the ``prefill`` silence rows emitted
    yet) waits for ``first_chunk`` frames, every later one for ``chunk``. Once the history is longer than the context
    window only the last ``context`` frames of left context are re-decoded, so the cost per chunk stays bounded."""
    decodable = complete - s.hold_back
    need = s.first_chunk if emitted == prefill else s.chunk
    if decodable - emitted < need:
        return None
    start = max(0, emitted - s.context) if decodable > s.context + s.chunk else 0
    return start, decodable


def _host(x) -> np.ndarray:
    """A numpy view of rows or codes (torch tensors on any device, arrays, lists)."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class StreamDecoder:
    """One request's rows as they come -> float32 PCM. ``decode(codes)`` runs the codec on undelayed int64 codes
    [frames, n_codebooks] (the reference tail prepended at the stream start; its samples are dropped here)."""

    def __init__(
        self,
        decode: Callable[[np.ndarray], np.ndarray],
        *,
        n_codebooks: int,
        audio_pad_id: int,
        hop: int,
        prefill_rows=None,
        context_codes=None,
        mode: str = "whole",
        overlap_frames: int = 4,
        settings: StreamSettings = StreamSettings(),
    ):
        if mode not in ("whole", "prefix"):
            raise ValueError(f"stream decode mode must be whole or prefix, got {mode!r}")
        self.decode = decode
        self.n_codebooks, self.pad, self.eoa, self.hop = n_codebooks, audio_pad_id, audio_pad_id - 1, hop
        self.mode, self.overlap, self.settings = mode, overlap_frames, settings
        self.buffer: list[list[int]] = (
            []
            if prefill_rows is None
            else [[int(v) for v in row[:n_codebooks]] for row in _host(prefill_rows).tolist()]
        )
        self.prefill = len(self.buffer)
        self.context = None
        if context_codes is not None and len(context_codes):
            self.context = _host(context_codes).astype(np.int64).reshape(-1, n_codebooks)
        self.decoded = self.prefill  # whole mode: frames already counted
        self.emitted = self.prefill  # prefix mode: frames already streamed
        self.eos: int | None = None
        self.done = False

    def _decode(self, start: int, frames_end: int) -> np.ndarray:
        """Output frames [start, frames_end) from the delayed rows, with the reference tail as left context when the
        slice starts at the stream start (its samples dropped)."""
        codes = undelay_window(self.buffer, start, frames_end, self.n_codebooks, self.pad).numpy()
        if start == 0 and self.context is not None:
            audio = self.decode(np.concatenate([self.context, codes], axis=0))
            return np.asarray(audio, dtype=np.float32)[len(self.context) * self.hop :]
        return np.asarray(self.decode(codes), dtype=np.float32)

    def _target(self) -> int:
        target = len(self.buffer) - (self.n_codebooks - 1)
        return target if self.eos is None else min(target, self.eos)

    def push(self, row, eos_frame: int | None, finished: bool = False) -> np.ndarray:
        """One generated row (``eos_frame``: the scheduler's aligned EOS frame in generated rows); the float32 PCM it
        completes (maybe none). ``finished`` marks the stream's last row (the same as pushing it and calling
        ``finish``)."""
        if self.done:
            raise RuntimeError("stream already finished")
        head = row[: self.n_codebooks]
        codes = [int(v) for v in (head.tolist() if hasattr(head, "tolist") else head)]
        self.buffer.append(codes)
        candidates = []
        if eos_frame is not None:
            candidates.append(max(0, int(eos_frame)) + self.prefill)
        cols = [i for i, tok in enumerate(codes) if tok == self.eoa]
        if cols:
            candidates.append(max(0, len(self.buffer) - 1 - max(cols)))
        if candidates:
            observed = min(candidates)
            self.eos = observed if self.eos is None else min(self.eos, observed)
        if finished:
            return self.finish()
        return self._prefix_chunk() if self.mode == "prefix" else np.zeros(0, dtype=np.float32)

    def finish(self) -> np.ndarray:
        """End of stream: the rest of the audio (the whole decode, or the prefix stream's exact tail)."""
        if self.done:
            raise RuntimeError("stream already finished")
        self.done = True
        return self._prefix_final() if self.mode == "prefix" else self._whole_final()

    def _whole_final(self) -> np.ndarray:
        """Every complete pre-EOS frame in one decode; the last ``overlap`` counted rows (the silence prefill) are
        re-decoded and emitted with the rest, as TTSVocoderManager._decode_final does."""
        target = self._target()
        if target <= self.decoded:
            return np.zeros(0, dtype=np.float32)
        start = self.decoded - min(self.overlap, self.decoded)
        audio = self._decode(start, target)
        self.decoded = target
        return audio

    def _prefix_chunk(self) -> np.ndarray:
        complete = self._target()
        window = prefix_window(complete, self.emitted, self.prefill, self.settings)
        if window is None:
            return np.zeros(0, dtype=np.float32)
        start, decodable = window
        audio = self._decode(start, complete)
        chunk = audio[(self.emitted - start) * self.hop : (decodable - start) * self.hop]
        self.emitted = decodable
        return chunk

    def _prefix_final(self) -> np.ndarray:
        target = self._target()
        if target <= 0:
            return np.zeros(0, dtype=np.float32)
        emitted = min(self.emitted, target)
        return self._decode(0, target)[emitted * self.hop :]
