"""Per-request streaming decode of the generated rows: one request's TTSVocoderManager (zonos3_sgl.tokenizer.vocoder).

Rows arrive delayed (codebook j of frame f in row f + j). The buffer opens with the silence-prefill rows (decoded,
never emitted). ``whole`` mode decodes every pre-EOS frame once at end of stream, re-decoding the last
``overlap_frames`` already-counted rows as left context; ``prefix`` mode re-decodes the growing prefix (the causal
decoder) every ``chunk_frames`` frames (the first chunk after ``first_chunk_frames``, for an earlier first audio) behind
a hold-back, with at most ``context_frames`` of left context, and the exact whole decode supplies the tail at EOS.
``decode`` runs the codec (on the scheduler's codec thread in the server).
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import torch
from zonos3.tts.delay import shear_up

PREFIX_ERROR = ("prefix stream settings: chunk_frames >= 1, first_chunk_frames >= 1, hold_back_frames >= 0, "
                "context_frames >= chunk_frames")


def prefix_settings() -> tuple[int, int, int, int]:
    """(chunk, first_chunk, hold_back, context) frames from ZONOS3_PREFIX_STREAM_* (defaults 15, 4, 3, 450); ValueError
    if invalid."""
    env = os.environ.get
    try:
        values = (int(env("ZONOS3_PREFIX_STREAM_CHUNK_FRAMES", "15")),
                  int(env("ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES", "4")),
                  int(env("ZONOS3_PREFIX_STREAM_HOLD_BACK_FRAMES", "3")),
                  int(env("ZONOS3_PREFIX_STREAM_CONTEXT_FRAMES", "450")))
    except ValueError:
        raise ValueError(PREFIX_ERROR) from None
    chunk, first, hold, context = values
    if chunk < 1 or first < 1 or hold < 0 or context < chunk:
        raise ValueError(PREFIX_ERROR)
    return values


class StreamDecoder:
    def __init__(self, decode: Callable[[np.ndarray], np.ndarray], *, n_codebooks: int = 10, audio_pad_id: int = 4097,
                 hop: int = 2940, prefill_rows=None, context_codes=None, mode: str = "whole",
                 overlap_frames: int = 4, chunk_frames: int = 15, first_chunk_frames: int = 4, hold_back_frames: int = 3,
                 context_frames: int = 450):
        if mode not in ("whole", "prefix"):
            raise ValueError(f"stream decode mode must be whole or prefix, got {mode!r}")
        if chunk_frames < 1 or first_chunk_frames < 1 or hold_back_frames < 0 or context_frames < chunk_frames:
            raise ValueError(PREFIX_ERROR)
        self.decode = decode
        self.n_codebooks, self.pad, self.eoa, self.hop = n_codebooks, audio_pad_id, audio_pad_id - 1, hop
        self.mode = mode
        self.overlap, self.chunk, self.hold_back, self.context_frames = overlap_frames, chunk_frames, hold_back_frames, context_frames
        self.first_chunk = first_chunk_frames
        self.buffer: list[list[int]] = [] if prefill_rows is None else [
            [int(v) for v in row[:n_codebooks]] for row in np.asarray(prefill_rows).tolist()]
        self.prefill = len(self.buffer)
        self.context = None
        if context_codes is not None and len(context_codes):
            self.context = np.asarray(context_codes, dtype=np.int64).reshape(-1, n_codebooks)
        self.decoded = self.prefill  # whole mode: frames already counted
        self.emitted = self.prefill  # prefix mode: frames already streamed
        self.eos: int | None = None
        self.done = False

    # ------------------------------------------------------------------------------------------------- decode
    def _decode(self, start: int, frames_end: int) -> np.ndarray:
        """Output frames [start, frames_end) from the delayed rows, with the reference tail as left context when the
        slice starts at the stream start (its samples dropped)."""
        raw_end = min(frames_end + self.n_codebooks - 1, len(self.buffer))
        codes = shear_up(torch.tensor(self.buffer[start:raw_end], dtype=torch.int64), self.pad)[: frames_end - start]
        codes = codes.numpy()
        if start == 0 and self.context is not None:
            audio = self.decode(np.concatenate([self.context, codes], axis=0))
            return np.asarray(audio, dtype=np.float32)[len(self.context) * self.hop:]
        return np.asarray(self.decode(codes), dtype=np.float32)

    def _target(self) -> int:
        target = len(self.buffer) - (self.n_codebooks - 1)
        return target if self.eos is None else min(target, self.eos)

    def push(self, row, eos_frame: int | None, finished: bool = False) -> bytes:
        """One generated row (``eos_frame``: the scheduler's aligned EOS frame in generated rows); float32 PCM bytes.
        ``finished`` marks the stream's last row (the same as pushing it and calling ``finish``)."""
        if self.done:
            raise RuntimeError("stream already finished")
        codes = [int(v) for v in np.asarray(row)[: self.n_codebooks].tolist()]
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
        return self._prefix_chunk() if self.mode == "prefix" else b""

    def finish(self) -> bytes:
        """End of stream: the rest of the audio (the whole decode, or the prefix stream's exact tail)."""
        if self.done:
            raise RuntimeError("stream already finished")
        self.done = True
        return self._prefix_final() if self.mode == "prefix" else self._whole_final()

    def _whole_final(self) -> bytes:
        """Every complete pre-EOS frame in one decode; the last ``overlap`` counted rows (the silence prefill) are
        re-decoded and emitted with the rest, as TTSVocoderManager._decode_final and synth.decode_rows do."""
        target = self._target()
        if target <= self.decoded:
            return b""
        start = self.decoded - min(self.overlap, self.decoded)
        audio = self._decode(start, target)
        self.decoded = target
        return audio.astype(np.float32).tobytes()

    def _prefix_chunk(self) -> bytes:
        complete = self._target()
        decodable = complete - self.hold_back
        # the first chunk (nothing past the silence prefill streamed yet) waits for first_chunk frames
        need = self.first_chunk if self.emitted == self.prefill else self.chunk
        if decodable - self.emitted < need:
            return b""
        windowed = decodable > self.context_frames + self.chunk
        start = max(0, self.emitted - self.context_frames) if windowed else 0
        audio = self._decode(start, complete)
        chunk = audio[(self.emitted - start) * self.hop:(decodable - start) * self.hop]
        self.emitted = decodable
        return chunk.astype(np.float32).tobytes()

    def _prefix_final(self) -> bytes:
        target = self._target()
        emitted = min(self.emitted, max(target, 0))
        if target <= 0:
            return b""
        whole = self._decode(0, target)
        return whole[emitted * self.hop:].astype(np.float32).tobytes()
