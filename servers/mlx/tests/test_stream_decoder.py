"""Streaming decode of the generated rows (zonos3_mlx.vocoder.StreamDecoder), ported from
servers/sgl/tests/test_tts_prefix_stream.py: a causal fake decoder (audio of frame f depends on frames [f - memory, f])
makes prefix re-decodes reproduce the whole decode exactly, so every property is checked bit for bit."""

from __future__ import annotations

import numpy as np
import pytest
from zonos3_mlx.vocoder import StreamDecoder, prefix_settings

N_CB, HOP, PAD, EOA = 10, 2940, 4097, 4096


class CausalFakeDecoder:
    def __init__(self, memory: int = 3):
        self.memory = memory
        self.calls: list[int] = []

    def __call__(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.float64)
        assert codes.ndim == 2 and codes.shape[1] == N_CB
        frames = len(codes)
        self.calls.append(frames)
        out = np.zeros(frames * HOP, dtype=np.float64)
        ramp = np.linspace(0.5, 1.0, HOP)
        for f in range(frames):
            out[f * HOP:(f + 1) * HOP] = codes[max(0, f - self.memory):f + 1].sum() * 1e-5 * ramp
        return out.astype(np.float32)


def delayed_rows(frames: np.ndarray) -> np.ndarray:
    n, cb = frames.shape
    rows = np.full((n + cb - 1, cb), PAD, dtype=np.int64)
    for j in range(cb):
        rows[j:j + n, j] = frames[:, j]
    return rows


def make_frames(n: int, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 4096, size=(n, N_CB)).astype(np.int64)


def run(decoder: StreamDecoder, rows, eos_frame=None) -> list[bytes]:
    return [decoder.push(row, eos_frame, i == len(rows) - 1) for i, row in enumerate(rows)]


def whole(frames: np.ndarray) -> bytes:
    return CausalFakeDecoder()(frames).tobytes()


def test_whole_mode_is_one_exact_chunk_at_eos():
    frames = make_frames(40)
    fake = CausalFakeDecoder()
    chunks = run(StreamDecoder(fake, mode="whole"), delayed_rows(frames))
    assert all(c == b"" for c in chunks[:-1])
    assert chunks[-1] == whole(frames)
    assert fake.calls == [40]


def test_prefix_mode_streams_and_matches_the_whole_decode():
    frames = make_frames(60, seed=1)
    fake = CausalFakeDecoder()
    dec = StreamDecoder(fake, mode="prefix", chunk_frames=5, hold_back_frames=2, context_frames=12)
    chunks = run(dec, delayed_rows(frames))
    live = [c for c in chunks[:-1] if c]
    assert len(live) >= 8, "chunks must arrive while frames are still coming"
    first = next(i for i, c in enumerate(chunks) if c)
    assert first == 5 + 2 + (N_CB - 1) - 1
    assert all(len(c) % (HOP * 4) == 0 for c in live)
    assert b"".join(chunks) == whole(frames)
    assert any(call < 60 - 2 and call <= 12 + 5 for call in fake.calls[1:-1])
    assert fake.calls[-1] == 60


def test_prefix_mode_respects_eos():
    frames = make_frames(50, seed=2)
    eos = 31
    frames[eos:, :] = EOA
    fake = CausalFakeDecoder()
    chunks = run(StreamDecoder(fake, mode="prefix", chunk_frames=4, hold_back_frames=1, context_frames=100),
                 delayed_rows(frames))
    assert b"".join(chunks) == whole(frames[:eos])
    assert fake.calls[-1] == eos


@pytest.mark.parametrize("mode", ["whole", "prefix"])
def test_prefill_rows_are_decoded_but_never_streamed(mode):
    """The silence prefill opens the stream: prefix mode never emits its audio; the whole decode re-decodes the
    last overlap rows with the rest (the end-of-stream rule of zonos3.reference.synth.decode_rows)."""
    frames = make_frames(30, seed=3)
    rows = delayed_rows(frames)
    prefill, gen = rows[:3], rows[3:]
    fake = CausalFakeDecoder()
    chunks = run(StreamDecoder(fake, mode=mode, prefill_rows=prefill, chunk_frames=4, hold_back_frames=1,
                               context_frames=100), gen)
    audio = np.frombuffer(b"".join(chunks), dtype=np.float32)
    full = np.frombuffer(whole(frames), dtype=np.float32)
    if mode == "prefix":
        np.testing.assert_array_equal(audio, full[3 * HOP:])
    else:
        np.testing.assert_array_equal(audio, full)


def test_prefix_settings_validation(monkeypatch):
    with pytest.raises(ValueError):
        StreamDecoder(CausalFakeDecoder(), chunk_frames=0)
    with pytest.raises(ValueError):
        StreamDecoder(CausalFakeDecoder(), chunk_frames=20, context_frames=10)
    monkeypatch.delenv("ZONOS3_PREFIX_STREAM_CHUNK_FRAMES", raising=False)
    assert prefix_settings() == (15, 3, 450)
    monkeypatch.setenv("ZONOS3_PREFIX_STREAM_CHUNK_FRAMES", "0")
    with pytest.raises(ValueError, match="chunk_frames >= 1"):
        prefix_settings()
