"""Streaming decode of the generated rows (zonos3.tts.stream.StreamDecoder, the MLX server's and the reference's), ported
from servers/sgl/tests/test_tts_prefix_stream.py: a causal fake decoder (audio of frame f depends on frames
[f - memory, f]) makes prefix re-decodes reproduce the whole decode exactly, so every property is checked bit for bit."""

from __future__ import annotations

import numpy as np
import pytest
from zonos3.tts.stream import StreamDecoder, StreamSettings

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


def stream(decode, mode: str = "whole", prefill_rows=None, **settings) -> StreamDecoder:
    return StreamDecoder(decode, n_codebooks=N_CB, audio_pad_id=PAD, hop=HOP, prefill_rows=prefill_rows, mode=mode,
                         settings=StreamSettings(**settings))


def delayed_rows(frames: np.ndarray) -> np.ndarray:
    n, cb = frames.shape
    rows = np.full((n + cb - 1, cb), PAD, dtype=np.int64)
    for j in range(cb):
        rows[j:j + n, j] = frames[:, j]
    return rows


def make_frames(n: int, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 4096, size=(n, N_CB)).astype(np.int64)


def run(decoder: StreamDecoder, rows, eos_frame=None) -> list[bytes]:
    return [decoder.push(row, eos_frame, i == len(rows) - 1).tobytes() for i, row in enumerate(rows)]


def whole(frames: np.ndarray) -> bytes:
    return CausalFakeDecoder()(frames).tobytes()


def test_whole_mode_is_one_exact_chunk_at_eos():
    frames = make_frames(40)
    fake = CausalFakeDecoder()
    chunks = run(stream(fake), delayed_rows(frames))
    assert all(c == b"" for c in chunks[:-1])
    assert chunks[-1] == whole(frames)
    assert fake.calls == [40]


def test_prefix_mode_streams_and_matches_the_whole_decode():
    frames = make_frames(60, seed=1)
    fake = CausalFakeDecoder()
    dec = stream(fake, "prefix", chunk=5, first_chunk=3, hold_back=2, context=12)
    chunks = run(dec, delayed_rows(frames))
    live = [c for c in chunks[:-1] if c]
    assert len(live) >= 8, "chunks must arrive while frames are still coming"
    first = next(i for i, c in enumerate(chunks) if c)
    assert first == 3 + 2 + (N_CB - 1) - 1
    assert len(chunks[first]) == 3 * HOP * 4
    assert all(len(c) == 5 * HOP * 4 for c in live[1:])
    assert b"".join(chunks) == whole(frames)
    assert any(call < 60 - 2 and call <= 12 + 5 for call in fake.calls[1:-1])
    assert fake.calls[-1] == 60


def test_first_prefix_chunk_lands_at_the_first_chunk_boundary_after_the_prefill():
    """The default first chunk (4 frames) is emitted once the first 4 generated frames past the silence prefill are
    decodable; every later chunk waits for the chunk size; the stream still equals the whole decode bit for bit."""
    frames = make_frames(70, seed=4)
    rows = delayed_rows(frames)
    prefill, gen = rows[:3], rows[3:]
    fake = CausalFakeDecoder()
    dec = stream(fake, "prefix", prefill_rows=prefill)
    assert (dec.settings.first_chunk, dec.settings.chunk, dec.settings.hold_back) == (4, 15, 3)
    chunks = run(dec, gen)
    first = next(i for i, c in enumerate(chunks) if c)
    # decodable = (3 + i + 1) - (N_CB - 1) - 3 must reach prefill + 4: 4 + 3 + (N_CB - 1) generated rows
    assert first + 1 == 4 + 3 + (N_CB - 1)
    assert len(chunks[first]) == 4 * HOP * 4
    live = [c for c in chunks[first + 1:-1] if c]
    assert live and all(len(c) == 15 * HOP * 4 for c in live)
    full = np.frombuffer(whole(frames), dtype=np.float32)
    np.testing.assert_array_equal(np.frombuffer(b"".join(chunks), dtype=np.float32), full[3 * HOP:])


def test_prefix_mode_respects_eos():
    frames = make_frames(50, seed=2)
    eos = 31
    frames[eos:, :] = EOA
    fake = CausalFakeDecoder()
    chunks = run(stream(fake, "prefix", chunk=4, hold_back=1, context=100), delayed_rows(frames))
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
    chunks = run(stream(fake, mode, prefill_rows=prefill, chunk=4, hold_back=1, context=100), gen)
    audio = np.frombuffer(b"".join(chunks), dtype=np.float32)
    full = np.frombuffer(whole(frames), dtype=np.float32)
    if mode == "prefix":
        np.testing.assert_array_equal(audio, full[3 * HOP:])
    else:
        np.testing.assert_array_equal(audio, full)


def test_prefix_settings_validation(monkeypatch):
    for bad in (dict(chunk=0), dict(first_chunk=0), dict(hold_back=-1), dict(chunk=20, context=10)):
        with pytest.raises(ValueError):
            StreamSettings(**bad)
    for name in ("CHUNK", "FIRST_CHUNK", "HOLD_BACK", "CONTEXT"):
        monkeypatch.delenv(f"ZONOS3_PREFIX_STREAM_{name}_FRAMES", raising=False)
    assert StreamSettings.from_env() == StreamSettings(chunk=15, first_chunk=4, hold_back=3, context=450)
    monkeypatch.setenv("ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES", "0")
    with pytest.raises(ValueError):
        StreamSettings.from_env()
    monkeypatch.setenv("ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES", "15")
    assert StreamSettings.from_env() == StreamSettings(chunk=15, first_chunk=15, hold_back=3, context=450)
    monkeypatch.setenv("ZONOS3_PREFIX_STREAM_CHUNK_FRAMES", "x")
    with pytest.raises(ValueError):
        StreamSettings.from_env()
