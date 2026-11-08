"""Streaming decode of the generated rows (zonos3.tts.stream: StreamDecoder, the MLX server's and the reference's, and
PrefixDecoder, the prefix driver of every Python server), ported from servers/sgl/tests/test_tts_prefix_stream.py: a
causal fake decoder (audio of frame f depends on frames [f - memory, f]) and a fake codec stream over it make the prefix
stream reproduce the whole decode exactly, so every property is checked bit for bit."""

from __future__ import annotations

import numpy as np
import pytest
from zonos3.tts.delay import undelay_window
from zonos3.tts.stream import PrefixDecoder, StreamDecoder, StreamSettings

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
            out[f * HOP : (f + 1) * HOP] = codes[max(0, f - self.memory) : f + 1].sum() * 1e-5 * ramp
        return out.astype(np.float32)


class FakeStream:
    """Stateful codec stream over the fake decoder: decode() returns the new frames' audio; nothing lags."""

    def __init__(self):
        self.history = np.zeros((0, N_CB), np.int64)
        self.chunks: list[int] = []

    @property
    def frames(self) -> int:
        return len(self.history)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        start = len(self.history)
        self.history = np.concatenate([self.history, codes])
        self.chunks.append(len(codes))
        return CausalFakeDecoder()(self.history)[start * HOP :]

    def flush(self) -> np.ndarray:
        return np.zeros(0, np.float32)


def stream(decode, mode: str = "whole", prefill_rows=None, codec_stream=None, **settings) -> StreamDecoder:
    return StreamDecoder(
        decode,
        stream=codec_stream if codec_stream is not None else (FakeStream() if mode == "prefix" else None),
        n_codebooks=N_CB,
        audio_pad_id=PAD,
        hop=HOP,
        prefill_rows=prefill_rows,
        mode=mode,
        settings=StreamSettings(**settings),
    )


def delayed_rows(frames: np.ndarray) -> np.ndarray:
    n, cb = frames.shape
    rows = np.full((n + cb - 1, cb), PAD, dtype=np.int64)
    for j in range(cb):
        rows[j : j + n, j] = frames[:, j]
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


def test_prefix_mode_ramps_after_the_prefill_and_matches_the_whole_decode():
    """Default settings: the first chunk (1 frame) once the first generated frame past the silence prefill is
    complete, then 2, 4, 8 and steady 15-frame chunks; the stream equals the whole decode bit for bit."""
    frames = make_frames(70, seed=4)
    rows = delayed_rows(frames)
    prefill, gen = rows[:3], rows[3:]
    fake = CausalFakeDecoder()
    chunks = run(stream(fake, "prefix", prefill_rows=prefill), gen)
    first = next(i for i, c in enumerate(chunks) if c)
    assert first + 1 == 1 + (N_CB - 1)
    sizes = [len(c) // (HOP * 4) for c in chunks[:-1] if c]
    assert sizes[:4] == [1, 2, 4, 8] and sizes[4:] and all(s == 15 for s in sizes[4:])
    full = np.frombuffer(whole(frames), dtype=np.float32)
    np.testing.assert_array_equal(np.frombuffer(b"".join(chunks), dtype=np.float32), full[3 * HOP :])
    assert fake.calls == []


def test_prefix_mode_hold_back_and_steady_chunks():
    frames = make_frames(60, seed=1)
    chunks = run(stream(CausalFakeDecoder(), "prefix", chunk=5, first_chunk=5, hold_back=2), delayed_rows(frames))
    assert next(i for i, c in enumerate(chunks) if c) == 5 + 2 + (N_CB - 1) - 1
    assert all(len(c) == 5 * HOP * 4 for c in chunks[:-1] if c)
    assert b"".join(chunks) == whole(frames)


def test_prefix_mode_respects_eos():
    frames = make_frames(50, seed=2)
    eos = 31
    frames[eos:, :] = EOA
    codec_stream = FakeStream()
    chunks = run(
        stream(CausalFakeDecoder(), "prefix", codec_stream=codec_stream, chunk=4, hold_back=1), delayed_rows(frames)
    )
    assert b"".join(chunks) == whole(frames[:eos])
    assert sum(codec_stream.chunks) == eos


@pytest.mark.parametrize("mode", ["whole", "prefix"])
def test_prefill_rows_are_decoded_but_never_streamed(mode):
    """The silence prefill opens the stream: prefix mode never emits its audio; the whole decode re-decodes the
    last overlap rows with the rest (the end-of-stream rule of zonos3.reference.synth.decode_rows)."""
    frames = make_frames(30, seed=3)
    rows = delayed_rows(frames)
    prefill, gen = rows[:3], rows[3:]
    chunks = run(stream(CausalFakeDecoder(), mode, prefill_rows=prefill, chunk=4, hold_back=1), gen)
    audio = np.frombuffer(b"".join(chunks), dtype=np.float32)
    full = np.frombuffer(whole(frames), dtype=np.float32)
    if mode == "prefix":
        np.testing.assert_array_equal(audio, full[3 * HOP :])
    else:
        np.testing.assert_array_equal(audio, full)


def test_prefix_context_frames_are_decoded_but_not_emitted():
    """The reference tail opens the codec stream ahead of the prefill: decoded once, its audio dropped."""
    frames = make_frames(40, seed=5)
    rows = delayed_rows(frames)
    prefill, gen = rows[:3], rows[3:]
    context = make_frames(5, seed=9)
    codec_stream = FakeStream()
    dec = StreamDecoder(
        CausalFakeDecoder(),
        stream=codec_stream,
        n_codebooks=N_CB,
        audio_pad_id=PAD,
        hop=HOP,
        prefill_rows=prefill,
        context_codes=context,
        mode="prefix",
    )
    chunks = run(dec, gen)
    assert codec_stream.chunks[0] == 5 + 3 + 1
    full = CausalFakeDecoder()(np.concatenate([context, frames]))
    np.testing.assert_array_equal(np.frombuffer(b"".join(chunks), dtype=np.float32), full[(5 + 3) * HOP :])


def test_prefix_finish_without_generated_frames_and_twice():
    rows = delayed_rows(make_frames(4, seed=6)).tolist()
    dec = PrefixDecoder(FakeStream(), n_codebooks=N_CB, audio_pad_id=PAD, hop=HOP, prefill=3)
    assert len(dec.finish(rows, 3)) == 0
    with pytest.raises(RuntimeError):
        dec.finish(rows, 3)


def test_prefix_decoder_ramp_keeps_doubling_after_a_burst():
    """Rows arriving several at a time: each chunk takes every decodable frame, and the next need stays geometric
    (first_chunk + frames emitted: after 5 frames the next chunk waits for 6)."""
    frames = make_frames(60, seed=7)
    rows = delayed_rows(frames).tolist()
    codec_stream = FakeStream()
    dec = PrefixDecoder(codec_stream, n_codebooks=N_CB, audio_pad_id=PAD, hop=HOP, prefill=0)
    out = [dec.step(rows[:n], n - (N_CB - 1)) for n in (N_CB, N_CB + 4, N_CB + 9, N_CB + 30, len(rows))]
    assert len(out[2]) == 0  # 5 new frames decodable, 6 needed
    out.append(dec.finish(rows, len(frames)))
    assert codec_stream.chunks == [1, 4, 26, 29]
    np.testing.assert_array_equal(
        np.concatenate(out), CausalFakeDecoder()(undelay_window(rows, 0, len(frames), N_CB, PAD).numpy())
    )


def test_prefix_mode_needs_a_codec_stream():
    with pytest.raises(ValueError):
        StreamDecoder(CausalFakeDecoder(), n_codebooks=N_CB, audio_pad_id=PAD, hop=HOP, mode="prefix")


def test_early_decode_stops_at_eos_and_never_sends_a_missing_semantic_code():
    """codebooks=1 on every chunk: frame f reaches the codec with row f, its missing residual codes as -1, and
    nothing at or past the EOS frame (EOA in codebook 0) is decoded."""
    frames = make_frames(12, seed=11)
    eos = 7
    frames[eos:] = EOA
    codec = FakeStream()
    dec = StreamDecoder(
        CausalFakeDecoder(), stream=codec, n_codebooks=N_CB, audio_pad_id=PAD, hop=HOP, mode="prefix", codebooks=1
    )
    chunks = run(dec, delayed_rows(frames))
    assert chunks[0] != b""
    assert len(codec.history) == eos and (codec.history[:, 0] == frames[:eos, 0]).all()
    assert (codec.history[eos - 1, 1:] == -1).all()
    known = codec.history >= 0
    assert (codec.history[known] == frames[:eos][known]).all()
    assert len(b"".join(chunks)) == eos * HOP * 4


def test_prefix_settings_validation(monkeypatch):
    for bad in ({"chunk": 0}, {"first_chunk": 0}, {"hold_back": -1}, {"chunk": 20, "context": 10}):
        with pytest.raises(ValueError):
            StreamSettings(**bad)
    for name in ("CHUNK", "FIRST_CHUNK", "HOLD_BACK", "CONTEXT"):
        monkeypatch.delenv(f"ZONOS3_PREFIX_STREAM_{name}_FRAMES", raising=False)
    assert StreamSettings.from_env() == StreamSettings(chunk=15, first_chunk=1, hold_back=0, context=450)
    monkeypatch.setenv("ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES", "0")
    with pytest.raises(ValueError):
        StreamSettings.from_env()
    monkeypatch.setenv("ZONOS3_PREFIX_STREAM_FIRST_CHUNK_FRAMES", "15")
    assert StreamSettings.from_env() == StreamSettings(chunk=15, first_chunk=15, hold_back=0, context=450)
    monkeypatch.setenv("ZONOS3_PREFIX_STREAM_CHUNK_FRAMES", "x")
    with pytest.raises(ValueError):
        StreamSettings.from_env()
