"""Progressive (prefix) streaming of the matched long-context codecs behind stream_decode="prefix".

A fake decoder stands in for the codec: audio of frame f depends on the codes of frames [f - memory, f] only, i.e. it is
causal with bounded context like the codec decoder; the fake codec stream decodes incrementally with that decoder, so
every property below is checked bit for bit:
  * default (no flag): one exact chunk at EOS, the decoder is called once, bytes identical to the whole decode;
  * prefix mode: chunks ramp 1, 2, 4, 8, then chunk frames while frames are still coming, and their concatenation plus
    the EOS tail equals the whole decode;
  * an EOS frame bounds both the streamed chunks and the tail;
  * the flag travels through the message serialization.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from zonos3.tokenizer.matched_codec import codec_names, entry
from zonos3.tts.stream import StreamSettings
from zonos3_sgl.message import BaseTTSTokenizerMsg, TTSDetokenizeMsg, TTSSamplingParams, TTSTokenizeMsg
from zonos3_sgl.tokenizer import vocoder

CODEC = sorted(codec_names())[0]
SPEC = entry(CODEC)
N_CB = int(SPEC["codebooks"])
HOP = int(SPEC["hop"])
PAD = 4097
EOA = 4096


class CausalFakeDecoder:
    """audio[f] = (sum of the codes of frames f-memory..f) * ramp, for every codebook; counts its calls."""

    def __init__(self, memory: int = 3):
        self.memory = memory
        self.calls: list[int] = []

    def __call__(self, codes: torch.Tensor, audio_codec: str) -> torch.Tensor:
        assert audio_codec == CODEC
        assert codes.ndim == 3 and codes.shape[-1] == N_CB
        b, frames, _ = codes.shape
        self.calls.append(int(frames))
        c = codes.to(torch.float64).cpu()
        out = torch.zeros(b, frames * HOP, dtype=torch.float64)
        ramp = torch.linspace(0.5, 1.0, HOP, dtype=torch.float64)
        for f in range(frames):
            value = c[:, max(0, f - self.memory) : f + 1, :].sum(dim=(1, 2)) * 1e-5
            out[:, f * HOP : (f + 1) * HOP] = value[:, None] * ramp[None, :]
        return out.to(torch.float32)


def delayed_rows(frames: np.ndarray, pad: int = PAD) -> list[list[int]]:
    """The model's delayed row layout: row r carries codebook j of frame r - j (the inverse of shear_up)."""
    n, cb = frames.shape
    rows = np.full((n + cb - 1, cb), pad, dtype=np.int64)
    for j in range(cb):
        rows[j : j + n, j] = frames[:, j]
    return rows.tolist()


def whole_decode(frames: np.ndarray) -> bytes:
    """The reference: a fresh decoder over the whole sequence (so the manager's call count stays untouched)."""
    audio = CausalFakeDecoder(memory=3)(torch.tensor(frames, dtype=torch.int64).unsqueeze(0), CODEC)
    return audio[0].numpy().astype(np.float32).tobytes()


def run(manager: vocoder.TTSVocoderManager, rows: list[list[int]], uid: int = 7, stream_decode=None, eos_frame=None):
    chunks = []
    for i, row in enumerate(rows):
        msg = TTSDetokenizeMsg(
            uid=uid,
            audio_codes=list(row),
            finished=i == len(rows) - 1,
            eos_frame=eos_frame,
            stream_decode=stream_decode,
        )
        chunks.extend(manager.decode_frames([msg]))
    return chunks


class FakeCodecStream:
    """Stateful stream over the fake decoder: decode() returns the new frames' audio, nothing lags, flush() is empty."""

    def __init__(self, decoder: CausalFakeDecoder):
        self.decoder = decoder
        self.history = np.zeros((0, N_CB), np.int64)
        self.chunks: list[int] = []

    @property
    def frames(self) -> int:
        return len(self.history)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        start = len(self.history)
        self.history = np.concatenate([self.history, codes])
        self.chunks.append(len(codes))
        audio = self.decoder(torch.tensor(self.history).unsqueeze(0), CODEC)
        return audio[0, start * HOP :].numpy()

    def flush(self) -> np.ndarray:
        return np.zeros(0, np.float32)


@pytest.fixture
def fake(monkeypatch):
    decoder = CausalFakeDecoder(memory=3)
    decoder.streams = []

    def stream(codec):
        assert codec == CODEC
        decoder.streams.append(FakeCodecStream(CausalFakeDecoder(memory=3)))
        return decoder.streams[-1]

    monkeypatch.setattr(vocoder, "decode_matched", decoder)
    monkeypatch.setattr(vocoder, "stream_matched", stream)
    monkeypatch.setattr(vocoder.torch.cuda, "is_available", lambda: False)
    return decoder


def make_frames(n: int, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 4096, size=(n, N_CB)).astype(np.int64)


def test_default_mode_is_one_exact_chunk_at_eos(fake):
    frames = make_frames(40)
    manager = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC)
    assert manager.prefix_stream_supported
    chunks = run(manager, delayed_rows(frames))
    assert all(c == b"" for c in chunks[:-1])
    assert chunks[-1] == whole_decode(frames)
    assert fake.calls == [40]
    assert fake.streams == []


def test_prefix_mode_ramps_and_matches_the_whole_decode(fake):
    frames = make_frames(60, seed=1)
    manager = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC)
    chunks = run(manager, delayed_rows(frames), stream_decode="prefix")
    # the first chunk (1 frame) once that frame is complete: after its row plus the shear tail
    assert next(i for i, c in enumerate(chunks) if c) == 1 + (N_CB - 1) - 1
    live = [len(c) // (HOP * 4) for c in chunks[:-1] if c]
    assert live[:6] == [1, 2, 4, 8, 15, 15]
    assert fake.streams[0].chunks[:6] == live[:6]
    assert b"".join(chunks) == whole_decode(frames)
    assert fake.calls == []
    assert not manager._prefix


def test_prefix_mode_hold_back_and_steady_chunks(fake):
    frames = make_frames(60, seed=3)
    manager = vocoder.TTSVocoderManager(
        n_codebooks=N_CB,
        audio_pad_id=PAD,
        audio_codec=CODEC,
        prefix_chunk_frames=5,
        prefix_hold_back_frames=2,
        prefix_context_frames=12,
        prefix_first_chunk_frames=5,
    )
    chunks = run(manager, delayed_rows(frames), stream_decode="prefix")
    assert next(i for i, c in enumerate(chunks) if c) == 5 + 2 + (N_CB - 1) - 1
    assert all(len(c) == 5 * HOP * 4 for c in chunks[:-1] if c)
    assert b"".join(chunks) == whole_decode(frames)


def test_prefix_mode_respects_eos(fake):
    frames = make_frames(50, seed=2)
    eos = 31
    frames[eos:, :] = EOA
    manager = vocoder.TTSVocoderManager(
        n_codebooks=N_CB,
        audio_pad_id=PAD,
        audio_codec=CODEC,
        prefix_chunk_frames=4,
        prefix_hold_back_frames=1,
        prefix_context_frames=100,
    )
    chunks = run(manager, delayed_rows(frames), stream_decode="prefix")
    assert b"".join(chunks) == whole_decode(frames[:eos])
    assert sum(fake.streams[0].chunks) == eos


def test_stream_decode_survives_serialization():
    detok = TTSDetokenizeMsg(uid=3, audio_codes=[1, 2, 3], finished=False, stream_decode="prefix")
    back = BaseTTSTokenizerMsg.decoder(BaseTTSTokenizerMsg.encoder(detok))
    assert back.stream_decode == "prefix" and back.audio_codes == [1, 2, 3]
    assert TTSDetokenizeMsg(uid=3, audio_codes=[1], finished=True).stream_decode is None
    tok = TTSTokenizeMsg(uid=4, text="hi", sampling_params=TTSSamplingParams(stream_decode="prefix", seed=5))
    back = BaseTTSTokenizerMsg.decoder(BaseTTSTokenizerMsg.encoder(tok))
    assert back.sampling_params.stream_decode == "prefix" and back.sampling_params.seed == 5
    assert TTSSamplingParams().stream_decode is None


def test_prefix_settings_validation():
    with pytest.raises(ValueError):
        vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC, prefix_chunk_frames=0)
    with pytest.raises(ValueError):
        vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC, prefix_first_chunk_frames=0)
    with pytest.raises(ValueError):
        vocoder.TTSVocoderManager(
            n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC, prefix_chunk_frames=20, prefix_context_frames=10
        )
    m = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC)
    assert m.prefix == StreamSettings(chunk=15, first_chunk=1, hold_back=0, context=450)
