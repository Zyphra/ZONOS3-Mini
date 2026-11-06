"""Progressive (prefix re-decode) streaming of the matched long-context codecs behind stream_decode="prefix".

A fake decoder stands in for the codec: audio of frame f depends on the codes of frames [f - memory, f] only, i.e. it is
causal with bounded context like the fixed-600k dual codec decoder, so prefix decodes must reproduce the whole decode
exactly and every property below is checked bit for bit:
  * default (no flag): one exact chunk at EOS, the decoder is called once, bytes identical to the whole decode;
  * prefix mode: chunks arrive while frames are still coming, their concatenation plus the EOS tail equals the whole
    decode, the left-context window is used once the history is long, the receipt reports zero deviation;
  * an EOS frame bounds both the streamed chunks and the tail;
  * the flag travels through the message serialization.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zonos3_sgl.message import BaseTTSTokenizerMsg, TTSDetokenizeMsg, TTSSamplingParams, TTSTokenizeMsg
from zonos3_sgl.tokenizer import vocoder
from zonos3.tokenizer.matched_codec import codec_names, entry

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
        msg = TTSDetokenizeMsg(uid=uid, audio_codes=list(row), finished=i == len(rows) - 1, eos_frame=eos_frame, stream_decode=stream_decode)
        chunks.extend(manager.decode_frames([msg]))
    return chunks


@pytest.fixture
def fake(monkeypatch):
    decoder = CausalFakeDecoder(memory=3)
    monkeypatch.setattr(vocoder, "decode_matched", decoder)
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
    assert manager.pop_prefix_receipts() == []


def test_prefix_mode_streams_and_matches_the_whole_decode(fake):
    frames = make_frames(60, seed=1)
    manager = vocoder.TTSVocoderManager(
        n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC,
        prefix_chunk_frames=5, prefix_hold_back_frames=2, prefix_context_frames=12, prefix_first_chunk_frames=5,
    )
    rows = delayed_rows(frames)
    chunks = run(manager, rows, stream_decode="prefix")
    live = [c for c in chunks[:-1] if c]
    assert len(live) >= 8, "chunks must arrive while frames are still coming"
    first = next(i for i, c in enumerate(chunks) if c)
    # first chunk once chunk + hold_back frames are complete, i.e. after that many rows plus the shear tail
    assert first == 5 + 2 + (N_CB - 1) - 1
    assert all(len(c) % (HOP * 4) == 0 for c in live)
    assert b"".join(chunks) == whole_decode(frames)
    # the left-context window was used (decodes shorter than the full prefix) once the history was long
    assert any(call < 60 - 2 and call <= 12 + 5 for call in fake.calls[1:-1])
    assert fake.calls[-1] == 60
    receipt = manager.pop_prefix_receipts()
    assert len(receipt) == 1 and receipt[0]["chunks"] == len(live) and receipt[0]["frames"] == 60
    assert receipt[0]["emitted_frames"] == 5 * len(live)
    assert receipt[0]["live_vs_whole_max_abs"] == 0.0 and receipt[0]["live_vs_whole_rms_db"] < -200
    assert manager.pop_prefix_receipts() == []
    assert not manager._prefix_mode and not manager._prefix_live


def test_prefix_mode_first_chunk_is_shorter(fake):
    frames = make_frames(60, seed=3)
    manager = vocoder.TTSVocoderManager(
        n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC,
        prefix_chunk_frames=5, prefix_hold_back_frames=2, prefix_context_frames=12, prefix_first_chunk_frames=2,
    )
    chunks = run(manager, delayed_rows(frames), stream_decode="prefix")
    live = [c for c in chunks[:-1] if c]
    # the first chunk arrives once first_chunk + hold_back frames are complete, and holds first_chunk frames
    assert next(i for i, c in enumerate(chunks) if c) == 2 + 2 + (N_CB - 1) - 1
    assert len(live[0]) == 2 * HOP * 4
    assert all(len(c) == 5 * HOP * 4 for c in live[1:])
    assert b"".join(chunks) == whole_decode(frames)
    receipt = manager.pop_prefix_receipts()
    assert receipt[0]["emitted_frames"] == 2 + 5 * (len(live) - 1)
    assert receipt[0]["live_vs_whole_max_abs"] == 0.0


def test_prefix_mode_respects_eos(fake):
    frames = make_frames(50, seed=2)
    eos = 31
    frames[eos:, :] = EOA
    manager = vocoder.TTSVocoderManager(
        n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC,
        prefix_chunk_frames=4, prefix_hold_back_frames=1, prefix_context_frames=100,
    )
    chunks = run(manager, delayed_rows(frames), stream_decode="prefix")
    assert b"".join(chunks) == whole_decode(frames[:eos])
    assert fake.calls[-1] == eos
    receipt = manager.pop_prefix_receipts()[0]
    assert receipt["frames"] == eos and receipt["emitted_frames"] <= eos


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
        vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC, prefix_chunk_frames=20, prefix_context_frames=10)
    m = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC)
    assert (m.prefix_first_chunk_frames, m.prefix_chunk_frames, m.prefix_hold_back_frames, m.prefix_context_frames) == (4, 15, 3, 450)
