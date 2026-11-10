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
        self.decoders: list[str | None] = []  # the decoder id of every call

    def __call__(self, codes: torch.Tensor, audio_codec: str, decoder: str | None = None) -> torch.Tensor:
        assert audio_codec == CODEC
        assert codes.ndim == 3 and codes.shape[-1] == N_CB
        b, frames, _ = codes.shape
        self.calls.append(int(frames))
        self.decoders.append(decoder)
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


def run(
    manager: vocoder.TTSVocoderManager, rows: list[list[int]], uid: int = 7, stream_decode=None, eos_frame=None, **extra
):
    chunks = []
    for i, row in enumerate(rows):
        msg = TTSDetokenizeMsg(
            uid=uid,
            audio_codes=list(row),
            finished=i == len(rows) - 1,
            eos_frame=eos_frame,
            stream_decode=stream_decode,
            **extra,
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

    def fork(self) -> FakeCodecStream:
        twin = FakeCodecStream(self.decoder)
        twin.history = self.history.copy()
        return twin


@pytest.fixture
def fake(monkeypatch):
    decoder = CausalFakeDecoder(memory=3)
    decoder.streams = []
    decoder.stream_decoders = []  # the decoder id of every stream opened

    def stream(codec, decoder_id=None):
        assert codec == CODEC
        decoder.streams.append(FakeCodecStream(CausalFakeDecoder(memory=3)))
        decoder.stream_decoders.append(decoder_id)
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
    chunks = run(manager, delayed_rows(frames), stream_decode="prefix", stream_codebooks=N_CB)
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
    chunks = run(manager, delayed_rows(frames), stream_decode="prefix", stream_codebooks=N_CB)
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
    chunks = run(manager, delayed_rows(frames), stream_decode="prefix", stream_codebooks=N_CB)
    assert b"".join(chunks) == whole_decode(frames[:eos])
    assert sum(fake.streams[0].chunks) == eos


def lm_completed_silence_prefill(gen_frames: np.ndarray, prefill: int = 3, seed: int = 4):
    """[silence prefill | generated] frames as the LM leaves them: the prefill rows hold the silence frame's sheared
    lower triangle, the codebooks of the prefill frames that fall in generated rows (f + j >= prefill) are the LM's."""
    silence = np.asarray(SPEC["silence_frame"], np.int64)
    frames = np.concatenate([np.repeat(silence[None], prefill, 0), gen_frames])
    rng = np.random.RandomState(seed)
    for f in range(prefill):
        for j in range(prefill - f, N_CB):
            frames[f, j] = rng.randint(0, 4096)
    return silence, frames


def test_silence_prefill_continues_a_primed_stream(fake):
    """A prefix request opened by the silence prefill (no decode context) continues a stream that decoded the prefill
    frames once as pure silence (shared by every such request): the first chunk is one generated frame, and the
    audio is the generated frames' decoded behind pure silence, not behind the LM-completed prefill frames."""
    manager = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC)
    for uid, seed in ((1, 5), (2, 6)):
        gen = make_frames(40, seed=seed)
        silence, frames = lm_completed_silence_prefill(gen)
        rows = delayed_rows(frames)
        manager.register_request_context(uid, prefill_rows=np.asarray(rows[:3]))
        chunks = run(manager, rows[3:], uid=uid, stream_decode="prefix", stream_codebooks=N_CB)
        live = [len(c) // (HOP * 4) for c in chunks[:-1] if c]
        assert live[:5] == [1, 2, 4, 8, 15]
        primed = whole_decode(np.concatenate([np.repeat(silence[None], 3, 0), gen]))[3 * HOP * 4 :]
        assert b"".join(chunks) == primed
        assert primed != whole_decode(frames)[3 * HOP * 4 :]  # the LM's prefill codebooks would change the audio
    assert [s.chunks for s in fake.streams] == [[3]]  # one primed stream, decoded once


def test_other_prefill_rows_decode_from_the_stream_start(fake):
    manager = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC)
    frames = make_frames(43, seed=7)
    rows = delayed_rows(frames)
    manager.register_request_context(9, prefill_rows=np.asarray(rows[:3]))
    chunks = run(manager, rows[3:], uid=9, stream_decode="prefix", stream_codebooks=N_CB)
    assert b"".join(chunks) == whole_decode(frames)[3 * HOP * 4 :]
    assert fake.streams[0].chunks[0] == 4  # prefill + first generated frame


@pytest.mark.parametrize("codebooks", [1, 4])
def test_stream_codebooks_decode_frames_before_their_last_codebooks(fake, codebooks):
    """stream_codebooks = k: a frame is decoded once its first k codebooks arrived (row f + k - 1), the missing
    residual codes reach the codec as -1; every frame is decoded once and the audio keeps the whole decode's length."""
    frames = make_frames(40, seed=8)
    manager = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC)
    chunks = run(
        manager, delayed_rows(frames), stream_decode="prefix", stream_codebooks=codebooks, stream_codebooks_scope="all"
    )
    assert next(i for i, c in enumerate(chunks) if c) == codebooks - 1
    decoded = fake.streams[0].history
    assert (decoded[0, :codebooks] == frames[0, :codebooks]).all() and (decoded[0, codebooks:] == -1).all()
    known = decoded >= 0
    assert (decoded[known] == frames[known]).all() and (decoded[-1] == frames[-1]).all()  # the EOS tail is complete
    assert len(b"".join(chunks)) == len(whole_decode(frames))


def test_default_decodes_the_first_chunk_from_one_codebook_then_complete_frames(fake):
    frames = make_frames(40, seed=9)
    manager = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC)
    rows = delayed_rows(frames)
    chunks = run(manager, rows, stream_decode="prefix")  # the registry default: k = 1 on the first chunk
    assert next(i for i, c in enumerate(chunks) if c) == 0
    decoded = fake.streams[0].history
    assert (decoded[0, 1:] == -1).all() and (decoded[1:] == frames[1:]).all()
    # the second chunk (frames 1-2) waits for frame 2's last codebook: row 2 + N_CB - 1
    assert next(i for i, c in enumerate(chunks) if c and i > 0) == 2 + N_CB - 1


def test_each_request_decodes_through_its_decoder(fake):
    """A message's decoder (None = the default, keep2_plain) picks the codec model of the whole decode, the prefix
    stream and the primed silence stream (one per decoder); an id the server does not serve is refused."""
    manager = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC)
    assert manager.decoders == ["keep2_plain", "full_mr"]
    frames = make_frames(20, seed=10)
    run(manager, delayed_rows(frames), uid=1)
    run(manager, delayed_rows(frames), uid=2, decoder="full_mr")
    assert fake.decoders == ["keep2_plain", "full_mr"]
    run(manager, delayed_rows(frames), uid=3, stream_decode="prefix", decoder="full_mr")
    run(manager, delayed_rows(frames), uid=4, stream_decode="prefix")
    assert fake.stream_decoders == ["full_mr", "keep2_plain"]
    for uid, decoder in ((5, "full_mr"), (6, None), (7, "full_mr")):
        _, primed = lm_completed_silence_prefill(make_frames(20, seed=uid))
        rows = delayed_rows(primed)
        manager.register_request_context(uid, prefill_rows=np.asarray(rows[:3]))
        run(manager, rows[3:], uid=uid, stream_decode="prefix", stream_codebooks=N_CB, decoder=decoder)
    assert fake.stream_decoders[2:] == ["full_mr", "keep2_plain"]  # one primed stream per decoder
    assert not manager._decoder
    with pytest.raises(ValueError):
        run(manager, delayed_rows(frames), uid=8, decoder="nope")


def test_decoders_restrict_and_order_the_served_decoders():
    manager = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC, decoders=["full_mr"])
    assert manager.decoders == ["full_mr"] and manager.request_decoder(None) == "full_mr"
    with pytest.raises(ValueError):
        manager.request_decoder("keep2_plain")
    both = vocoder.TTSVocoderManager(
        n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC, decoders=["full_mr", "keep2_plain"]
    )
    assert both.decoders == ["keep2_plain", "full_mr"]  # the registry default stays the default
    with pytest.raises(ValueError):
        vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC, decoders=["nope"])


def test_stream_decode_survives_serialization():
    detok = TTSDetokenizeMsg(uid=3, audio_codes=[1, 2, 3], finished=False, stream_decode="prefix")
    back = BaseTTSTokenizerMsg.decoder(BaseTTSTokenizerMsg.encoder(detok))
    assert back.stream_decode == "prefix" and back.audio_codes == [1, 2, 3]
    assert TTSDetokenizeMsg(uid=3, audio_codes=[1], finished=True).stream_decode is None
    tok = TTSTokenizeMsg(uid=4, text="hi", sampling_params=TTSSamplingParams(stream_decode="prefix", seed=5))
    back = BaseTTSTokenizerMsg.decoder(BaseTTSTokenizerMsg.encoder(tok))
    assert back.sampling_params.stream_decode == "prefix" and back.sampling_params.seed == 5
    assert TTSSamplingParams().stream_decode is None
    detok = TTSDetokenizeMsg(uid=3, audio_codes=[1], finished=False, decoder="full_mr")
    assert BaseTTSTokenizerMsg.decoder(BaseTTSTokenizerMsg.encoder(detok)).decoder == "full_mr"
    tok = TTSTokenizeMsg(uid=4, text="hi", sampling_params=TTSSamplingParams(decoder="full_mr"))
    assert BaseTTSTokenizerMsg.decoder(BaseTTSTokenizerMsg.encoder(tok)).sampling_params.decoder == "full_mr"
    assert TTSSamplingParams().decoder is None


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
