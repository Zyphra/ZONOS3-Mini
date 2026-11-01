from __future__ import annotations

import numpy as np
import torch
from zonos3.message import TTSDetokenizeMsg
from zonos3.tokenizer import vocoder
from zonos3.tokenizer.matched_codec import codec_names, entry

CODEC = sorted(codec_names())[0]
N_CB = int(entry(CODEC)["codebooks"])
HOP = int(entry(CODEC)["hop"])
PAD = 4097


def delayed_rows(frames: np.ndarray, pad: int = PAD) -> list[list[int]]:
    """The model's delayed row layout: row r carries codebook j of frame r - j (the inverse of shear_up)."""
    n, cb = frames.shape
    rows = np.full((n + cb - 1, cb), pad, dtype=np.int64)
    for j in range(cb):
        rows[j : j + n, j] = frames[:, j]
    return rows.tolist()


def test_matched_decode_stops_before_the_eos_frame(monkeypatch):
    captured: list[torch.Tensor] = []

    def fake_decode_matched(codes: torch.Tensor, audio_codec: str) -> torch.Tensor:
        assert audio_codec == CODEC
        captured.append(codes.cpu().clone())
        return torch.zeros((codes.shape[0], codes.shape[1] * HOP), dtype=torch.float32)

    monkeypatch.setattr(vocoder, "decode_matched", fake_decode_matched)
    monkeypatch.setattr(vocoder.torch.cuda, "is_available", lambda: False)

    manager = vocoder.TTSVocoderManager(n_codebooks=N_CB, audio_pad_id=PAD, audio_codec=CODEC)
    frames = np.random.RandomState(0).randint(0, 4096, size=(6, N_CB)).astype(np.int64)
    rows = delayed_rows(frames)
    msgs = [TTSDetokenizeMsg(uid=1, audio_codes=row, finished=False) for row in rows[:-1]]
    msgs.append(TTSDetokenizeMsg(uid=1, audio_codes=rows[-1], finished=True, eos_frame=2))
    chunks = manager.decode_frames(msgs)

    assert all(chunk == b"" for chunk in chunks[:-1])
    assert chunks[-1]
    assert len(captured) == 1
    assert captured[0].tolist() == [frames[:2].tolist()]
