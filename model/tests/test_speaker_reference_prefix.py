"""The runtime's raw-reference prefix equals the training loader's encode_reference (apps.zonos.reference_packing)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from zonos3.tts.prompt import TTSPromptConfig, speaker_reference_prefix_tokens


def training_reference_prefix(codes: np.ndarray, n_codebooks: int, audio_pad: int, text_pad: int) -> np.ndarray:
    """Independent re-implementation of apps.zonos.reference_packing.encode_reference + gom_jabbar._shear."""
    length = len(codes) + n_codebooks - 1
    columns = np.full((length, n_codebooks), audio_pad, dtype=np.int64)
    columns[: len(codes)] = codes
    padded = np.full((length + n_codebooks - 1, n_codebooks), audio_pad, dtype=np.int64)
    padded[n_codebooks - 1 :] = columns
    rows = n_codebooks - 1 + np.arange(length)[:, None] - np.arange(n_codebooks)[None, :]
    sheared = np.take_along_axis(padded, rows, axis=0)
    return np.concatenate((sheared, np.full((length, 1), text_pad, dtype=np.int64)), axis=1)


@pytest.mark.parametrize(
    "n_codebooks, codebook_sizes, audio_pad, text_vocab, frames",
    [
        (10, (4096,) * 10, 4097, 8196, 37),  # frozen dual-codec / SuperBPE family
        (10, (4096,) * 10, 4097, 8196, 1),
    ],
)
def test_prefix_matches_the_training_encoder(n_codebooks, codebook_sizes, audio_pad, text_vocab, frames):
    config = TTSPromptConfig(
        n_codebooks=n_codebooks, codebook_sizes=codebook_sizes, audio_pad_id=audio_pad, text_vocab=text_vocab
    )
    size = max(codebook_sizes) if codebook_sizes else 1024
    codes = np.random.default_rng(frames).integers(0, size, size=(frames, n_codebooks))

    prefix = speaker_reference_prefix_tokens(torch.from_numpy(codes), config)

    expected = training_reference_prefix(codes, n_codebooks, audio_pad, text_vocab)
    assert prefix.shape == (frames + n_codebooks - 1, n_codebooks + 1)
    np.testing.assert_array_equal(prefix.numpy(), expected)
    # codebook c is delayed by c rows, the tail is flushed with audio padding, the text column is all text_pad
    for c in range(n_codebooks):
        np.testing.assert_array_equal(prefix[c : c + frames, c].numpy(), codes[:, c])
        assert (prefix[:c, c] == audio_pad).all() and (prefix[c + frames :, c] == audio_pad).all()
    assert (prefix[:, n_codebooks] == text_vocab).all()


def test_large_codebooks_need_codebook_sizes():
    codes = torch.full((5, 10), 4000, dtype=torch.int64)
    with pytest.raises(ValueError):
        speaker_reference_prefix_tokens(codes, TTSPromptConfig(n_codebooks=10, audio_pad_id=4097, text_vocab=8196))
    speaker_reference_prefix_tokens(
        codes, TTSPromptConfig(n_codebooks=10, codebook_sizes=(4096,) * 10, audio_pad_id=4097, text_vocab=8196)
    )


def test_rejects_bad_shapes_and_values():
    config = TTSPromptConfig(n_codebooks=10, codebook_sizes=(4096,) * 10, audio_pad_id=4097, text_vocab=8196)
    with pytest.raises(ValueError):
        speaker_reference_prefix_tokens(torch.zeros((0, 10), dtype=torch.int64), config)
    with pytest.raises(ValueError):
        speaker_reference_prefix_tokens(torch.zeros((4, 9), dtype=torch.int64), config)
    with pytest.raises(ValueError):
        speaker_reference_prefix_tokens(torch.full((4, 10), 4096, dtype=torch.int64), config)
    with pytest.raises(ValueError):
        speaker_reference_prefix_tokens(torch.full((4, 10), -1, dtype=torch.int64), config)
