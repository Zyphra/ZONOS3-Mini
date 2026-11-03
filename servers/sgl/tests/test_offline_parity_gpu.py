"""Weights-backed offline<->server parity (loads the real model; GPU).

Skipped unless ZONOS3_TEST_MODEL points at a checkpoint (e.g. "hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/rate30_20k_r2grpo").
Run with:

    ZONOS3_TEST_MODEL=hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/rate30_20k_r2grpo CUDA_VISIBLE_DEVICES=2 \
        PYTHONPATH=.../ZONOS3-offline-paritiy/python \
        python -m pytest -o addopts="" tests/misc/test_offline_parity_gpu.py -q

Asserts the offline path builds the *same* model input_ids the server (tokenizer
worker + scheduler) builds, and that normalization / voice cloning actually take
effect end to end. Everything downstream of the front end is shared code, so
token-exact input == token-exact output by construction.
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL = os.environ.get("ZONOS3_TEST_MODEL")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not MODEL, reason="set ZONOS3_TEST_MODEL to run (loads weights)"),
]

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VOICE = os.path.join(REPO, "default_voices", "AmericanFemale.mp3")
TEXT = "Call me at 123 Main Street."
SEED = 12345


@pytest.fixture(scope="module")
def tts():
    from zonos3_sgl.tts.llm import TTSLLM

    # the server's backend (servers/sgl/scripts/serve.sh): GDN layers keep per-request state, and FlashInfer has no head_dim-32 prefill
    return TTSLLM(model_path=MODEL, attention_backend="fa", cache_type="naive")


def _offline_input(tts, speaker_embedding):
    """Final model input the offline path feeds the scheduler."""
    ids = tts._tokenize_one(TEXT, language="en_us", text_normalization=True)
    if speaker_embedding is None:
        return ids
    return tts._with_speaker_frames(ids, -1)


def _server_input(tts, speaker_embedding):
    """Final model input the server (tokenizer worker + scheduler) feeds.

    tokenizer/server.py: normalize -> build(text, quality_buckets=default), then
    prepend the speaker slot (position 0) when an embedding is present; the
    scheduler adds a legacy checkpoint's background/accurate markers.
    """
    from zonos3.tts.conditioning import default_quality_buckets

    norm = tts._text_normalizer.normalize(TEXT, "en_us")
    default_quality = default_quality_buckets(tts.quality_features, tts.quality_bucket_counts)
    ids = tts._prompt_builder.build(norm, quality_buckets=default_quality)
    pos = -1
    if speaker_embedding is not None:
        slot = tts._prompt_builder.speaker_slot(dtype=ids.dtype, device=ids.device)
        ids = torch.cat([slot, ids], dim=0)
        pos = 0
    if speaker_embedding is None:
        return ids
    return tts._with_speaker_frames(ids, pos)


def test_text_normalization_rewrites_digits(tts):
    norm = tts._text_normalizer.normalize(TEXT, "en_us")
    assert norm != TEXT and "123" not in norm


def test_embed_speaker_file_dim(tts):
    emb = tts.embed_speaker_file(VOICE)
    assert emb.numel() == tts.speaker_embedding_dim
    assert emb.dtype == torch.float32


@pytest.mark.parametrize("clone", [False, True])
def test_front_end_input_ids_match_server(tts, clone):
    emb = tts.embed_speaker_file(VOICE) if clone else None
    a = _offline_input(tts, emb)
    b = _server_input(tts, emb)
    assert a.shape == b.shape
    assert torch.equal(a, b)


def test_normalization_changes_generated_tokens(tts):
    from zonos3_sgl.message.tts import TTSSamplingParams

    norm = tts.generate_one(TEXT, TTSSamplingParams(seed=SEED), text_normalization=True)
    raw = tts.generate_one(TEXT, TTSSamplingParams(seed=SEED), text_normalization=False)
    assert norm["audio_tokens"] != raw["audio_tokens"]


def test_voice_cloning_changes_tokens_and_produces_audio(tts):
    from zonos3_sgl.message.tts import TTSSamplingParams

    emb = tts.embed_speaker_file(VOICE)
    plain = tts.generate_one(TEXT, TTSSamplingParams(seed=SEED))
    clone = tts.generate_one(TEXT, TTSSamplingParams(seed=SEED), speaker_embedding=emb)
    assert clone["audio_tokens"] != plain["audio_tokens"]
    assert clone["audio"] and len(clone["audio"]) > 0
