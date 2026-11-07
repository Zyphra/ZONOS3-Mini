"""Scheduler samples GDN-family logits in float32; reference codes pass through the HTTP layer checked and cut to
the reference limit."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from zonos3.tts.sampler import sample_logits_in_fp32
from zonos3_sgl.message.tts import TTSTokenizeMsg


def test_gdn_family_samples_from_fp32_logits(monkeypatch):
    monkeypatch.delenv("ZONOS3_SAMPLE_FP32_LOGITS", raising=False)
    assert sample_logits_in_fp32(SimpleNamespace(attention_pattern="gdn3_global1"))
    assert sample_logits_in_fp32(SimpleNamespace(attention_pattern="gdn"))
    assert not sample_logits_in_fp32(SimpleNamespace(attention_pattern="global"))
    assert not sample_logits_in_fp32(SimpleNamespace())  # legacy configs without the field


def test_fp32_sampling_can_be_forced_for_legacy_models(monkeypatch):
    monkeypatch.setenv("ZONOS3_SAMPLE_FP32_LOGITS", "1")
    assert sample_logits_in_fp32(SimpleNamespace(attention_pattern="global"))
    monkeypatch.setenv("ZONOS3_SAMPLE_FP32_LOGITS", "0")
    assert not sample_logits_in_fp32(SimpleNamespace(attention_pattern="global"))


def test_tokenize_message_carries_optional_reference_codes():
    field = {f.name: f for f in dataclasses.fields(TTSTokenizeMsg)}["speaker_reference_codes"]
    assert field.default is None


def test_inline_reference_codes_are_checked_and_cut():
    api_server = pytest.importorskip("zonos3_sgl.server.api_server")
    from zonos3.generation import SAMPLE_RATE
    from zonos3.tokenizer.matched_codec import entry
    from zonos3.tts import plan

    codec = "opus_dual_3n_600k_sched600k_r4096_step600000"
    config = SimpleNamespace(
        model_config=SimpleNamespace(n_codebooks=3, codebook_size=4097),
        tts_audio_codec=codec,
        tts_speaker_reference_max_seconds=1.0,
    )
    ctx = api_server._RulesContext(config)
    convert = lambda codes: plan.reference_codes(ctx, codes)  # noqa: E731
    assert convert(None) is None
    codes = convert([[1, 2, 4095], [4096, 0, 7]])
    assert codes.dtype == torch.int32 and codes.shape == (2, 3) and codes.is_contiguous()
    assert codes.tolist() == [[1, 2, 4095], [4096, 0, 7]]
    for bad in ([], [1, 2, 3], [[-1, 2]], [[-1, 2, 3]], [[4097, 0, 0]]):
        with pytest.raises(ValueError):
            convert(bad)
    # longer codes keep their first frames: the checkpoint's speaker_reference_max_seconds at the codec hop
    assert ctx.reference_max_frames == int(SAMPLE_RATE // int(entry(codec)["hop"]))
    long = [[i % 4096, 0, 0] for i in range(ctx.reference_max_frames + 5)]
    assert convert(long).tolist() == long[: ctx.reference_max_frames]
