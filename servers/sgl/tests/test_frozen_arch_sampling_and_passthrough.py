"""Scheduler samples GDN-family logits in float32; reference codes pass through the HTTP layer unchanged."""
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


def test_reference_codes_tensor_validation():
    api_server = pytest.importorskip("zonos3_sgl.server.api_server")
    convert = api_server._reference_codes_tensor
    assert convert(None) is None
    codes = convert([[1, 2, 4095], [4096, 0, 7]])
    assert codes.dtype == torch.int32 and codes.shape == (2, 3) and codes.is_contiguous()
    assert codes.tolist() == [[1, 2, 4095], [4096, 0, 7]]
    for bad in ([], [1, 2, 3], [[-1, 2]]):
        with pytest.raises(ValueError, match="speaker_reference_codes"):
            convert(bad)
