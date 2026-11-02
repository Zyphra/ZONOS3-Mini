from __future__ import annotations

import pytest
from zonos3_sgl.server import api_server
from zonos3.server.schemas import GenerateRequest, SpeechRequest
from zonos3.tts.prompt import (
    BYTE_TEXT_VOCAB_SIZE,
    text_to_byte_ids,
    tokens_to_prompt_tokens,
)
from zonos3.utils.hf import cached_load_checkpoint_config


class _MaxTokenConfig:
    max_seq_len = 4096


def test_tokens_to_prompt_tokens_emits_no_speaking_rate_frame():
    # Legacy rate-bucket checkpoints keep their vocab layout but get no rate token.
    prompt = tokens_to_prompt_tokens(
        [7, 8],
        n_codebooks=2,
        audio_pad_id=99,
        text_vocab=469,
        speaking_rate_num_buckets=21,
    )

    assert prompt == [
        [99, 99, 7],
        [99, 99, 8],
    ]


def test_tts_request_defaults():
    tts_req = GenerateRequest(text="Hello")
    assert not hasattr(tts_req, "byte_tokenize_all")
    assert tts_req.language == "en_us"
    assert tts_req.text_normalization is True
    assert tts_req.max_tokens is None

    speech_req = SpeechRequest(model="zonos3", input="Hello")
    assert not hasattr(speech_req, "byte_tokenize_all")
    assert speech_req.voice is None and speech_req.response_format == "pcm"


def test_tts_request_language_normalization():
    assert api_server._normalize_tts_request_language("EN-US") == "en_us"
    assert api_server._normalize_tts_request_language("cmn") == "cmn"
    with pytest.raises(ValueError, match="Unsupported language"):
        api_server._normalize_tts_request_language("klingon")


def test_byte_tokenizer_uses_training_byte_offset():
    assert BYTE_TEXT_VOCAB_SIZE == 448
    assert text_to_byte_ids("Az") == [2, 192 + ord("A"), 192 + ord("z"), 3]


def test_tts_max_tokens_default_resolves_to_model_max():
    assert api_server._resolve_tts_max_tokens(_MaxTokenConfig(), None) == 4096
    assert api_server._resolve_tts_max_tokens(_MaxTokenConfig(), 9999) == 4096
    assert api_server._resolve_tts_max_tokens(_MaxTokenConfig(), 256) == 256

    with pytest.raises(ValueError, match="positive"):
        api_server._resolve_tts_max_tokens(_MaxTokenConfig(), 0)


def test_zonos3_sidecar_loads_speaking_rate_conditioning(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """
model:
  n_layers: 1
  dim: 64
  head_dim: 32
  ffn_dim_multiplier: 1.0
  max_seqlen: 512
  text_vocab: 999
data:
  speaking_rate_enabled: true
  speaking_rate_buckets: ["0-10", "10+"]
""",
        encoding="utf-8",
    )

    cfg = cached_load_checkpoint_config(str(tmp_path))

    assert cfg.speaking_rate_num_buckets == 2


def test_zonos3_sidecar_preserves_checkpoint_text_vocab(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """
model:
  n_layers: 1
  dim: 64
  head_dim: 32
  ffn_dim_multiplier: 1.0
  max_seqlen: 512
  text_vocab: 469
  speaking_rate_num_buckets: 2
data:
  speaking_rate_enabled: true
  speaking_rate_buckets: ["0-10", "10+"]
""",
        encoding="utf-8",
    )

    cfg = cached_load_checkpoint_config(str(tmp_path))

    assert cfg.text_vocab == 469
    assert cfg.speaking_rate_num_buckets == 2
