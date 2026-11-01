"""The matched codec decodes only through the causal iSTFT head declared in the registry (no conv-tail fallback)."""
import json
from pathlib import Path

import pytest

from zonos3.tokenizer import matched_codec
from zonos3.utils.release import hf_join, resolve_file


def test_registry_declares_the_istft_decoder():
    for codec in matched_codec.codec_names():
        spec = matched_codec.decoder_spec(codec)
        assert spec["kind"] == "istft_mr"
        config = Path(resolve_file(hf_join(spec["campaign"], spec["config"])))
        assert config.is_file() and Path(resolve_file(hf_join(spec["campaign"], spec["checkpoint"]))).is_file()
        cfg = json.loads(config.read_text())
        assert cfg.get("decoder", {}).get("type") != "original"          # never the conv tail
        assert cfg["init"]["generator_checkpoint"] == matched_codec.entry(codec)["checkpoint"]
        assert matched_codec.decoder_headroom_db(codec) == 3.0
        info = matched_codec.decoder_info(codec)
        assert info["expect_step"] == 50000 and info["causal"]


def test_a_codec_without_the_decoder_block_is_refused(monkeypatch):
    reg = matched_codec.registry()
    for value in reg.values():
        value.pop("decoder", None)
    monkeypatch.setattr(matched_codec, "registry", lambda: reg)
    codec = next(iter(matched_codec.codec_names()))
    with pytest.raises(RuntimeError, match="conv-tail decoder is disabled"):
        matched_codec.decoder_spec(codec)
    with pytest.raises(RuntimeError):
        matched_codec.load_model(codec, "cpu")


def test_no_conv_tail_code_path_left():
    source = Path(matched_codec.__file__).read_text()
    assert "MossCodec" not in source and "decode_tokens" not in source
