import copy
import json
from pathlib import Path

import pytest
from zonos3.tokenizer import matched_codec
from zonos3.utils.release import hf_join, resolve_file

CAMPAIGNS = {
    "keep2_plain": ("config_full_unfreezeall_mr_drop30_4x10.json", "autoencoder/heads/keep2_plain"),
    "full_mr": ("config_full_unfreezeall_mr.json", "autoencoder/heads/full_mr"),
}


def test_registry_declares_the_istft_decoders():
    for codec in matched_codec.codec_names():
        assert matched_codec.decoder_ids(codec) == ["keep2_plain", "full_mr"]  # the default first
        for name in matched_codec.decoder_ids(codec):
            spec = matched_codec.decoder_spec(codec, name)
            assert spec["kind"] == "istft_mr"
            assert spec["config"] == CAMPAIGNS[name][0] and spec["campaign"].endswith(CAMPAIGNS[name][1])
            config = Path(resolve_file(hf_join(spec["campaign"], spec["config"])))
            assert config.is_file() and Path(resolve_file(hf_join(spec["campaign"], spec["checkpoint"]))).is_file()
            cfg = json.loads(config.read_text())
            assert cfg.get("decoder", {}).get("type") != "original"  # never the conv tail
            assert cfg["init"]["generator_checkpoint"] == matched_codec.entry(codec)["checkpoint"]
            assert matched_codec.decoder_headroom_db(codec, name) == 3.0
            info = matched_codec.decoder_info(codec, name)
            assert info["id"] == name and info["expect_step"] == 50000 and info["causal"]
        assert matched_codec.decoder_info(codec) == matched_codec.decoder_info(codec, "keep2_plain")


def test_decoders_are_keyed_by_id():
    codec = next(iter(matched_codec.codec_names()))
    assert matched_codec.decoder_id(codec) == "keep2_plain" and matched_codec.decoder_id(codec, "full_mr") == "full_mr"
    with pytest.raises(ValueError):
        matched_codec.decoder_spec(codec, "nope")
    assert matched_codec._key(codec, None) == matched_codec._key(codec, "keep2_plain")
    assert matched_codec._key(codec, None) != matched_codec._key(codec, "full_mr")


def test_served_decoders():
    codec = next(iter(matched_codec.codec_names()))
    assert matched_codec.served_decoders(codec) == ["keep2_plain", "full_mr"]
    assert matched_codec.served_decoders(codec, ["full_mr", "keep2_plain", "full_mr"]) == ["keep2_plain", "full_mr"]
    assert matched_codec.served_decoders(codec, ["full_mr"]) == ["full_mr"]
    for wanted in (["nope"], [], [" "]):
        with pytest.raises(ValueError):
            matched_codec.served_decoders(codec, wanted)


def test_a_codec_without_the_decoder_block_is_refused(monkeypatch):
    reg = copy.deepcopy(matched_codec.registry())  # the module's registry is parsed once and shared
    for value in reg.values():
        value.pop("decoders", None)
    monkeypatch.setattr(matched_codec, "registry", lambda: reg)
    codec = next(iter(matched_codec.codec_names()))
    with pytest.raises(RuntimeError):
        matched_codec.decoder_spec(codec)
    with pytest.raises(RuntimeError):
        matched_codec.load_model(codec, "cpu")


def test_no_conv_tail_code_path_left():
    source = Path(matched_codec.__file__).read_text()
    assert "MossCodec" not in source and "decode_tokens" not in source
