"""SuperBPE text ids: BOS 2, payload ids + 4, EOS 3, pinned to the training artifact by SHA-256."""
from __future__ import annotations

import json

import pytest

tokenizers = pytest.importorskip("tokenizers")

from tiny_lm import wordlevel_spec  # noqa: E402
from zonos3.tts.superbpe import load_spec, text_ids, validate_spec  # noqa: E402
from zonos3.utils.hashing import sha256_file  # noqa: E402


@pytest.fixture
def spec(tmp_path):
    tokenizer, size = wordlevel_spec(["hello", "world", "[UNK]"], whitespace=True)
    path = tmp_path / "superbpe.json"
    path.write_text(tokenizer)
    return {"kind": "superbpe", "path": str(path), "sha256": sha256_file(path), "vocab_size": size, "offset": 4, "bos": 2, "eos": 3}


def test_text_ids_follow_the_training_convention(spec):
    assert text_ids("hello world", spec) == [2, 0 + 4, 1 + 4, 3]
    assert text_ids("", spec) == [2, 3]


def test_validate_spec_checks_the_id_convention_and_vocab(spec):
    validate_spec(spec, text_vocab=3 + 4)
    with pytest.raises(ValueError):
        validate_spec(spec, text_vocab=3 + 5)
    with pytest.raises(ValueError):
        validate_spec({**spec, "offset": 0}, text_vocab=3 + 4)


def test_sha_mismatch_is_refused(spec, tmp_path):
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(open(spec["path"], "rb").read() + b" ")
    with pytest.raises(ValueError):
        text_ids("hello", {**spec, "path": str(tampered)})


def test_load_spec_reads_params_json(spec, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "params.json").write_text(json.dumps({"model": {"text_tokenizer_spec": spec, "text_vocab": 7}}))
    assert load_spec(model_dir) == spec

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "params.json").write_text(json.dumps({"model": {"audio_codec": "opus_dual_3n_600k_sched600k_r4096_step600000", "text_vocab": 448}}))
    assert load_spec(legacy) is None  # byte-level text, no spec needed
