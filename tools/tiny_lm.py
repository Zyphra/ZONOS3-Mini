"""Test helpers for tiny random LM checkpoints and tokenizers (tools/tests, servers/mlx/tests, servers/ggml/tests/parity,
model/tests): the checkpoint directory layout `convert_gguf.py lm` and `Zonos3LM.load` read, the pinned tokenizer of a
GGUF fixture, and WordLevel stand-ins for the SuperBPE tokenizer."""
from __future__ import annotations

import json
from pathlib import Path

import torch

TINY_TOKENIZER_JSON = '{"model": {"type": "BPE"}}'


def write_lm_dir(path: Path, params: dict, state: dict[str, torch.Tensor], *, tokenizer: str = TINY_TOKENIZER_JSON,
                 silence: list[int] | None = None) -> Path:
    """Write ``model.pth``, ``params.json`` (``{"model": params}``), ``tokenizer.json`` and, with ``silence``, the
    silence-frame JSON ``silence.json`` (codec ``params["audio_codec"]``) into ``path``; returns ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, path / "model.pth")
    (path / "params.json").write_text(json.dumps({"model": params}))
    (path / "tokenizer.json").write_text(tokenizer)
    if silence is not None:
        (path / "silence.json").write_text(json.dumps({"codec": params["audio_codec"], "frame": list(silence)}))
    return path


def fixture_tokenizer(fixture: Path) -> str:
    """The tokenizer JSON embedded in a zonos3 GGUF (field ``zonos3.tokenizer.json``)."""
    from gguf import GGUFReader

    field = GGUFReader(fixture).fields["zonos3.tokenizer.json"]
    return bytes(field.parts[field.data[0]]).decode()


def wordlevel_spec(vocab: list[str], *, whitespace: bool = False) -> tuple[str, int]:
    """A WordLevel tokenizer over ``vocab`` (ids in list order, unknown words -> ``[UNK]``, which ``vocab`` must hold),
    split on whitespace and punctuation when ``whitespace``: its JSON and vocabulary size."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    ids = {w: i for i, w in enumerate(vocab)}
    tokenizer = Tokenizer(models.WordLevel(ids, unk_token="[UNK]"))
    if whitespace:
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer.to_str(), len(ids)
