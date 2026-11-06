#!/usr/bin/env python3
"""Append trained extra text rows (lingua apps.zonos.text_extra, zonos-text-extra/1) to a
converted checkpoint, e.g. the release LM:

    python tools/add_text_extra_rows.py --base $ZONOS3_RELEASE_LM \
        --rows <phoneme_tokens.pt> \
        --out checkpoints/z3m_mect60k_phoneme_tokens

The rows were trained on a frozen base (text_extra_only), so the output is the base checkpoint
with a longer text table: the rows follow the padding row, params.json gains
``text_extra_tokens`` and, for phoneme tokens, ``text_tokenizer_spec.phonemes``. Every other
tensor and file is copied unchanged.

A base that already has extra rows (the speech-edit checkpoints' marker rows) keeps them at their
ids; the new rows go after them and the phoneme ids shift by their count:

    python tools/add_text_extra_rows.py --base checkpoints/z3m_mect_edit \
        --rows .../phoneme_tokens_s15000.pt --out .../merged_z3m_mect_edit
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from zonos3.models.text_extra import TextExtraRows
from zonos3.utils.hashing import sha256_file


def add_text_extra_rows(base: Path, rows_path: Path, out: Path) -> Path:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    extra = TextExtraRows.load(rows_path)
    params = json.loads((base / "params.json").read_text())
    model = params["model"]
    text_vocab, existing = int(model["text_vocab"]), int(model.get("text_extra_tokens", 0))
    spec = extra.spec(model.get("text_tokenizer_spec"), text_vocab=text_vocab, existing=existing)
    state = torch.load(base / "model.pth", map_location="cpu", weights_only=True)
    key = f"multi_embedder.embedders.{int(model['n_codebooks'])}.weight"
    table, rows = state[key], extra.rows
    if table.shape != (text_vocab + 1 + existing, rows.shape[1]):
        raise ValueError(f"unexpected text table {tuple(table.shape)} for {existing} extra rows and rows {tuple(rows.shape)}")
    state[key] = extra.append(table)
    model["text_extra_tokens"] = existing + extra.count
    if spec is not None:
        model["text_tokenizer_spec"] = spec

    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for item in base.iterdir():
        if item.name in ("model.pth", "params.json"):
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, staging / item.name)
    torch.save(state, staging / "model.pth")
    (staging / "params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n")
    (staging / "text_extra.json").write_text(json.dumps({
        "base": str(base), "base_model_sha256": sha256_file(base / "model.pth"),
        "rows": str(rows_path), "rows_sha256": sha256_file(rows_path), "rows_run": extra.run,
        "text_extra_tokens": int(rows.shape[0]), "base_text_extra_tokens": existing,
        "first_row": text_vocab + 1 + existing, "table_key": key,
    }, indent=1) + "\n")
    back = torch.load(staging / "model.pth", map_location="cpu", weights_only=True)
    if not torch.equal(back[key][: table.shape[0]], table) or not torch.equal(back[key][table.shape[0]:], rows.to(table.dtype)):
        raise ValueError("merged text table read-back mismatch")
    if out.exists():
        out.rmdir()
    staging.rename(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="converted checkpoint directory (model.pth, params.json)")
    parser.add_argument("--rows", type=Path, required=True, help="zonos-text-extra/1 export")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(add_text_extra_rows(args.base, args.rows, args.out))
