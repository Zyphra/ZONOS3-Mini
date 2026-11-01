"""Phoneme-token checkpoints: open/phone/close text ids after the padding row, as the training loader."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("tokenizers")

from test_superbpe_convention import spec  # noqa: E402,F401  (fixture)
from zonos3.tts.superbpe import load_spec, text_ids, validate_spec  # noqa: E402

O, C = "\ue000", "\ue001"
TEXT_VOCAB = 3 + 4


def _phonemes(symbols="kæt"):
    return dict(open_id=TEXT_VOCAB + 1, close_id=TEXT_VOCAB + 2, symbols=list(symbols),
                open_marker=O, close_marker=C, inventory_sha256="0" * 64)


def test_groups_encode_as_open_phones_close(spec):
    phoneme_spec = {**spec, "phonemes": _phonemes()}
    k, ae, t = TEXT_VOCAB + 3, TEXT_VOCAB + 4, TEXT_VOCAB + 5
    assert text_ids(f"hello {O}kæt{C} world", phoneme_spec) == [2, 4, TEXT_VOCAB + 1, k, ae, t, TEXT_VOCAB + 2, 5, 3]
    assert text_ids("hello world", phoneme_spec) == text_ids("hello world", spec)
    for bad in (f"{O}kæt", f"kæt{C}", f"{O}{C}", f"{O}kx{C}"):
        with pytest.raises(ValueError):
            text_ids(bad, phoneme_spec)


def test_phoneme_ids_must_follow_the_padding_row_and_match_the_extra_rows(spec, tmp_path):
    validate_spec({**spec, "phonemes": _phonemes()}, TEXT_VOCAB)
    with pytest.raises(ValueError, match="padding row"):
        validate_spec({**spec, "phonemes": {**_phonemes(), "open_id": TEXT_VOCAB}}, TEXT_VOCAB)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for extra, ok in ((5, True), (4, False)):
        (model_dir / "params.json").write_text(json.dumps({"model": {
            "text_tokenizer_spec": {**spec, "phonemes": _phonemes()}, "text_vocab": TEXT_VOCAB, "text_extra_tokens": extra}}))
        if ok:
            assert load_spec(model_dir)["phonemes"]["symbols"] == list("kæt")
        else:
            with pytest.raises(ValueError, match="text_extra_tokens"):
                load_spec(model_dir)


def test_phoneme_rows_are_not_edit_marker_rows(spec):
    """Edit checkpoints and phoneme checkpoints share text_extra_tokens: only rows the phoneme
    tokens do not claim are edit markers (edit_max_spans), so a phoneme checkpoint cannot edit."""
    from zonos3.tts.edit import edit_marker_rows

    assert edit_marker_rows(3, None) == 3                       # edit checkpoint: three marker rows
    assert edit_marker_rows(3, spec) == 3                       # SuperBPE spec without phonemes
    assert edit_marker_rows(5, {**spec, "phonemes": _phonemes()}) == 0


def _merged_spec(spec, markers):
    phonemes = _phonemes()
    return {**spec, "phonemes": {**phonemes, "open_id": phonemes["open_id"] + markers,
                                 "close_id": phonemes["close_id"] + markers}}


def test_phoneme_block_after_edit_markers(spec, tmp_path):
    """Edit model + phoneme rows: markers keep text_vocab + 1 + j, the phoneme block follows them
    and must be the last extra rows; groups encode with the shifted ids."""
    from zonos3.tts.edit import edit_marker_rows

    merged = _merged_spec(spec, 3)
    k, ae, t = TEXT_VOCAB + 6, TEXT_VOCAB + 7, TEXT_VOCAB + 8
    assert text_ids(f"{O}kæt{C}", merged) == [2, TEXT_VOCAB + 4, k, ae, t, TEXT_VOCAB + 5, 3]
    assert edit_marker_rows(3 + 5, merged) == 3
    with pytest.raises(ValueError, match="last extra"):
        edit_marker_rows(3 + 6, merged)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for extra, ok in ((8, True), (9, False), (5, False)):
        (model_dir / "params.json").write_text(json.dumps({"model": {
            "text_tokenizer_spec": merged, "text_vocab": TEXT_VOCAB, "text_extra_tokens": extra}}))
        if ok:
            assert load_spec(model_dir)["phonemes"]["open_id"] == TEXT_VOCAB + 4
        else:
            with pytest.raises(ValueError, match="text_extra_tokens"):
                load_spec(model_dir)
    with pytest.raises(ValueError, match="padding row"):
        validate_spec({**merged, "phonemes": {**merged["phonemes"], "close_id": TEXT_VOCAB + 6}}, TEXT_VOCAB)


def test_add_text_extra_rows_appends_after_existing_rows(spec, tmp_path):
    """A base with edit marker rows keeps them; the phoneme rows and ids go after them."""
    import importlib.util
    from pathlib import Path

    import torch

    script = Path(__file__).resolve().parents[2] / "scripts" / "add_text_extra_rows.py"
    module = importlib.util.spec_from_file_location("add_text_extra_rows", script)
    add = importlib.util.module_from_spec(module)
    module.loader.exec_module(add)
    base = tmp_path / "base"
    base.mkdir()
    table = torch.arange((TEXT_VOCAB + 1 + 3) * 2, dtype=torch.float32).reshape(-1, 2)   # 3 marker rows
    torch.save({"multi_embedder.embedders.1.weight": table, "other": torch.ones(3)}, base / "model.pth")
    (base / "params.json").write_text(json.dumps({"model": {
        "n_codebooks": 1, "text_vocab": TEXT_VOCAB, "text_extra_tokens": 3, "text_tokenizer_spec": dict(spec)}}))
    rows = torch.full((5, 2), -1.0)
    torch.save({"format": "zonos-text-extra/1", "text_vocab": TEXT_VOCAB, "rows": rows, "phonemes": _phonemes()},
               tmp_path / "rows.pt")
    out = add.add_text_extra_rows(base, tmp_path / "rows.pt", tmp_path / "out")
    model = json.loads((out / "params.json").read_text())["model"]
    assert model["text_extra_tokens"] == 8
    assert (model["text_tokenizer_spec"]["phonemes"]["open_id"], model["text_tokenizer_spec"]["phonemes"]["close_id"]) \
        == (TEXT_VOCAB + 4, TEXT_VOCAB + 5)
    merged = torch.load(out / "model.pth", weights_only=True)["multi_embedder.embedders.1.weight"]
    assert torch.equal(merged[: len(table)], table) and torch.equal(merged[len(table):], rows)
    assert json.loads((out / "text_extra.json").read_text())["first_row"] == TEXT_VOCAB + 4
