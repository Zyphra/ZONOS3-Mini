"""Phoneme-token checkpoints: open/phone/close text ids after the padding row, as the training loader; the start and
pause tokens after them."""

from __future__ import annotations

import json

import add_text_extra_rows
import pytest
import torch

pytest.importorskip("tokenizers")

import test_superbpe_convention  # noqa: E402
from zonos3.tts.superbpe import load_spec, text_ids, validate_spec  # noqa: E402

superbpe_spec = test_superbpe_convention.spec  # the SuperBPE spec fixture

O, C = "\ue000", "\ue001"
S, P1, P2 = "\ue002", "\ue003", "\ue004"
TEXT_VOCAB = 3 + 4


def _phonemes(symbols="kæt"):
    return {
        "open_id": TEXT_VOCAB + 1,
        "close_id": TEXT_VOCAB + 2,
        "symbols": list(symbols),
        "open_marker": O,
        "close_marker": C,
        "inventory_sha256": "0" * 64,
    }


def test_groups_encode_as_open_phones_close(superbpe_spec):
    phoneme_spec = {**superbpe_spec, "phonemes": _phonemes()}
    k, ae, t = TEXT_VOCAB + 3, TEXT_VOCAB + 4, TEXT_VOCAB + 5
    assert text_ids(f"hello {O}kæt{C} world", phoneme_spec) == [2, 4, TEXT_VOCAB + 1, k, ae, t, TEXT_VOCAB + 2, 5, 3]
    assert text_ids("hello world", phoneme_spec) == text_ids("hello world", superbpe_spec)
    for bad in (f"{O}kæt", f"kæt{C}", f"{O}{C}", f"{O}kx{C}"):
        with pytest.raises(ValueError):
            text_ids(bad, phoneme_spec)


def test_phoneme_ids_must_follow_the_padding_row_and_match_the_extra_rows(superbpe_spec, tmp_path):
    validate_spec({**superbpe_spec, "phonemes": _phonemes()}, TEXT_VOCAB)
    with pytest.raises(ValueError):
        validate_spec({**superbpe_spec, "phonemes": {**_phonemes(), "open_id": TEXT_VOCAB}}, TEXT_VOCAB)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for extra, ok in ((5, True), (4, False)):
        (model_dir / "params.json").write_text(
            json.dumps(
                {
                    "model": {
                        "text_tokenizer_spec": {**superbpe_spec, "phonemes": _phonemes()},
                        "text_vocab": TEXT_VOCAB,
                        "text_extra_tokens": extra,
                    }
                }
            )
        )
        if ok:
            assert load_spec(model_dir)["phonemes"]["symbols"] == list("kæt")
        else:
            with pytest.raises(ValueError):
                load_spec(model_dir)


def test_phoneme_rows_are_not_edit_marker_rows(superbpe_spec):
    """Edit checkpoints and phoneme checkpoints share text_extra_tokens: only rows the phoneme
    tokens do not claim are edit markers (edit_max_spans), so a phoneme checkpoint cannot edit."""
    from zonos3.tts.edit import edit_marker_rows

    assert edit_marker_rows(3, None) == 3  # edit checkpoint: three marker rows
    assert edit_marker_rows(3, superbpe_spec) == 3  # SuperBPE spec without phonemes
    assert edit_marker_rows(5, {**superbpe_spec, "phonemes": _phonemes()}) == 0


def _merged_spec(base, markers):
    phonemes = _phonemes()
    return {
        **base,
        "phonemes": {**phonemes, "open_id": phonemes["open_id"] + markers, "close_id": phonemes["close_id"] + markers},
    }


def test_phoneme_block_after_edit_markers(superbpe_spec, tmp_path):
    """Edit model + phoneme rows: markers keep text_vocab + 1 + j, the phoneme block follows them
    and must be the last extra rows; groups encode with the shifted ids."""
    from zonos3.tts.edit import edit_marker_rows

    merged = _merged_spec(superbpe_spec, 3)
    k, ae, t = TEXT_VOCAB + 6, TEXT_VOCAB + 7, TEXT_VOCAB + 8
    assert text_ids(f"{O}kæt{C}", merged) == [2, TEXT_VOCAB + 4, k, ae, t, TEXT_VOCAB + 5, 3]
    assert edit_marker_rows(3 + 5, merged) == 3
    with pytest.raises(ValueError):
        edit_marker_rows(3 + 6, merged)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for extra, ok in ((8, True), (9, False), (5, False)):
        (model_dir / "params.json").write_text(
            json.dumps({"model": {"text_tokenizer_spec": merged, "text_vocab": TEXT_VOCAB, "text_extra_tokens": extra}})
        )
        if ok:
            assert load_spec(model_dir)["phonemes"]["open_id"] == TEXT_VOCAB + 4
        else:
            with pytest.raises(ValueError):
                load_spec(model_dir)
    with pytest.raises(ValueError):
        validate_spec({**merged, "phonemes": {**merged["phonemes"], "close_id": TEXT_VOCAB + 6}}, TEXT_VOCAB)


def _block_spec(base, markers=0, text_vocab=TEXT_VOCAB):
    """kæt phoneme rows, then the start row, then two pause rows, after ``markers`` edit marker rows."""
    first = text_vocab + 1 + markers
    return {
        **base,
        "phonemes": {**_phonemes(), "open_id": first, "close_id": first + 1},
        "start": {"id": first + 5, "marker": S},
        "pauses": {"ids": [first + 6, first + 7], "markers": [P1, P2], "seconds": [0.25, 0.5]},
    }


def test_start_and_pause_tokens_follow_the_phoneme_block(superbpe_spec, tmp_path):
    """The start row follows the phoneme rows and the pause rows follow it; the start marker is one id first in the
    text, each pause marker one id between the stretches it splits, all of them the last extra rows."""
    from zonos3.tts.edit import edit_marker_rows

    spec = _block_spec(superbpe_spec)
    validate_spec(spec, TEXT_VOCAB)
    assert text_ids(f"{S}hello{P2} world", spec) == [2, TEXT_VOCAB + 6, 4, TEXT_VOCAB + 8, 5, 3]
    for text, message in (
        (f"hello {S}world", "first character"),
        (f"{S}hello{S}", "first character"),
        ("hello\ue005 world", "outside the 2 pause buckets"),
    ):
        with pytest.raises(ValueError, match=message):
            text_ids(text, spec)
    no_start = {k: v for k, v in spec.items() if k != "start"}
    no_pauses = {k: v for k, v in spec.items() if k != "pauses"}
    with pytest.raises(ValueError, match="no start token"):
        text_ids(f"{S}hello", no_start)
    with pytest.raises(ValueError, match="no pause tokens"):
        text_ids(f"hello{P2} world", no_pauses)
    with pytest.raises(ValueError, match="Start token ID must follow the phoneme rows"):
        validate_spec({**spec, "start": {**spec["start"], "id": TEXT_VOCAB + 7}}, TEXT_VOCAB)
    with pytest.raises(ValueError, match="Invalid pause markers"):
        validate_spec({**spec, "pauses": {**spec["pauses"], "markers": [P2, P1]}}, TEXT_VOCAB)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for extra, ok in ((8, True), (7, False), (9, False)):
        (model_dir / "params.json").write_text(
            json.dumps({"model": {"text_tokenizer_spec": spec, "text_vocab": TEXT_VOCAB, "text_extra_tokens": extra}})
        )
        if ok:
            assert load_spec(model_dir)["pauses"]["ids"] == [TEXT_VOCAB + 7, TEXT_VOCAB + 8]
        else:
            with pytest.raises(ValueError, match="do not match text_extra_tokens"):
                load_spec(model_dir)
    assert edit_marker_rows(3 + 8, _block_spec(superbpe_spec, 3)) == 3


def test_start_immediately_puts_the_start_id_first_without_prefill(tmp_path):
    """start_immediately: the prompt text opens with the start marker (once, also when typed) and the silence prefill
    defaults to off; an explicit silence_prefill_frames still applies."""
    import test_request_rules
    from tiny_lm import wordlevel_spec
    from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig
    from zonos3.utils.hashing import sha256_file

    # a vocabulary past the byte range, which the prompt builder requires
    tokenizer, size = wordlevel_spec(["hello", "world", "[UNK]", *(f"w{i}" for i in range(445))], whitespace=True)
    (tmp_path / "superbpe.json").write_text(tokenizer)
    text_vocab = size + 4
    base = {
        "kind": "superbpe",
        "path": str(tmp_path / "superbpe.json"),
        "sha256": sha256_file(tmp_path / "superbpe.json"),
        "vocab_size": size,
        "offset": 4,
        "bos": 2,
        "eos": 3,
    }
    spec = _block_spec(base, text_vocab=text_vocab)
    validate_spec(spec, text_vocab)

    class Context(test_request_rules.Context):
        text_tokenizer_spec = spec

        def __init__(self):
            self.text_vocab = text_vocab
            super().__init__()
            self._builder = TTSPromptBuilder(
                TTSPromptConfig(
                    n_codebooks=test_request_rules.N_CB,
                    codebook_sizes=(test_request_rules.CODEBOOK,) * test_request_rules.N_CB,
                    audio_pad_id=self.audio_pad_id,
                    text_vocab=text_vocab,
                    text_tokenizer_spec=spec,
                )
            )

    ctx = Context()
    want = [2, text_vocab + 6, 4, 5, 3]
    for request, prefill in (
        ({"text": "hello world", "start_immediately": True}, 0),
        ({"text": "hello world", "start_immediately": True, "silence_prefill_frames": 2}, 2),
        ({"text": f"{S}hello world"}, 0),
        ({"text": f"{S}hello world", "start_immediately": True}, 0),
    ):
        p = test_request_rules.run(request, ctx)
        assert p.text == f"{S}hello world"
        assert (0 if p.prefill_rows is None else len(p.prefill_rows)) == prefill
        assert ctx.prompt_builder().text_ids(p.text) == want
        column = p.cond.input_ids[:, test_request_rules.N_CB].tolist()
        assert any(column[i : i + len(want)] == want for i in range(len(column)))


def test_add_text_extra_rows_appends_after_existing_rows(superbpe_spec, tmp_path):
    """A base with edit marker rows keeps them; the phoneme rows and ids go after them."""
    base = tmp_path / "base"
    base.mkdir()
    table = torch.arange((TEXT_VOCAB + 1 + 3) * 2, dtype=torch.float32).reshape(-1, 2)  # 3 marker rows
    torch.save({"multi_embedder.embedders.1.weight": table, "other": torch.ones(3)}, base / "model.pth")
    (base / "params.json").write_text(
        json.dumps(
            {
                "model": {
                    "n_codebooks": 1,
                    "text_vocab": TEXT_VOCAB,
                    "text_extra_tokens": 3,
                    "text_tokenizer_spec": dict(superbpe_spec),
                }
            }
        )
    )
    rows = torch.full((5, 2), -1.0)
    torch.save(
        {"format": "zonos-text-extra/1", "text_vocab": TEXT_VOCAB, "rows": rows, "phonemes": _phonemes()},
        tmp_path / "rows.pt",
    )
    out = add_text_extra_rows.add_text_extra_rows(base, tmp_path / "rows.pt", tmp_path / "out")
    model = json.loads((out / "params.json").read_text())["model"]
    assert model["text_extra_tokens"] == 8
    assert (
        model["text_tokenizer_spec"]["phonemes"]["open_id"],
        model["text_tokenizer_spec"]["phonemes"]["close_id"],
    ) == (TEXT_VOCAB + 4, TEXT_VOCAB + 5)
    merged = torch.load(out / "model.pth", weights_only=True)["multi_embedder.embedders.1.weight"]
    assert torch.equal(merged[: len(table)], table) and torch.equal(merged[len(table) :], rows)
    assert json.loads((out / "text_extra.json").read_text())["first_row"] == TEXT_VOCAB + 4


def test_runtime_rows_follow_the_edit_vocabulary_and_leave_plain_text_alone(superbpe_spec, tmp_path):
    """Zonos3LM.append_text_rows (the servers' --phoneme-rows) on a model that already has 3 edit marker rows: the
    rows follow them, the served spec encodes groups with the shifted ids, text without groups gives the same logits,
    and the result equals tools/add_text_extra_rows.py's checkpoint."""
    from test_reference_lm import N_CB, TINY, tiny_model
    from zonos3.models.text_extra import TextExtraRows
    from zonos3.reference.lm import LMCache, Zonos3LM

    text_vocab = superbpe_spec["vocab_size"] + 4
    params = {**TINY, "text_vocab": text_vocab, "text_extra_tokens": 3, "text_tokenizer_spec": dict(superbpe_spec)}
    base = tmp_path / "base"
    base.mkdir()
    torch.save(
        tiny_model(text_vocab=text_vocab, text_extra_tokens=3, text_tokenizer_spec=dict(superbpe_spec)).state_dict(),
        base / "model.pth",
    )
    (base / "params.json").write_text(json.dumps({"model": params}))
    phonemes = {
        "open_id": text_vocab + 1,
        "close_id": text_vocab + 2,
        "symbols": list("kæt"),
        "open_marker": O,
        "close_marker": C,
        "inventory_sha256": "0" * 64,
    }
    start = {"id": text_vocab + 6, "marker": S}
    pauses = {"ids": [text_vocab + 7, text_vocab + 8], "markers": [P1, P2], "seconds": [0.25, 0.5]}
    rows = torch.randn(8, TINY["dim"], generator=torch.Generator().manual_seed(1))
    torch.save(
        {
            "format": "zonos-text-extra/1",
            "text_vocab": text_vocab,
            "rows": rows,
            "phonemes": phonemes,
            "start": start,
            "pauses": pauses,
        },
        tmp_path / "rows.pt",
    )

    def prompt(ids):
        out = torch.full((len(ids), N_CB + 1), TINY["audio_pad_id"], dtype=torch.long)
        out[:, N_CB] = torch.tensor(ids)
        return out

    plain_model = Zonos3LM.load(base)
    grown = Zonos3LM.load(base).append_text_rows(TextExtraRows.load(tmp_path / "rows.pt"))
    served = grown.config.text_tokenizer_spec
    assert (served["phonemes"]["open_id"], grown.config.text_extra_tokens) == (text_vocab + 4, 11)
    assert served["start"]["id"] == text_vocab + 9
    assert served["pauses"]["ids"] == [text_vocab + 10, text_vocab + 11]
    plain = prompt(text_ids("hello world", served))
    assert torch.equal(grown(plain, LMCache()).logits, plain_model(plain, LMCache()).logits)
    group = prompt(text_ids(f"hello {O}kæt{C}", served))
    assert {text_vocab + 4, text_vocab + 5} <= set(group[:, N_CB].tolist())

    merged = Zonos3LM.load(add_text_extra_rows.add_text_extra_rows(base, tmp_path / "rows.pt", tmp_path / "merged"))
    assert merged.config.text_tokenizer_spec["phonemes"] == served["phonemes"]
    assert merged.config.text_tokenizer_spec["start"] == served["start"]
    assert merged.config.text_tokenizer_spec["pauses"] == served["pauses"]
    assert torch.equal(merged(group, LMCache()).logits, grown(group, LMCache()).logits)
