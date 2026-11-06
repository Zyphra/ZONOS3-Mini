"""Extra text rows (lingua ``apps.zonos.text_extra``, ``zonos-text-extra/1``), e.g. phoneme tokens.

The rows were trained on a frozen base (``text_extra_only``), so a model gains them by appending them to its text
table, after the padding row and after any extra rows it already has (a speech-edit vocabulary's marker rows stay at
``text_vocab + 1 + j``). Text that never reads them runs exactly as before. Phoneme rows also add
``text_tokenizer_spec.phonemes`` (group markers, ids, symbols) with ids shifted by the rows already there.

``tools/add_text_extra_rows.py`` applies them to a checkpoint on disk; the servers and the reference apply them to a
loaded model (``--phoneme-rows``; the Space with the release rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

FORMAT = "zonos-text-extra/1"
_PHONEME_KEYS = ("open_id", "close_id", "symbols", "open_marker", "close_marker", "inventory_sha256")


@dataclass(frozen=True)
class TextExtraRows:
    """One parsed export: float32 ``rows`` [n, dim] trained after a ``text_vocab`` + padding table, and the phoneme
    block they encode (ids relative to a table without other extra rows), if any."""

    rows: torch.Tensor
    text_vocab: int
    phonemes: dict | None
    run: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "TextExtraRows":
        extra = torch.load(path, map_location="cpu", weights_only=True)
        if extra.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} export")
        text_vocab = int(extra["text_vocab"])
        phonemes = None
        if "phonemes" in extra:
            phonemes = {k: extra["phonemes"][k] for k in _PHONEME_KEYS}
            if (phonemes["open_id"], phonemes["close_id"]) != (text_vocab + 1, text_vocab + 2):
                raise ValueError(f"{path}: exported phoneme rows must start at the padding row + 1")
            if 2 + len(phonemes["symbols"]) != extra["rows"].shape[0]:
                raise ValueError(f"{path}: {extra['rows'].shape[0]} rows for {len(phonemes['symbols'])} phoneme symbols")
        return cls(extra["rows"].float(), text_vocab, phonemes, extra.get("run"))

    @property
    def count(self) -> int:
        return int(self.rows.shape[0])

    def spec(self, spec: dict | None, *, text_vocab: int, existing: int) -> dict | None:
        """``spec`` (a ``text_tokenizer_spec``, or None) of a model with ``existing`` extra rows once these rows follow
        them: phoneme rows add ``phonemes`` with their ids shifted by ``existing``. Raises ValueError for a model the
        rows do not fit."""
        if text_vocab != self.text_vocab:
            raise ValueError(f"text_vocab {text_vocab} != the rows' {self.text_vocab}")
        if self.phonemes is None:
            return spec
        if spec is None:
            raise ValueError("phoneme rows need a SuperBPE text_tokenizer_spec")
        if spec.get("phonemes") is not None:
            raise ValueError("the model already has phoneme tokens")
        phonemes = dict(self.phonemes, open_id=self.phonemes["open_id"] + existing,
                        close_id=self.phonemes["close_id"] + existing)
        return {**spec, "phonemes": phonemes}

    def append(self, table: torch.Tensor) -> torch.Tensor:
        """``table`` (the text embedder weight) with these rows after it, in its dtype and device."""
        if table.shape[1] != self.rows.shape[1]:
            raise ValueError(f"rows of width {self.rows.shape[1]} for a text table of width {table.shape[1]}")
        return torch.cat((table, self.rows.to(table)), dim=0).contiguous()
