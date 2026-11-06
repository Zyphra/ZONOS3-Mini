"""Parity of src/zonos3-edit.cpp (via the edit-replay binary) with model/src/zonos3/tts/edit.py: the edit prompt
layout the Python host's fill calls build.

Every case runs the same call through ``edit-replay`` ({"fn", "args"} JSON on stdin) and through the Python module;
results must be equal. ``test_fixture_file_is_current`` keeps tests/fixtures/edit-replay/cases.json (the ctest replay)
equal to the Python answers; regenerate it with ``ZONOS3_WRITE_EDIT_FIXTURES=1``.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import pytest
import torch
from zonos3.tts import edit as E
from zonos3.tts.prompt import TTSPromptConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "edit-replay" / "cases.json"
K, PAD, TEXT_VOCAB, SPAN_END = 4, 13, 500, 14  # a small layout: codebook 12, EOA 12, pad 13, span-end 14


def _cases() -> list[dict]:
    rng = random.Random(1234)
    cases = []

    def add(name, fn, **args):
        cases.append({"name": name, "fn": fn, "args": args})

    def codes(n):
        return [[rng.randrange(12) for _ in range(K)] for _ in range(n)]

    layout = {"n_codebooks": K, "audio_pad_id": PAD, "text_vocab": TEXT_VOCAB, "span_end_id": SPAN_END}
    text_rows = [[PAD] * K + [t] for t in (2, 70, 71, 3)]
    c20 = codes(20)
    add("prompt-one-span", "edit_prompt_tokens", text_rows=text_rows, codes=c20, spans=[[5, 4]], fills=[], **layout)
    add(
        "prompt-span-at-0",
        "edit_prompt_tokens",
        text_rows=text_rows,
        codes=c20,
        spans=[[0, 3], [10, 2]],
        fills=[],
        **layout,
    )
    add(
        "prompt-second-fill",
        "edit_prompt_tokens",
        text_rows=text_rows,
        codes=c20,
        spans=[[2, 3], [9, 4], [15, 5]],
        fills=[codes(6), codes(2)],
        **layout,
    )
    add("prompt-span-to-end", "edit_prompt_tokens", text_rows=text_rows, codes=c20, spans=[[14, 6]], fills=[], **layout)
    add(
        "prompt-no-ungenerated",
        "edit_prompt_tokens",
        text_rows=text_rows,
        codes=c20,
        spans=[[2, 3]],
        fills=[codes(1)],
        **layout,
    )
    add(
        "prompt-bad-span-end",
        "edit_prompt_tokens",
        text_rows=text_rows,
        codes=c20,
        spans=[[2, 3]],
        fills=[],
        **{**layout, "span_end_id": 15},
    )
    add(
        "prompt-overlap",
        "edit_prompt_tokens",
        text_rows=text_rows,
        codes=c20,
        spans=[[2, 3], [5, 2]],
        fills=[],
        **layout,
    )
    add("validate-ok", "validate_edit_spans", spans=[[0, 1], [2, 3]], total_frames=5, max_spans=3)
    add("validate-adjacent", "validate_edit_spans", spans=[[0, 2], [2, 3]], total_frames=5, max_spans=3)
    add("validate-too-many", "validate_edit_spans", spans=[[0, 1], [3, 1]], total_frames=5, max_spans=1)
    return cases


# ---------------------------------------------------------------------------------------------------- python side


def py_call(fn: str, a: dict):
    if fn == "edit_prompt_tokens":
        config = TTSPromptConfig(
            n_codebooks=a["n_codebooks"], audio_pad_id=a["audio_pad_id"], text_vocab=a["text_vocab"]
        )
        rows = E.edit_prompt_tokens(
            torch.tensor(a["text_rows"], dtype=torch.int32),
            torch.tensor(a["codes"]),
            a["spans"],
            [torch.tensor(f).reshape(-1, a["n_codebooks"]) for f in a["fills"]],
            config,
            span_end_id=a["span_end_id"],
        )
        return rows.tolist()
    if fn == "validate_edit_spans":
        return E.validate_edit_spans(a["spans"], a["total_frames"], a["max_spans"])
    raise AssertionError(fn)


def py_run(case: dict):
    try:
        return py_call(case["fn"], case["args"])
    except ValueError as exc:
        return {"error": "ValueError", "message": str(exc)}


def _same(x, y) -> bool:
    if isinstance(x, bool) or isinstance(y, bool):
        return x == y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return x == y or math.isclose(x, y, rel_tol=1e-9)
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    return x == y


CASES = _cases()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_edit_replay(tool, case):
    want = json.loads(json.dumps(py_run(case)))
    got = tool("edit-replay", {"fn": case["fn"], "args": case["args"]})
    assert _same(got, want), f"{case['name']}: ggml {got!r} != python {want!r}"


def test_fixture_file_is_current():
    want = [{**c, "expected": json.loads(json.dumps(py_run(c)))} for c in CASES]
    if os.environ.get("ZONOS3_WRITE_EDIT_FIXTURES"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(want, ensure_ascii=False, indent=1) + "\n")
    assert FIXTURE.exists(), f"{FIXTURE} is missing (ZONOS3_WRITE_EDIT_FIXTURES=1)"
    assert _same(json.loads(FIXTURE.read_text()), want), "regenerate with ZONOS3_WRITE_EDIT_FIXTURES=1"
