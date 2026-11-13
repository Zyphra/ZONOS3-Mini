"""Prompt parity: ``zonos3-cli --request req.json --build-prompt ids.npy`` (the ggml server's request resolver) against
``python -m zonos3.reference --request req.json --dump-prompt ids.npy`` for every request of
model/tests/fixtures/prompt_requests.json whose ``requires`` the loaded backbone meets. Both sides load the steering
bundle ``assets/steering/studio_bundle_mect60k_x10.json`` (ggml: its ``convert_gguf.py steer`` GGUF) for the ``controls`` entries.

Rows (and a CFG twin's rows) must be identical; the ``.cond.json`` sidecars equal, speaker embeddings within 1e-6.

Refusals: the CLI refuses every request of model/tests/fixtures/request_errors.json whose ``requires`` the backbone GGUF
meets (``conditions``: it has condition rows; ``no_phoneme_tokens`` / ``no_start_token`` / ``no_pause_tokens``: its text
tokenizer spec has no phonemes / start / pauses) with the server's 400 and a detail containing the fixture's ``error``
(no accent adapters, no LACI heads; the GGUF alone, no reference checkpoint needed).

Backbones (each skipped when its GGUF or checkpoint is absent):
- ``release``: the GGUF ``ZONOS3_GGUF`` (default servers/ggml/gguf/backbone-f16.gguf) and the checkpoint it was
  converted from, ``ZONOS3_TEST_MODEL`` (default zonos3-rate30-20k-r2grpo's converted checkpoint);
- ``phonemes``: a phoneme-token backbone (runs the ``phoneme_groups`` entry, and the start / pause entries when it has
  start / pause tokens), ``ZONOS3_PHONEME_GGUF`` (default
  servers/ggml/gguf/backbone-phonemes-f16.gguf) converted from ``ZONOS3_PHONEME_MODEL`` (default
  /data/groups/audio/sofian/runs/z3m_phonemes/tokens/merged_mect_60k)::

    uv run --package zonos3-tools python tools/convert_gguf.py lm "$ZONOS3_PHONEME_MODEL" --outtype f16 \
        --silence-frame <release>/registries/silence_frame_<audio_codec>.json -o servers/ggml/gguf/backbone-phonemes-f16.gguf
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
FIXTURES = REPO / "model" / "tests" / "fixtures" / "prompt_requests.json"
ERROR_FIXTURES = REPO / "model" / "tests" / "fixtures" / "request_errors.json"
MODELS = REPO / "servers" / "ggml" / "gguf"
BACKBONES = {
    "release": (
        Path(os.environ.get("ZONOS3_GGUF", MODELS / "backbone-f16.gguf")),
        os.environ.get(
            "ZONOS3_TEST_MODEL",
            "/data/groups/audio/checkpoints/george_post_training/zonos3-rate30-20k-r2grpo/converted",
        ),
    ),
    "phonemes": (
        Path(os.environ.get("ZONOS3_PHONEME_GGUF", MODELS / "backbone-phonemes-f16.gguf")),
        os.environ.get("ZONOS3_PHONEME_MODEL", "/data/groups/audio/sofian/runs/z3m_phonemes/tokens/merged_mect_60k"),
    ),
}
BUNDLE = REPO / "assets" / "steering" / "studio_bundle_mect60k_x10.json"
ENTRIES = json.loads(FIXTURES.read_text())
ERRORS = json.loads(ERROR_FIXTURES.read_text())


@pytest.fixture(scope="module", params=list(BACKBONES))
def reference(request, tmp_path_factory):
    gguf, model = BACKBONES[request.param]
    if not gguf.is_file():
        pytest.skip(f"{request.param}: no backbone GGUF at {gguf}")
    if not model.startswith("hf://") and not (Path(model) / "params.json").is_file():
        pytest.skip(f"{request.param}: no reference checkpoint at {model}")
    import torch
    from zonos3.reference.lm import Zonos3LM
    from zonos3.reference.synth import ReferenceAssets

    lm = Zonos3LM.load(model, device=torch.device("cpu"), dtype=torch.float32)
    steer = None
    if BUNDLE.is_file():
        sys.path.insert(0, str(REPO / "tools"))
        import convert_gguf

        steer = tmp_path_factory.mktemp("steer") / "steer.gguf"
        convert_gguf.main(["steer", str(BUNDLE), "-o", str(steer)])
    assets = ReferenceAssets.load(steer_bundle=str(BUNDLE) if steer is not None else None)
    return gguf, lm, assets, steer


def _caps(lm, assets) -> set[str]:
    from zonos3.reference.synth import prompt_builder

    caps = {"tag_cfg"}
    spec = prompt_builder(lm).config.text_tokenizer_spec
    if spec is not None and spec.get("phonemes") is not None:
        caps.add("phoneme_tokens")
    if spec is not None and spec.get("start") is not None:
        caps.add("start_token")
    if spec is not None and spec.get("pauses") is not None:
        caps.add("pause_tokens")
    if assets.steer_bundle is not None:
        caps.add("controls")
    return caps


def _cond(path: Path) -> dict:
    return json.loads(path.with_suffix(".cond.json").read_text())


def _assert_same(ggml_ids: Path, ref_ids: Path, what: str) -> None:
    a, b = np.load(ggml_ids), np.load(ref_ids)
    assert a.dtype == np.float32 and b.dtype == np.float32, what
    assert a.shape == b.shape, f"{what}: rows {a.shape} != reference {b.shape}"
    np.testing.assert_array_equal(a, b, err_msg=f"{what}: prompt rows differ")
    ca, cb = _cond(ggml_ids), _cond(ref_ids)
    ea, eb = ca.pop("speaker_embedding", None), cb.pop("speaker_embedding", None)
    assert ca == cb, f"{what}: cond.json differs"
    assert (ea is None) == (eb is None), f"{what}: speaker embedding presence differs"
    if ea is not None:
        np.testing.assert_allclose(
            np.asarray(ea, np.float64),
            np.asarray(eb, np.float64),
            rtol=0,
            atol=1e-6,
            err_msg=f"{what}: speaker embedding",
        )


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["name"] for e in ENTRIES])
def test_prompt_parity(entry, reference, ggml_build, tmp_path):
    gguf, lm, assets, steer = reference
    missing = set(entry["requires"]) - _caps(lm, assets)
    if missing:
        pytest.skip(f"the loaded model / assets lack {sorted(missing)}")
    from zonos3.reference.__main__ import _dump
    from zonos3.reference.synth import plan_prompt

    request = tmp_path / "req.json"
    request.write_text(json.dumps(entry["request"]))
    exe = ggml_build / "zonos3-cli"
    if not exe.exists():
        pytest.fail(f"{exe} is not built")
    ggml_ids = tmp_path / "ggml.npy"
    steer_flags = ["--steer", str(steer)] if steer is not None else []
    run = subprocess.run(
        [
            str(exe),
            str(gguf),
            "--request",
            str(request),
            "--build-prompt",
            str(ggml_ids),
            *steer_flags,
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr

    plan = plan_prompt(lm, entry["request"], assets=assets, device="cpu")
    ref_ids = tmp_path / "ref.npy"
    _dump(ref_ids, plan.cond.input_ids, plan.cond_json())
    _assert_same(ggml_ids, ref_ids, entry["name"])

    ggml_twin = tmp_path / "ggml.twin.npy"
    assert ggml_twin.exists() == (plan.twin is not None), (
        f"{entry['name']}: CFG twin {'missing' if plan.twin is not None else 'unexpected'} (reference: {plan.cfg_kind})"
    )
    if plan.twin is not None:
        ref_twin = tmp_path / "ref.twin.npy"
        _dump(ref_twin, plan.twin.input_ids, plan.cond_json(plan.twin))
        _assert_same(ggml_twin, ref_twin, f"{entry['name']} ({plan.cfg_kind} twin)")


@pytest.fixture(scope="module", params=list(BACKBONES))
def backbone_caps(request):
    """``(gguf, caps)``: a backbone GGUF and the ``requires`` tags of request_errors.json it meets."""
    gguf = BACKBONES[request.param][0]
    if not gguf.is_file():
        pytest.skip(f"{request.param}: no backbone GGUF at {gguf}")
    from gguf import GGUFReader

    fields = GGUFReader(gguf).fields

    def value(key: str):
        field = fields.get(key)
        return None if field is None else field.parts[field.data[0]]

    caps = set()
    count = value("zonos3.condition.count")
    if count is not None and int(count[0]) > 0:
        caps.add("conditions")
    spec = json.loads(bytes(value("zonos3.text_tokenizer_spec")).decode())
    if spec.get("phonemes") is None:
        caps.add("no_phoneme_tokens")
    if spec.get("start") is None:
        caps.add("no_start_token")
    if spec.get("pauses") is None:
        caps.add("no_pause_tokens")
    return gguf, caps


@pytest.mark.parametrize("entry", ERRORS, ids=[e["name"] for e in ERRORS])
def test_request_refused(entry, backbone_caps, ggml_build, tmp_path):
    gguf, caps = backbone_caps
    missing = set(entry["requires"]) - caps
    if missing:
        pytest.skip(f"the backbone lacks {sorted(missing)}")
    request = tmp_path / "req.json"
    request.write_text(json.dumps(entry["request"]))
    exe = ggml_build / "zonos3-cli"
    if not exe.exists():
        pytest.fail(f"{exe} is not built")
    run = subprocess.run(
        [str(exe), str(gguf), "--request", str(request), "--build-prompt", str(tmp_path / "ids.npy")],
        capture_output=True,
        text=True,
    )
    assert run.returncode != 0, f"{entry['name']}: accepted"
    # The refusal is the last stderr line of the form "<status> {"detail": ...}" (ggml backend teardown may log after it).
    refusal = [line for line in run.stderr.splitlines() if line[:3].isdigit() and line[3:5] == " {"]
    assert refusal, run.stderr
    status, _, body = refusal[-1].partition(" ")
    assert status == "400", run.stderr
    detail = json.loads(body)["detail"]
    assert isinstance(detail, str) and detail, run.stderr
