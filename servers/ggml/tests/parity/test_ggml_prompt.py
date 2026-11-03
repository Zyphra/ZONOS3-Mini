"""Prompt parity: ``zonos3-cli --request req.json --build-prompt ids.npy`` (the ggml server's request resolver) against
``python -m zonos3.reference --request req.json --dump-prompt ids.npy`` for every request of
model/tests/fixtures/prompt_requests.json whose ``requires`` the loaded backbone meets.

Rows (and a CFG twin's rows) must be identical; the ``.cond.json`` sidecars equal, speaker embeddings within 1e-6.

Backbones (each skipped when its GGUF or checkpoint is absent):
- ``release``: the GGUF ``ZONOS3_GGUF`` (default servers/ggml/models/zonos3-f16.gguf) and the checkpoint it was
  converted from, ``ZONOS3_TEST_MODEL`` (default the Mini release's lm/mect_60k);
- ``phonemes``: a phoneme-token backbone (runs the ``phoneme_groups`` entry), ``ZONOS3_PHONEME_GGUF`` (default
  servers/ggml/models/zonos3-phonemes-f16.gguf) converted from ``ZONOS3_PHONEME_MODEL`` (default
  /data/groups/audio/sofian/runs/z3m_phonemes/tokens/merged_mect_60k)::

    uv run --package zonos3-tools python tools/convert_gguf.py lm "$ZONOS3_PHONEME_MODEL" --outtype f16 \
        --silence-frame <release>/registries/silence_frame_<audio_codec>.json -o servers/ggml/models/zonos3-phonemes-f16.gguf
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
MODELS = REPO / "servers" / "ggml" / "models"
BACKBONES = {
    "release": (Path(os.environ.get("ZONOS3_GGUF", MODELS / "zonos3-f16.gguf")),
                os.environ.get("ZONOS3_TEST_MODEL",
                               "/data/groups/audio/gabriel/hf_release_zonos3mini_20261020/lm/mect_60k")),
    "phonemes": (Path(os.environ.get("ZONOS3_PHONEME_GGUF", MODELS / "zonos3-phonemes-f16.gguf")),
                 os.environ.get("ZONOS3_PHONEME_MODEL",
                                "/data/groups/audio/sofian/runs/z3m_phonemes/tokens/merged_mect_60k")),
}
EMOTION_DIR = REPO / "steering" / "emotion_directions_mect60k_axes"
ENTRIES = json.loads(FIXTURES.read_text())


@pytest.fixture(scope="module", params=list(BACKBONES))
def reference(request):
    gguf, model = BACKBONES[request.param]
    if not gguf.is_file():
        pytest.skip(f"{request.param}: no backbone GGUF at {gguf}")
    if not model.startswith("hf://") and not (Path(model) / "params.json").is_file():
        pytest.skip(f"{request.param}: no reference checkpoint at {model}")
    import torch

    from zonos3.reference.lm import Zonos3LM
    from zonos3.reference.synth import ReferenceAssets

    lm = Zonos3LM.load(model, device=torch.device("cpu"), dtype=torch.float32)
    assets = ReferenceAssets.load(emotion_dir=str(EMOTION_DIR) if EMOTION_DIR.is_dir() else None)
    return gguf, lm, assets


def _caps(lm, assets) -> set[str]:
    from zonos3.reference.synth import prompt_builder

    caps = {"tag_cfg"}
    spec = prompt_builder(lm).config.text_tokenizer_spec
    if spec is not None and spec.get("phonemes") is not None:
        caps.add("phoneme_tokens")
    if assets.emotion_directions is not None:
        caps.add("emotion")
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
        np.testing.assert_allclose(np.asarray(ea, np.float64), np.asarray(eb, np.float64), rtol=0, atol=1e-6,
                                   err_msg=f"{what}: speaker embedding")


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["name"] for e in ENTRIES])
def test_prompt_parity(entry, reference, ggml_build, tmp_path):
    gguf, lm, assets = reference
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
    run = subprocess.run(
        [str(exe), str(gguf), "--request", str(request), "--build-prompt", str(ggml_ids),
         "--worker-python", sys.executable, "--python-src", str(REPO / "model" / "src")],
        capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    plan = plan_prompt(lm, entry["request"], assets=assets, device="cpu")
    ref_ids = tmp_path / "ref.npy"
    _dump(ref_ids, plan.cond.input_ids, plan.cond_json())
    _assert_same(ggml_ids, ref_ids, entry["name"])

    ggml_twin = tmp_path / "ggml.twin.npy"
    assert ggml_twin.exists() == (plan.twin is not None), (
        f"{entry['name']}: CFG twin {'missing' if plan.twin is not None else 'unexpected'} (reference: {plan.cfg_kind})")
    if plan.twin is not None:
        ref_twin = tmp_path / "ref.twin.npy"
        _dump(ref_twin, plan.twin.input_ids, plan.cond_json(plan.twin))
        _assert_same(ggml_twin, ref_twin, f"{entry['name']} ({plan.cfg_kind} twin)")
