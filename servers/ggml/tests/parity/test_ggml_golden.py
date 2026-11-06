"""Release-backbone parity of the ggml graph against the plain-PyTorch reference (zonos3.reference.lm.Zonos3LM):

(a) golden logits: `tools/dump_golden.py lm` (the reference forward with every capture, matrices rounded to F16 as the
    exporter stores them) against `zonos3-cli --validate` on the F16 GGUF, checked by `tools/compare_golden.py`: every
    capture at whole-tensor cosine >= 0.9999 and exact all-row logits argmax. The element tolerance (atol 0.05,
    rtol 0.01) covers the CPU backend's F16 matmul, which rounds activations to F16 (logits reach |x| ~ 30);
(b) the LACI alignment probe on one English prompt (assets/laci_heads/zonos3_mini_mect60k_heads.json): `zonos3-cli --generate
    --greedy --probe-heads --dump-probe` against `Zonos3LM.forward(probe=ProbeSpec)` teacher-forced on the rows ggml
    generated: identical argmax per decode step, centroids within 0.05 text rows.

Inputs (skipped when absent): the backbone checkpoint directory ``ZONOS3_TEST_MODEL`` (default
/data/groups/audio/checkpoints/george_post_training/zonos3-rate30-20k-r2grpo/converted) and the GGUF converted from it,
``ZONOS3_GGUF`` (default servers/ggml/models/zonos3-f16.gguf, `convert_gguf.py lm --outtype f16`).

    uv run --package zonos3-tools pytest servers/ggml/tests/parity/test_ggml_golden.py --ggml-build servers/ggml/build -s
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
TOOLS = REPO / "tools"
sys.path.insert(0, str(TOOLS))

LM_DIR = Path(
    os.environ.get(
        "ZONOS3_TEST_MODEL", "/data/groups/audio/checkpoints/george_post_training/zonos3-rate30-20k-r2grpo/converted"
    )
)
GGUF = Path(os.environ.get("ZONOS3_GGUF", REPO / "servers" / "ggml" / "models" / "zonos3-f16.gguf"))
SPEAKER = REPO / "servers" / "ggml" / "tests" / "fixtures" / "speaker-embedding.npy"
HEADS = REPO / "assets" / "laci_heads" / "zonos3_mini_mect60k_heads.json"
BUCKETS = [3, 5, 29]
GOLDEN_TEXT = "The quick brown fox jumps."
PROBE_TEXT = "Alignment heads follow the text one word at a time, from the first syllable to the last."
PROBE_FRAMES = 60


@pytest.fixture(scope="module")
def release():
    if not (LM_DIR / "params.json").is_file():
        pytest.skip(f"no backbone checkpoint at {LM_DIR} (set ZONOS3_TEST_MODEL)")
    if not GGUF.is_file():
        pytest.skip(f"no backbone GGUF at {GGUF} (set ZONOS3_GGUF)")


def cli(ggml_build: Path, *args: str) -> str:
    exe = ggml_build / "zonos3-cli"
    if not exe.exists():
        pytest.fail(f"{exe} is not built")
    run = subprocess.run([str(exe), str(GGUF), "--cpu", *args], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return run.stdout


# ------------------------------------------------------------------------------------------------- (a) golden


def test_golden_logits(release, ggml_build, tmp_path):
    golden, native = tmp_path / "golden", tmp_path / "cpp"
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "dump_golden.py"),
            "lm",
            str(LM_DIR),
            "-o",
            str(golden),
            "--text",
            GOLDEN_TEXT,
            "--speaker-embedding",
            str(SPEAKER),
            "--condition-buckets",
            ",".join(map(str, BUCKETS)),
            "--silence-prefill",
            "3",
            "--weight-type",
            "f16",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli(ggml_build, "--validate", str(golden / "input_ids.npy"), str(native))
    run = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "compare_golden.py"),
            str(golden),
            str(native),
            "--cos",
            "0.9999",
            "--atol",
            "0.05",
            "--rtol",
            "0.01",
        ],
        capture_output=True,
        text=True,
    )
    print(run.stdout)
    assert run.returncode == 0, run.stdout + run.stderr
    rows = {m[1]: float(m[2]) for m in re.finditer(r"^(\S+)\s+\([^)]*\)\s+\S+\s+\S+\s+\S+\s+(\S+)", run.stdout, re.M)}
    assert len(rows) >= 100 and min(rows.values()) >= 0.9999, f"captures/cosines: {rows}"
    print(f"golden: {len(rows)} captures, min cos {min(rows.values()):.7f}, logits cos {rows['logits']:.7f}")


# ------------------------------------------------------------------------------------------------- (b) probe


def test_laci_probe_readout(release, ggml_build, tmp_path):
    from convert_gguf import load_lm
    from dump_golden import build_prompt, cond_inputs, load_speaker_embedding, round_weights_f16, save_cond
    from zonos3.reference.lm import LMCache, ProbeSpec
    from zonos3.tts.laci import LaciHeads
    from zonos3.tts.probe import locate_text_span

    src = load_lm(str(LM_DIR))
    model, params, meta = src.model, src.params, src.meta
    model.cast(torch.float32)
    round_weights_f16(model)  # the GGUF's F16 matrices
    speaker = load_speaker_embedding(SPEAKER, meta["speaker_dim"])
    prompt, cond = build_prompt(params, meta, src.tokenizer, PROBE_TEXT, speaker=speaker, buckets=BUCKETS)
    prompt_path = tmp_path / "prompt.npy"
    np.save(prompt_path, prompt.astype(np.float32))
    save_cond(prompt_path.with_suffix(".cond.json"), cond)

    probe_path, ids_path = tmp_path / "probe.npy", tmp_path / "ids.npy"
    cli(
        ggml_build,
        "--generate",
        str(prompt_path),
        str(tmp_path / "codes.npy"),
        "--greedy",
        "--max",
        str(PROBE_FRAMES),
        "--dump-ids",
        str(ids_path),
        "--probe-heads",
        str(HEADS),
        "--dump-probe",
        str(probe_path),
    )
    probe = np.load(probe_path)
    ids = np.load(ids_path).astype(np.int64)
    n0, steps = len(prompt), len(probe)
    assert steps >= 20, f"only {steps} decode steps"
    np.testing.assert_array_equal(ids[:n0], prompt, err_msg="--dump-ids does not start with the prompt")

    heads = LaciHeads.load(HEADS).heads
    start, length = locate_text_span(torch.from_numpy(prompt), model.config.n_codebooks)
    spec = ProbeSpec(tuple(heads), start, length)
    speaker_rows, condition_rows = cond_inputs(cond)
    cache = LMCache()
    with torch.no_grad():
        model(torch.from_numpy(ids[:n0]), cache, speaker_rows=speaker_rows, condition_rows=condition_rows)
        readouts = [model(torch.from_numpy(row[None]), cache, probe=spec).probe[0] for row in ids[n0 : n0 + steps]]

    got_argmax, got_centroid = probe[:, 0].astype(np.int64), probe[:, 1].astype(np.float64)
    want_argmax = np.array([a for a, _ in readouts])
    want_centroid = np.array([c for _, c in readouts], dtype=np.float64)
    drift = np.abs(got_centroid - want_centroid)
    print(
        f"probe: {steps} steps over a {length}-row text span, {len(heads)} heads; argmax "
        f"{int((got_argmax == want_argmax).sum())}/{steps} identical, argmax path {got_argmax.min()}..{got_argmax.max()}, "
        f"max centroid diff {drift.max():.2e} (mean {drift.mean():.2e})"
    )
    mismatch = np.flatnonzero(got_argmax != want_argmax)
    assert not len(mismatch), (
        f"argmax differs at steps {mismatch.tolist()}: ggml {got_argmax[mismatch].tolist()} vs "
        f"reference {want_argmax[mismatch].tolist()}"
    )
    assert drift.max() <= 0.05, f"centroid drift {drift.max()} at step {int(drift.argmax())}"
    assert got_argmax.max() > got_argmax.min(), "the probe never moved along the text"
