#!/usr/bin/env python3
"""Generation-based quant evaluation: greedy-decode held-out prompts with two models and
compare the produced code sequences. Greedy removes sampling noise, so any divergence is the
quantization itself. This closes the autoregressive feedback loop that teacher-forced KLD
cannot see — the metric that mattered for the q2 audio-quality regression.

Usage: eval_gen_divergence.py <ref.gguf> <test.gguf> [--max N] [--gpu G]
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

CLI = os.environ.get("ZONOS3_CLI", "./build-cuda/zonos3-cli")
# Held-out prompts — deliberately NOT in the calibration corpus, to avoid optimistic bias.
PROMPTS = [
    "The lighthouse keeper climbed the spiral staircase as the storm battered the rocky coast below.",
    "Quantum computers exploit superposition and entanglement to solve certain problems much faster.",
    "Good evening, and welcome back to the show; tonight we have a truly remarkable guest for you.",
    "She folded the letter carefully, placed it in the drawer, and never spoke of it again.",
    "The marathon route winds through the old town, past the cathedral, and finishes at the bridge.",
    "Frankly, I did not expect the committee to approve the proposal on its very first reading.",
]


def gen(model, text, max_frames, gpu, tmp, reference_args):
    out = os.path.join(tmp, "c.npy")
    for f in glob.glob(os.path.join(tmp, "c.npy*")):
        os.remove(f)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    r = subprocess.run(
        [CLI, model, "--tts", text, out, "--greedy", "--max", str(max_frames), "--gpu", *reference_args],
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode or not os.path.exists(out):
        sys.stderr.write(r.stderr[-2000:])
        raise RuntimeError(f"gen failed for {model}")
    codes = np.load(out).astype(np.int64)
    if codes.ndim != 2 or codes.shape[1] != 10:
        raise RuntimeError(f"invalid code sequence from {model}: {codes.shape}")
    return codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref")
    parser.add_argument("test")
    parser.add_argument("--max", type=int, default=300, dest="max_frames")
    parser.add_argument("--gpu", default="0")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference-codes")
    reference.add_argument("--clone")
    parser.add_argument("--zac-encoder")
    parser.add_argument("--ssl-teacher")
    args = parser.parse_args()
    ref, test, mx, gpu = args.ref, args.test, args.max_frames, args.gpu
    reference_args = []
    for flag in ("reference_codes", "clone", "zac_encoder", "ssl_teacher"):
        value = getattr(args, flag)
        if value:
            reference_args += ["--" + flag.replace("_", "-"), value]
    print(f"ref = {ref}\ntest= {test}\nmax = {mx}, gpu = {gpu}\n")
    print(f"{'prompt':6s} {'frames(r/t)':>12s} {'1st-div':>8s} {'frame-match%':>12s} {'code-match%':>11s}")
    fm_all, cm_all, div_all = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(PROMPTS):
            a = gen(ref, p, mx, gpu, tmp, reference_args)
            b = gen(test, p, mx, gpu, tmp, reference_args)
            nr, nt = len(a), len(b)
            n = min(nr, nt)
            denominator = max(nr, nt)
            frame_eq = np.all(a[:n] == b[:n], axis=1)
            code_eq = a[:n] == b[:n]
            first_div = int(np.argmin(frame_eq)) if not frame_eq.all() else n
            fm = 100.0 * frame_eq.sum() / denominator if denominator else 100.0
            cm = 100.0 * code_eq.sum() / (denominator * 10) if denominator else 100.0
            fm_all.append(fm)
            cm_all.append(cm)
            div_all.append(first_div)
            print(f"  p{i:<3d} {nr:5d}/{nt:<5d} {first_div:8d} {fm:12.2f} {cm:11.2f}")
    print(
        f"\nmean: 1st-div={np.mean(div_all):.0f} frame-match={np.mean(fm_all):.2f}% code-match={np.mean(cm_all):.2f}%"
    )
    print(
        "(frame-match% = fraction of frames where ALL codebooks match the f16 reference; "
        "higher = closer to f16. A good quant stays high and diverges late.)"
    )


if __name__ == "__main__":
    main()
