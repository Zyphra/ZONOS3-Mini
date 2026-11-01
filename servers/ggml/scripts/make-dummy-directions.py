#!/usr/bin/env python3
"""Emit a random emotion-directions set: tests/fixtures/dummy-emotion.gguf (and, optionally, its manifest directory).

The GGUF layout is the one `tools/convert_gguf.py directions` writes (arch "zonos3_directions", src/zonos3-emotion.h).
With --manifest-dir the same directions are also written as a zonos3.tts.emotion.EmotionDirections directory
(manifest.json, <name>.npy, calibration.json), so parity tests can load both sides from one draw. The default is a
raw-space set sized for tests/fixtures/dummy-zonos3.gguf (speaker embedding 192); --space proj|attnres_source makes a
hidden-space set (--dim = hidden size, --source-index for attnres_source).

Usage:  uv run --package zonos3-tools python servers/ggml/scripts/make-dummy-directions.py \
            -o tests/fixtures/dummy-emotion.gguf [--manifest-dir DIR] [--space raw] [--dim 192] [--seed 2]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

ARCH = "zonos3_directions"
NAMED = {"happy": "strong", "sad": "moderate", "angry": "subtle"}
AXES = ("valence", "arousal")
CALIBRATION = {
    "by_speaker": {"default_0123456789abcdef": {"happy": 0.0, "sad": 1.5, "valence-": 0.25}},
    "default": {"angry": 2.0, "arousal": 0.5},
    "global_default": 1.25,
}
PRESETS = {"furious": {"sliders": {"angry": 1.0, "arousal": 0.8, "missing": 1.0}}, "calm": {"arousal": -0.5}}


def served_presets(controls):
    """EmotionDirections.load's preset filter."""
    out = {}
    for name, entry in PRESETS.items():
        sliders = entry.get("sliders", entry) if isinstance(entry, dict) else {}
        kept = {c: float(w) for c, w in sliders.items() if c in controls and float(w) != 0.0}
        if kept:
            out[name] = kept
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--manifest-dir")
    ap.add_argument("--space", default="raw", choices=("raw", "proj", "attnres_source"))
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--source-index", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    if args.space == "attnres_source" and args.source_index < 0:
        ap.error("attnres_source needs --source-index")
    rng = np.random.default_rng(args.seed)
    vectors = {name: (rng.standard_normal(args.dim) * 0.3).astype(np.float32) for name in [*NAMED, *AXES]}
    source_index = args.source_index if args.space == "attnres_source" else -1
    ref_base_norm = 9.5

    w = GGUFWriter(args.out, ARCH)
    w.add_string("general.name", "zonos3-dummy-directions")
    w.add_string(f"{ARCH}.space", args.space)
    w.add_int32(f"{ARCH}.source_index", source_index)
    w.add_uint32(f"{ARCH}.dim", args.dim)
    w.add_uint32(f"{ARCH}.input_dim", args.dim)
    w.add_array(f"{ARCH}.names", list(NAMED))
    w.add_array(f"{ARCH}.axes", list(AXES))
    calibration = json.dumps(CALIBRATION, indent=2)
    w.add_string(f"{ARCH}.calibration", calibration)
    w.add_string(f"{ARCH}.tiers", json.dumps(NAMED))
    w.add_string(f"{ARCH}.presets", json.dumps(served_presets(set(NAMED) | set(AXES))))
    w.add_float32(f"{ARCH}.ref_base_norm", ref_base_norm if math.isfinite(ref_base_norm) else float("nan"))
    for i, name in enumerate(NAMED):
        w.add_tensor(f"dir.named.{i}", vectors[name])
    for i, name in enumerate(AXES):
        w.add_tensor(f"dir.axis.{i}", vectors[name])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")

    if args.manifest_dir:
        root = Path(args.manifest_dir)
        root.mkdir(parents=True, exist_ok=True)
        directions = {}
        for name, vec in vectors.items():
            np.save(root / f"{name}.npy", vec)
            entry = {"file": f"{name}.npy"}
            if name in AXES:
                entry["kind"] = "axis"
            else:
                entry["tier"] = NAMED[name]
            directions[name] = entry
        manifest = {"dim": args.dim, "input_dim": args.dim, "space": args.space, "ref_base_norm": ref_base_norm,
                    "directions": directions, "presets": PRESETS}
        if args.space == "attnres_source":
            manifest["source_index"] = source_index
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (root / "calibration.json").write_text(calibration)
        print(f"wrote {root}")


if __name__ == "__main__":
    main()
