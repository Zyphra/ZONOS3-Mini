#!/usr/bin/env python3
"""Strict whole-tensor golden/native comparison with explicit coverage failures.

python tools/compare_golden.py GOLDEN_DIR NATIVE_DIR [--only logits,emb_norm,...]
GOLDEN_DIR is a tools/dump_golden.py <sub> output directory. The golden manifest's
expected_tensors is authoritative. Without a manifest, every
NPY except input_ids is expected. Missing, empty, nonfinite and shape-mismatched
captures fail. Floating elements must satisfy atol + rtol*abs(golden), in addition
to whole-tensor cosine and exact all-row/codebook logits argmax agreement.
Codec-code captures and integer tensors require exact values, regardless of tolerance.
Reported errors and worst-row cosine always cover all rows, not just the last one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


def natural_key(name):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(a @ b / denominator)


def metrics(golden, native):
    g, c = golden.astype(np.float64).ravel(), native.astype(np.float64).ravel()
    difference = np.abs(g - c)
    return {"maxabs": float(difference.max()), "maxrel": float((difference / np.maximum(np.abs(g), 1e-6)).max()),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "cos": cosine(g, c)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("golden", type=Path)
    parser.add_argument("cpp", type=Path)
    parser.add_argument("--only", help="comma-separated REQUIRED capture names, never an intersection filter")
    parser.add_argument("--cos", type=float, default=0.99, help="whole-tensor cosine threshold")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    if not -1 <= args.cos <= 1 or args.atol < 0 or args.rtol < 0 or not np.isfinite([args.cos, args.atol, args.rtol]).all():
        parser.error("thresholds must be finite; cosine in [-1,1], atol/rtol nonnegative")
    manifest_path = args.golden / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get("expected_tensors")
        if not isinstance(expected, list) or not expected or any(not isinstance(n, str) or not n for n in expected):
            parser.error("golden manifest has a missing/empty/invalid expected_tensors set")
        if len(expected) != len(set(expected)):
            parser.error("duplicate expected tensors in golden manifest")
    else:
        expected = [p.stem for p in args.golden.glob("*.npy") if p.stem != "input_ids"]
    if args.only is not None:
        names = args.only.split(",")
        if not names or any(not n.strip() for n in names):
            parser.error("--only must specify a nonempty required tensor set")
        names = [n.strip() for n in names]
    else:
        names = expected
    if not names:
        parser.error("no expected tensors; refusing an empty comparison")
    for name in names:
        if Path(name).name != name or name in (".", ".."):
            parser.error(f"invalid tensor name: {name!r}")
    names = sorted(set(names), key=natural_key)
    failures, compared = 0, 0
    print(f"{'tensor':<26}{'shape':<23}{'maxabs':>11}{'maxrel':>11}{'rmse':>11}{'cos':>11}{'minrowcos':>11}  status")
    for name in names:
        golden_path, native_path = args.golden / (name + ".npy"), args.cpp / (name + ".npy")
        missing = [label for label, path in (("golden", golden_path), ("native", native_path)) if not path.is_file()]
        if missing:
            print(f"{name:<26}MISSING {','.join(missing)}")
            failures += 1
            continue
        g, c = np.load(golden_path, allow_pickle=False), np.load(native_path, allow_pickle=False)
        if g.shape != c.shape:
            print(f"{name:<26}SHAPE-MISMATCH {g.shape} vs {c.shape}")
            failures += 1
            continue
        if not g.size or not np.isfinite(g).all() or not np.isfinite(c).all():
            print(f"{name:<26}EMPTY/NONFINITE")
            failures += 1
            continue
        compared += 1
        m = metrics(g, c)
        row_cos = min(cosine(ga.astype(np.float64).ravel(), ca.astype(np.float64).ravel()) for ga, ca in zip(g, c)) if g.ndim >= 2 else None
        exact = np.issubdtype(g.dtype, np.integer) or name == "codes" or name.endswith(".codes")
        close = (g == c) if exact else np.isclose(g, c, atol=args.atol, rtol=args.rtol)
        ok = bool(close.all()) and m["cos"] >= args.cos
        extra = f" mismatched={int((~close).sum())}/{g.size}"
        if name == "logits":
            if g.ndim != 3:
                ok = False
                extra += " INVALID-LOGITS-RANK"
            else:
                agreement = float((g.argmax(-1) == c.argmax(-1)).mean())
                extra += f" argmax={agreement * 100:.2f}%"
                ok = ok and agreement == 1.0
        failures += not ok
        row_display = f"{row_cos:>11.6f}" if row_cos is not None else f"{'-':>11}"
        print(f"{name:<26}{str(g.shape):<23}{m['maxabs']:>11.4g}{m['maxrel']:>11.4g}{m['rmse']:>11.4g}{m['cos']:>11.6f}{row_display}  {'ok' if ok else 'FAIL'}{extra}")
    print(f"\n{len(names)} expected, {compared} numerically compared, {failures} failure(s); atol={args.atol:g} rtol={args.rtol:g} cos={args.cos:g}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
