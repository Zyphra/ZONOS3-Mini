#!/usr/bin/env python3
"""Emit tests/fixtures/dummy-accents.gguf: two random accent adapters for tests/fixtures/dummy-zonos3.gguf.

The layout is the one `tools/convert_gguf.py accents` writes (arch "zonos3_accents", src/zonos3-accent.h):
  zonos3_accents.{names, format, rank}; f32 tensors acc.<slot>.<base>.lora_down (numpy [K, in]) /
  .lora_up (numpy [out, K]) on backbone GGUF base names, expert factors acc.<slot>.blk.L.ffn_*_exps.lora_down
  (numpy [E, r, in]) / .lora_up (numpy [E, out, r]) and router shifts acc.<slot>.blk.L.router_bias (numpy [E]).
Adapter "alpha" (zonos-accent-lora/2, rank 4) adapts every linear kind, the experts of the first MoE layer and two
routers; "beta" (zonos-accent-lora/1, rank 2) adapts a subset, so the bank exercises zero-padded columns. Shapes are
read from the backbone GGUF.

Usage:  uv run --package zonos3-tools python servers/ggml/scripts/make-dummy-accents.py \
            [--model tests/fixtures/dummy-zonos3.gguf] -o tests/fixtures/dummy-accents.gguf
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from gguf import GGUFReader, GGUFWriter

ARCH = "zonos3_accents"
LINEAR_LEAVES = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "gdn_in", "ffn_up", "ffn_gate", "ffn_down",
                 "ffn_latent_down", "ffn_latent_up", "router")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(here, "..", "tests", "fixtures", "dummy-zonos3.gguf"))
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    reader = GGUFReader(args.model)
    shapes = {t.name: tuple(int(v) for v in reversed(t.shape.tolist())) for t in reader.tensors}  # numpy order
    del reader

    linear = sorted(name[: -len(".weight")] for name, shape in shapes.items()
                    if name.endswith(".weight") and len(shape) == 2
                    and (name == "output.weight" or name.split(".")[-2] in LINEAR_LEAVES))
    moe_layers = sorted({int(name.split(".")[1]) for name in shapes if name.endswith("ffn_up_exps.weight")})
    rng = np.random.default_rng(1)

    def rnd(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    adapters = [("alpha", "zonos-accent-lora/2", 4, linear, moe_layers[:1], moe_layers[:2]),
                ("beta", "zonos-accent-lora/1", 2, linear[::3] + ["output"], [], moe_layers[-1:])]
    w = GGUFWriter(args.out, ARCH)
    w.add_string("general.name", "zonos3-dummy-accents")
    w.add_array(f"{ARCH}.names", [a[0] for a in adapters])
    w.add_array(f"{ARCH}.format", [a[1] for a in adapters])
    w.add_array(f"{ARCH}.rank", [a[2] for a in adapters])
    for slot, (_, _, rank, modules, expert_layers, router_layers) in enumerate(adapters):
        for base in sorted(set(modules)):
            out_dim, in_dim = shapes[base + ".weight"]
            w.add_tensor(f"acc.{slot}.{base}.lora_down", rnd(rank, in_dim))
            w.add_tensor(f"acc.{slot}.{base}.lora_up", rnd(out_dim, rank))
        for layer in expert_layers:
            for leaf in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                experts, out_dim, in_dim = shapes[f"blk.{layer}.{leaf}.weight"]
                w.add_tensor(f"acc.{slot}.blk.{layer}.{leaf}.lora_down", rnd(experts, rank, in_dim))
                w.add_tensor(f"acc.{slot}.blk.{layer}.{leaf}.lora_up", rnd(experts, out_dim, rank))
        for layer in router_layers:
            (experts,) = shapes[f"blk.{layer}.router_bias"]
            w.add_tensor(f"acc.{slot}.blk.{layer}.router_bias", rnd(experts, scale=0.5))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
