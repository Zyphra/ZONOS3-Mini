#!/usr/bin/env python3
"""Checks the merged speech-edit export (public/models/edit, export_web_models.py) against the speech-edit adapter
applied at runtime by the MLX server (zonos3_mlx.companions.AccentBank / AccentState, strength 1), and writes the
parity goldens of harness/checks/backbone.ts backbone_edit.

(a) per adapted tensor: the exported weights against W + up @ down in f32 (W dequantized from zonos3-ptq-edit.gguf, the
    head factor fitted to the span-end rows as AccentBank.load does), and the router biases against bias + shift;
(b) logits over an edit prompt (training layout, speaker row, unknown condition buckets, one region and one
    teacher-forced fill ending on span-end): the MLX LM with the runtime adapter vs the same LM running the exported
    tensors with no adapter.

Run from the repository root (needs the MLX server package):
    uv run --package zonos3-mlx python servers/web/scripts/check_edit_export.py [--goldens servers/web/public/goldens]
"""
import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np
import torch
from gguf import GGMLQuantizationType
from zonos3.tts.edit import edit_prompt_tokens
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig
from zonos3_mlx.companions import AccentBank
from zonos3_mlx.gguf_io import Weight, _repack
from zonos3_mlx.models.lm import LMState, Zonos3MLX

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_web_models import Gguf, dequantize, edit_adapter  # noqa: E402

MODELS = "servers/ggml/models"
TEXT = "Hello there, this is a short speech editing test sentence."
VOICE = "default_voices/AmericanFemale"
SPANS = [(30, 20), (90, 20)]


class Export:
    """The exported tensors of public/models/<model> as float32 [..., out, in] (q8/q4 dequantized)."""

    def __init__(self, path):
        self.man = json.load(open(os.path.join(path, "manifest.json")))
        self.shards = [np.memmap(os.path.join(path, s), dtype=np.uint8, mode="r") for s in self.man["shards"]]

    def dtype(self, name):
        return self.man["tensors"][name]["dtype"]

    def __getitem__(self, name):
        e = self.man["tensors"][name]
        shape = list(reversed(e["shape"]))
        b = self.shards[e["shard"]]
        data = b[e["offset"]:e["offset"] + e["nbytes"]]
        if e["dtype"] in ("q8", "q4"):
            d = b[e["scales_offset"]:e["scales_offset"] + e["scales_nbytes"]].view(np.float32)
            if e["dtype"] == "q8":
                q = data.view(np.int8).reshape(-1, 32).astype(np.float32)
            else:
                qs = data.reshape(-1, 16)
                q = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.float32) - 8
            return (q * d[:, None]).reshape(shape)
        return data.view({"f16": np.float16, "f32": np.float32}[e["dtype"]]).reshape(shape).astype(np.float32)

    def weight(self, name, rows=None):
        """An MLX weight with the exported values: q8 repacked exactly, f16/f32 dense."""
        e = self.man["tensors"][name]
        if e["dtype"] != "q8":
            arr = self[name] if rows is None else self[name][rows]
            return Weight(mx.array(arr.astype(np.float16 if e["dtype"] == "f16" else np.float32)))
        b = self.shards[e["shard"]]
        q = b[e["offset"]:e["offset"] + e["nbytes"]].reshape(-1, 32)
        d = b[e["scales_offset"]:e["scales_offset"] + e["scales_nbytes"]].view(np.float32)
        shape = tuple(reversed(e["shape"]))
        if rows is not None:  # whole rows of 32-element blocks
            per = shape[-1] // 32
            q, d = q[rows.start * per:rows.stop * per], d[rows.start * per:rows.stop * per]
            shape = (rows.stop - rows.start, shape[-1])
        blocks = np.empty((d.size, 34), np.uint8)
        blocks[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
        blocks[:, 2:] = q
        return _repack(blocks.reshape(-1), shape, GGMLQuantizationType.Q8_0)


def tensors(g, ex, factors, shifts):
    print("(a) exported vs f32 W + up @ down")
    print(f"{'tensor':28s} {'dtype':5s} {'max|err|':>9s} {'rms err':>9s} {'rms delta':>9s} {'err/delta':>9s}")
    worst = {}
    qkvg = {"attn_q": 0, "attn_k": 512, "attn_v": 640, "attn_gate": 768}
    for base, (down, up) in factors.items():
        delta = np.matmul(up, down)
        ref = dequantize(g, base + ".weight") + delta
        kind = base.split(".")[-1]
        if kind in qkvg:
            name = f"blk.{base.split('.')[1]}.attn_qkvg.weight"
            got = ex[name][qkvg[kind]:qkvg[kind] + ref.shape[0]]
        else:
            name = base + ".weight"
            got = ex[name]
        err = got - ref
        rd, re = float(np.sqrt((delta ** 2).mean())), float(np.sqrt((err ** 2).mean()))
        worst[kind] = max(worst.get(kind, 0.0), re / rd)
        print(f"{base:28s} {ex.dtype(name):5s} {float(np.abs(err).max()):9.3g} {re:9.3g} {rd:9.3g} {re / rd:9.4f}")
    bias = max(float(np.abs(ex[f"blk.{L}.router_bias"] - g.arr(f"blk.{L}.router_bias") - s).max())
               for L, s in shifts.items())
    print(f"router biases (bias + shift): max|err| {bias:.3g}")
    print("worst rms err / rms delta per module:", {k: round(v, 4) for k, v in sorted(worst.items())})


def report(tag, got, ref):
    d = got - ref
    cos = (got * ref).sum(-1) / np.sqrt((got ** 2).sum(-1) * (ref ** 2).sum(-1))
    p = ref - ref.max(-1, keepdims=True)
    p = p - np.log(np.exp(p).sum(-1, keepdims=True))
    q = got - got.max(-1, keepdims=True)
    q = q - np.log(np.exp(q).sum(-1, keepdims=True))
    kl = (np.exp(p) * (p - q)).sum(-1)
    print(f"{tag:40s} rows {got.shape[0]:4d}  max|dlogit| {np.abs(d).max():7.4f}  rms {np.sqrt((d ** 2).mean()):.4f}  "
          f"min cos {cos.min():.5f}  argmax {(got.argmax(-1) == ref.argmax(-1)).mean() * 100:6.2f}%  "
          f"KL mean {kl.mean():.2e} max {kl.max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="servers/web/public/models/edit")
    ap.add_argument("--goldens", help="write goldens/edit/ for harness/checks/backbone.ts backbone_edit")
    a = ap.parse_args()
    gguf, acc = os.path.join(MODELS, "zonos3-ptq-edit.gguf"), os.path.join(MODELS, "edit/accents.gguf")
    g, ex = Gguf(gguf), Export(a.export)
    ncb = g.kv["zonos3.n_codebooks"]
    _, factors, shifts = edit_adapter(acc, ncb, g.tensors["output.weight"].ne[1])
    tensors(g, ex, factors, shifts)

    print("\n(b) logits: MLX runtime adapter vs MLX on the exported tensors")
    lm = Zonos3MLX.load(gguf)
    c = lm.config
    bank = AccentBank.load(acc, head_rows=c.n_codebooks * c.audio_vocab, n_codebooks=c.n_codebooks)
    builder = TTSPromptBuilder(TTSPromptConfig(n_codebooks=c.n_codebooks, codebook_sizes=c.codebook_sizes,
                                               audio_pad_id=c.audio_pad_id, text_vocab=c.text_vocab,
                                               text_tokenizer_spec=c.text_tokenizer_spec))
    codes = torch.as_tensor(np.load(VOICE + ".codes.npy")[:150], dtype=torch.int32)
    emb = np.load(VOICE + ".embedding.npy").astype(np.float32)
    text = builder.build(TEXT)
    lead = torch.cat((builder.speaker_slot(), builder.condition_rows(c.condition_count)))
    prompt = torch.cat((lead, edit_prompt_tokens(text, codes, SPANS, [], builder.config,
                                                 span_end_id=c.span_end_id))).numpy()
    filled = torch.cat((lead, edit_prompt_tokens(text, codes, SPANS, [codes[SPANS[0][0]:sum(SPANS[0])]],
                                                 builder.config, span_end_id=c.span_end_id))).numpy()
    # the fill rows continue the prompt: the prompt is a prefix of `filled` up to its last (marker) row
    n0 = len(prompt)
    if not np.array_equal(filled[:n0 - 1], prompt[:n0 - 1]):
        raise SystemExit("edit layout: the prompt is not a prefix of the filled layout")
    buckets = list(c.condition_bucket_counts)  # unknown
    cond_rows = {1 + i: int(n) for i, n in enumerate(buckets)}

    def run(ids, accents):
        out = lm.forward(ids, LMState(), speaker_rows={0: emb}, condition_rows=cond_rows, accents=accents)
        return np.array(out.logits.astype(mx.float32)).reshape(len(ids), -1)

    state = bank.state(bank.edit, 1.0)
    ref = run(filled, state)
    V = c.audio_vocab
    report("no adapter vs runtime adapter", run(filled, None).reshape(-1, V), ref.reshape(-1, V))
    for name in ex.man["tensors"]:
        if name.endswith(".router_bias"):
            lm.t[name] = mx.array(ex[name])
        elif name.endswith("attn_qkvg.weight"):
            L = name.split(".")[1]
            for kind, lo, hi in (("q", 0, 512), ("k", 512, 640), ("v", 640, 768), ("gate", 768, 784)):
                lm.w[f"blk.{L}.attn_{kind}.weight"] = ex.weight(name, slice(lo, hi))
        elif name in lm.w:
            lm.w[name] = ex.weight(name)
    mx.eval([w.w for w in lm.w.values()] + list(lm.t.values()))
    got = run(filled, None)
    report("exported (merged) vs runtime adapter", got.reshape(-1, V), ref.reshape(-1, V))
    report("  rows from the prompt's last row on", got[n0 - 1:].reshape(-1, V), ref[n0 - 1:].reshape(-1, V))

    if a.goldens:
        out = os.path.join(a.goldens, "edit")
        os.makedirs(out, exist_ok=True)
        # rows n0-1 ..: prefill of filled[:n0] then teacher-forced steps over filled[n0:]; float32 .npy (harness npy)
        np.save(os.path.join(out, "ids.npy"), filled.astype(np.float32))
        np.save(os.path.join(out, "logits.npy"), ref[n0 - 1:].astype(np.float32))
        with open(os.path.join(out, "cond.json"), "w") as f:
            json.dump({"n0": n0, "speaker_row": 0, "condition_row": 1, "condition_buckets": buckets,
                       "speaker_embedding": [float(v) for v in emb]}, f)
        print(f"goldens: {out} ({len(filled)} rows, prompt {n0}, {len(filled) - n0 + 1} logit rows)")


if __name__ == "__main__":
    main()
