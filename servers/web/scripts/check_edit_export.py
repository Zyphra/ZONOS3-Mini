#!/usr/bin/env python3
"""Checks the speech-edit adapter export (public/models/edit, export_web_models.py export_edit) against the speech-edit
adapter applied at runtime by the MLX server (zonos3_mlx.companions.AccentBank / AccentState, strength 1), and writes the
parity goldens of harness/checks/backbone.ts backbone_edit.

(a) the export against its sources: the grown rows against zonos3-ptq-edit.gguf, the f16 factors against the f32
    factors of edit/accents.gguf (the head factor fitted to the span-end rows as AccentBank.load does), the router
    biases against bias + shift;
(b) logits over an edit prompt (training layout, speaker row, unknown condition buckets, one region and one
    teacher-forced fill ending on span-end): the MLX LM with the runtime adapter of accents.gguf vs the same LM with the
    exported (f16) factors as its runtime adapter.

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
from zonos3.tts.edit import edit_prompt_tokens
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig
from zonos3_mlx.companions import AccentBank, AccentState, _Slot
from zonos3_mlx.models.lm import LMState, Zonos3MLX

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_web_models import Gguf, edit_adapter  # noqa: E402

MODELS = "servers/ggml/models"
TEXT = "Hello there, this is a short speech editing test sentence."
VOICE = "default_voices/AmericanFemale"
SPANS = [(30, 20), (90, 20)]
QKVG = ("q", "k", "v", "gate")


class Export:
    """The exported tensors of public/models/<model> as float32 [..., out, in] (q8 dequantized)."""

    def __init__(self, path):
        self.man = json.load(open(os.path.join(path, "manifest.json")))
        self.shards = [np.memmap(os.path.join(path, s), dtype=np.uint8, mode="r") for s in self.man["shards"]]

    def __contains__(self, name):
        return name in self.man["tensors"]

    def __getitem__(self, name):
        e = self.man["tensors"][name]
        shape = list(reversed(e["shape"]))
        b = self.shards[e["shard"]]
        data = b[e["offset"]:e["offset"] + e["nbytes"]]
        if e["dtype"] == "q8":
            d = b[e["scales_offset"]:e["scales_offset"] + e["scales_nbytes"]].view(np.float32)
            return (data.view(np.int8).reshape(-1, 32).astype(np.float32) * d[:, None]).reshape(shape)
        return data.view({"f16": np.float16, "f32": np.float32}[e["dtype"]]).reshape(shape).astype(np.float32)


def exported_factors(ex, g):
    """The exported adapter as AccentBank factors: {base: (down, up)}, {layer: {kind: (down, up)}} (f32 of the f16)."""
    linear, experts = {}, {}
    for name in ex.man["tensors"]:
        if not name.endswith(".lora_down"):
            continue
        base = name.removesuffix(".lora_down")
        down, up = ex[name], ex[base + ".lora_up"]
        if base.endswith(".attn_qkvg"):
            L = base.split(".")[1]
            k = down.shape[0] // len(QKVG)
            r0 = 0
            for i, s in enumerate(QKVG):
                rows = g.tensors[f"blk.{L}.attn_{s}.weight"].ne[1]
                block = up[r0:r0 + rows]
                if np.any(np.delete(block, np.s_[i * k:(i + 1) * k], axis=1)):
                    raise SystemExit(f"{base}: the stacked up factor is not block-diagonal")
                linear[f"blk.{L}.attn_{s}"] = (down[i * k:(i + 1) * k], block[:, i * k:(i + 1) * k])
                r0 += rows
        elif base.endswith(".ffn_gate_up_exps"):
            L = int(base.split(".")[1])
            r = down.shape[1] // 2
            experts.setdefault(L, {})["gate"] = (down[:, :r], up[:, 0])
            experts.setdefault(L, {})["up"] = (down[:, r:], up[:, 1])
        elif base.endswith(".ffn_down_exps"):
            experts.setdefault(int(base.split(".")[1]), {})["down"] = (down, up)
        else:
            linear[base] = (down, up)
    return linear, experts


def exports(g, ge, ex, factors, shifts):
    print("(a) export vs sources")
    ncb = g.kv["zonos3.n_codebooks"]
    worst = 0.0
    for k in range(ncb):
        old = g.tensors[f"audio_embd.{k}.weight"].ne[1]
        worst = max(worst, float(np.abs(ex["embd_extra"][k] - ge.arr(f"audio_embd.{k}.weight")[old].astype(np.float32)).max()))
    told = g.tensors["text_embd.weight"].ne[1]
    worst = max(worst, float(np.abs(ex["embd_extra"][ncb:] - ge.arr("text_embd.weight")[told:].astype(np.float32)).max()))
    print(f"embd_extra vs the grown rows: max|err| {worst:.3g}")
    v = g.kv["zonos3.audio_vocab"]
    head = ge.arr("output.weight").astype(np.float32).reshape(ncb, -1, g.tensors["output.weight"].ne[0])[:, v:]
    err = ex["output_extra"] - head.reshape(-1, head.shape[-1])
    print(f"output_extra (q8) vs the grown head rows: rms err {float(np.sqrt((err ** 2).mean())):.3g}, "
          f"rms rows {float(np.sqrt((head ** 2).mean())):.3g}")
    linear, experts = exported_factors(ex, g)
    rel = 0.0
    for base, (down, up) in factors.items():
        kind = base.rsplit(".", 1)[-1]
        if kind.endswith("_exps"):
            got = experts[int(base.split(".")[1])][kind.removeprefix("ffn_").removesuffix("_exps")]
        else:
            got = linear[base]
        for a, b in zip(got, (down, up)):
            rel = max(rel, float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean())))
    if len(linear) + sum(len(e) for e in experts.values()) != len(factors):
        raise SystemExit("export: adapted modules differ from accents.gguf")
    print(f"factors (f16) vs accents.gguf (f32): worst rms err / rms {rel:.3g}")
    bias = max(float(np.abs(ex[f"blk.{L}.router_bias"] - g.arr(f"blk.{L}.router_bias") - s).max()) for L, s in shifts.items())
    print(f"router biases (bias + shift): max|err| {bias:.3g}")
    return linear, experts


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
    g, ge, ex = Gguf(os.path.join(MODELS, "zonos3-ptq.gguf")), Gguf(gguf), Export(a.export)
    ncb = g.kv["zonos3.n_codebooks"]
    _, _, factors, shifts = edit_adapter(acc, ncb, ge.tensors["output.weight"].ne[1])
    linear, experts = exports(g, ge, ex, factors, shifts)

    print("\n(b) logits: MLX runtime adapter (accents.gguf) vs MLX runtime adapter (exported factors)")
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

    ref = run(filled, bank.state(bank.edit, 1.0))
    V = c.audio_vocab
    report("no adapter vs runtime adapter", run(filled, None).reshape(-1, V), ref.reshape(-1, V))
    slot = _Slot({b: (mx.array(d), mx.array(u)) for b, (d, u) in linear.items()},
                 {L: {k: (mx.array(d), mx.array(u)) for k, (d, u) in e.items()} for L, e in experts.items()},
                 {L: mx.array(ex[f"blk.{L}.router_bias"] - g.arr(f"blk.{L}.router_bias")) for L in shifts})
    got = run(filled, AccentState(slot, 1.0))
    report("exported factors vs runtime adapter", got.reshape(-1, V), ref.reshape(-1, V))
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
