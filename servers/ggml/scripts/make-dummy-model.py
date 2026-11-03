#!/usr/bin/env python3
"""Emit a tiny zonos3 GDN/global + FullAttnRes GGUF for CI smoke-testing.

The real checkpoint is too large for CI. This builds a structurally
faithful but minuscule model so release smoke tests run a forward pass through
zonos3-cli (prompt build -> backbone -> sampler), not just a usage string.
The audio is gibberish; the point is exercising the load, graph and decode path.

Dimensions that the prompt builder and output head index into are kept at their real
values (n_codebooks, audio_vocab, text_vocab, codebook ids) and the real SuperBPE
tokenizer is embedded, so --build-prompt ids match the real model; everything that only
affects compute cost (n_embd, n_ff, n_layer, n_head, n_expert, latent dims) is shrunk.
Tensor names/shapes mirror `tools/convert_gguf.py lm` (LM_NAMES) exactly.

Usage:  python scripts/make-dummy-model.py -o dummy.gguf [--tokenizer superbpe8192.json]
                                           [--edit-markers N] [--span-end] [--phonemes SYMBOLS]

The GGUF carries zonos3.text_tokenizer_spec like `tools/convert_gguf.py lm`. --edit-markers adds N speech-edit
marker rows after the text padding row, --phonemes adds a phoneme block (U+E000 / U+E001 groups over the code points
of SYMBOLS) as the last extra text rows, and --span-end writes zonos3.span_end_id (= audio_pad_id + 1, audio_vocab + 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter

ARCH = "zonos3"
TOKENIZER_SHA256 = "90322f910600027dee7fe63d400d1d9dc1035c167d560aadf0871e3661d08bf3"

# --- shrunk compute dims (free to change) ---
DIM       = 32          # n_embd
HEAD_DIM  = 16
N_HEAD    = 2           # N_HEAD*HEAD_DIM == DIM
N_KV      = 1           # GQA: divides N_HEAD
INTER     = 128         # dense feed_forward_length
N_EXPERT  = 4
LATENT    = 32          # block-aligned for quantized expert matmuls
EXP_FF    = 32          # per-expert hidden width
# First three layers exercise GDN; layer 3 exercises global GQA. Every sublayer
# contributes an independent FullAttnRes source, including dense and MoE outputs.
EXPERT_USED = [0, 2, 2, 2]
GDN_HEADS = 1
GDN_HEAD_DIM = 128      # native recurrence kernel width
GDN_KERNEL = 4
N_LAYER   = len(EXPERT_USED)
N_CTX     = 512

# --- real dims the runtime indexes into (must match the reference) ---
N_CODEBOOKS = 10
CODEBOOK    = 4096
AUDIO_VOCAB = CODEBOOK + 2
EOA_ID      = CODEBOOK
PAD_ID      = CODEBOOK + 1
TEXT_VOCAB  = 8196

# --- speaker / condition tokens (real names, edges, units and silence frame of the release LMs; per-table bucket
# counts and speaking-rate edges of zonos3-rate30-20k-r2grpo) ---
SPEAKER_DIM = 192
COND_BUCKET_COUNTS = [10, 10, 30]
COND_NAMES = ["loudness", "fmax", "speaking_rate"]
COND_UNITS = ["LUFS", "Hz", "utf-8 bytes per second"]
COND_EDGES = {
    "loudness": [-29.002, -26.861, -25.366, -23.607, -21.916, -20.505, -19.38, -18.151, -16.297],
    "fmax": [7578.125, 8000.0, 8000.01, 11062.5, 12445.94, 15328.125, 15773.438, 16623.633, 18758.496],
    "speaking_rate": [6.546, 8.182, 9.273, 10.31, 11.054, 11.667, 12.193, 12.656, 13.083, 13.476, 13.846, 14.196,
                      14.53, 14.887, 15.193, 15.517, 15.865, 16.208, 16.579, 16.972, 17.411, 17.898, 18.434, 19.069,
                      19.818, 20.816, 22.286, 24.722, 29.25],
}
SILENCE_FRAME = [1918, 746, 3544, 234, 1267, 2914, 3944, 2849, 326, 1337]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tokenizer", help="pinned superbpe8192.json; otherwise extract the checked-in fixture")
    ap.add_argument("--edit-markers", type=int, default=0, help="speech-edit marker rows after the padding row")
    ap.add_argument("--span-end", action="store_true", help="write zonos3.span_end_id (edit checkpoints)")
    ap.add_argument("--phonemes", help="phone inventory: each code point of this string is one phoneme token")
    args = ap.parse_args()
    spec = {"kind": "superbpe", "path": "tokenizer/superbpe8192.json", "sha256": TOKENIZER_SHA256,
            "vocab_size": TEXT_VOCAB - 4, "offset": 4, "bos": 2, "eos": 3}
    extra = max(0, args.edit_markers)
    if args.phonemes:
        symbols = list(dict.fromkeys(args.phonemes))
        spec["phonemes"] = {"open_marker": "\ue000", "close_marker": "\ue001", "open_id": TEXT_VOCAB + 1 + extra,
                            "close_id": TEXT_VOCAB + 2 + extra, "symbols": symbols}
        extra += 2 + len(symbols)
    audio_vocab = AUDIO_VOCAB + (1 if args.span_end else 0)
    if args.tokenizer:
        with open(args.tokenizer, "rb") as f:
            raw = f.read()
    else:
        fixture = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "dummy-zonos3.gguf")
        reader = GGUFReader(fixture)
        field = reader.fields[f"{ARCH}.tokenizer.json"]
        raw = bytes(field.parts[field.data[0]])
        del field, reader  # release the input mmap before potentially overwriting this fixture
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("tokenizer does not match the pinned target SHA256")
    tok_json = raw.decode("utf-8")

    rng = np.random.default_rng(0)

    def rnd(*shape, gen=None):
        # small init so a 4-layer random net stays numerically tame (no NaN/Inf
        # through the RMS norms and the tanh logit softcap). 1-D tensors (norms,
        # biases, temp, router bias) keep f32; matrices store f16 to halve the
        # committed fixture, mirroring the real converter's storage scheme.
        x = ((gen or rng).standard_normal(shape) * 0.02)
        return x.astype(np.float32 if len(shape) == 1 else np.float16)

    w = GGUFWriter(args.out, ARCH)
    w.add_string("general.name", "zonos3-gdn-fullattnres-dummy")
    w.add_uint32(f"{ARCH}.block_count", N_LAYER)
    w.add_uint32(f"{ARCH}.context_length", N_CTX)
    w.add_uint32(f"{ARCH}.embedding_length", DIM)
    w.add_uint32(f"{ARCH}.head_dim", HEAD_DIM)
    w.add_uint32(f"{ARCH}.attention.head_count", N_HEAD)
    w.add_uint32(f"{ARCH}.attention.head_count_kv", N_KV)
    w.add_uint32(f"{ARCH}.feed_forward_length", INTER)
    w.add_float32(f"{ARCH}.rope.freq_base", 10000.0)
    w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_float32(f"{ARCH}.attention.qk_norm_epsilon", 1e-6)
    w.add_float32(f"{ARCH}.logit_softcap", 15.0)
    w.add_uint32(f"{ARCH}.expert_count", N_EXPERT)
    w.add_array(f"{ARCH}.expert_used_count_per_layer", EXPERT_USED)
    w.add_uint32(f"{ARCH}.n_codebooks", N_CODEBOOKS)
    w.add_uint32(f"{ARCH}.codebook_size", CODEBOOK)
    w.add_uint32(f"{ARCH}.audio_vocab", audio_vocab)
    w.add_uint32(f"{ARCH}.eoa_id", EOA_ID)
    w.add_uint32(f"{ARCH}.audio_pad_id", PAD_ID)
    w.add_uint32(f"{ARCH}.text_vocab", TEXT_VOCAB)
    w.add_string(f"{ARCH}.text_tokenizer_spec", json.dumps(spec, ensure_ascii=False))
    if args.span_end:
        w.add_uint32(f"{ARCH}.span_end_id", PAD_ID + 1)
    w.add_string(f"{ARCH}.codec", "mossdual")
    w.add_string(f"{ARCH}.moe.balancing", "quantile")
    w.add_string(f"{ARCH}.attention.pattern", "gdn3_global1")
    w.add_string(f"{ARCH}.residual.type", "full_attnres")
    w.add_uint32(f"{ARCH}.gdn.head_count", GDN_HEADS)
    w.add_uint32(f"{ARCH}.gdn.head_dim", GDN_HEAD_DIM)
    w.add_uint32(f"{ARCH}.gdn.conv_kernel", GDN_KERNEL)
    w.add_string(f"{ARCH}.tokenizer.json", tok_json)
    w.add_uint32(f"{ARCH}.tokenizer.offset", 4)
    w.add_uint32(f"{ARCH}.tokenizer.bos", 2)
    w.add_uint32(f"{ARCH}.tokenizer.eos", 3)
    w.add_uint32(f"{ARCH}.speaker.embedding_length", SPEAKER_DIM)
    w.add_uint32(f"{ARCH}.condition.count", len(COND_NAMES))
    w.add_array(f"{ARCH}.condition.bucket_counts", COND_BUCKET_COUNTS)
    w.add_array(f"{ARCH}.condition.names", COND_NAMES)
    w.add_array(f"{ARCH}.condition.units", COND_UNITS)
    for name in COND_NAMES:
        w.add_array(f"{ARCH}.condition.edges.{name}", [float(v) for v in COND_EDGES[name]])
    w.add_array(f"{ARCH}.silence_frame", list(SILENCE_FRAME))
    w.add_file_type(1)  # f16 matrices, f32 1-D tensors

    # shapes below are numpy/torch order ([out, in], [rows, dim]); GGUFWriter stores
    # them reversed as ggml ne, matching `tools/convert_gguf.py lm`.
    for i in range(N_CODEBOOKS):
        w.add_tensor(f"audio_embd.{i}.weight", rnd(audio_vocab, DIM))
    w.add_tensor("text_embd.weight", rnd(TEXT_VOCAB + 1 + extra, DIM))
    w.add_tensor("output_norm.weight", rnd(DIM))
    w.add_tensor("output.weight",      rnd(N_CODEBOOKS * audio_vocab, DIM))
    w.add_tensor("attnres_q.weight",   rnd(2 * N_LAYER + 1, DIM))
    w.add_tensor("speaker_proj.weight", rnd(DIM, SPEAKER_DIM))
    w.add_tensor("speaker_proj.bias",   rnd(DIM))
    # The first 11 rows of each table come from the main stream and rows beyond them from their own, so every other
    # weight is independent of the bucket counts.
    extra_rng = np.random.default_rng(1)
    for i, n in enumerate(COND_BUCKET_COUNTS):
        table = rnd(min(n + 1, 11), DIM)
        if n + 1 > 11:
            table = np.concatenate([table, rnd(n + 1 - 11, DIM, gen=extra_rng)])
        w.add_tensor(f"cond_embd.{i}.weight", table)

    for L in range(N_LAYER):
        b = f"blk.{L}."
        w.add_tensor(b + "attn_norm.weight", rnd(DIM))
        w.add_tensor(b + "ffn_norm.weight",  rnd(DIM))
        if (L + 1) % 4 != 0:
            inner = GDN_HEADS * GDN_HEAD_DIM
            w.add_tensor(b + "gdn_in.weight", rnd(4 * inner + 2 * GDN_HEADS, DIM))
            w.add_tensor(b + "gdn_conv.weight", rnd(3 * inner, GDN_KERNEL).astype(np.float32))
            w.add_tensor(b + "gdn_norm.weight", np.ones(GDN_HEAD_DIM, dtype=np.float32))
            w.add_tensor(b + "gdn_A_log", np.zeros(GDN_HEADS, dtype=np.float32))
            w.add_tensor(b + "gdn_dt_bias", np.zeros(GDN_HEADS, dtype=np.float32))
            w.add_tensor(b + "attn_output.weight", rnd(DIM, inner))
        else:
            w.add_tensor(b + "attn_q.weight",      rnd(N_HEAD * HEAD_DIM, DIM))
            w.add_tensor(b + "attn_k.weight",      rnd(N_KV   * HEAD_DIM, DIM))
            w.add_tensor(b + "attn_v.weight",      rnd(N_KV   * HEAD_DIM, DIM))
            w.add_tensor(b + "attn_output.weight", rnd(DIM, N_HEAD * HEAD_DIM))
            w.add_tensor(b + "attn_temp",          rnd(N_HEAD))
            w.add_tensor(b + "attn_gate.weight",   rnd(N_HEAD, DIM))

        if EXPERT_USED[L] == 0:  # dense FFN
            w.add_tensor(b + "ffn_up.weight",   rnd(INTER, DIM))
            w.add_tensor(b + "ffn_gate.weight", rnd(INTER, DIM))
            w.add_tensor(b + "ffn_down.weight", rnd(DIM, INTER))
        else:                    # MoE
            w.add_tensor(b + "ffn_gate_exps.weight",   rnd(N_EXPERT, EXP_FF, LATENT))
            w.add_tensor(b + "ffn_up_exps.weight",     rnd(N_EXPERT, EXP_FF, LATENT))
            w.add_tensor(b + "ffn_down_exps.weight",   rnd(N_EXPERT, LATENT, EXP_FF))
            w.add_tensor(b + "ffn_latent_down.weight", rnd(LATENT, DIM))
            w.add_tensor(b + "ffn_latent_up.weight",   rnd(DIM, LATENT))
            w.add_tensor(b + "router.weight",          rnd(N_EXPERT, DIM))
            w.add_tensor(b + "router_norm.weight",     rnd(DIM))
            w.add_tensor(b + "router_bias",            rnd(N_EXPERT))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
