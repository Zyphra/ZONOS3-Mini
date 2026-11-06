#!/usr/bin/env python3
"""Export FuseAlign (hf download Zyphra/fusealign fusealign-multilingual-cjkft-step2500-q8_0.gguf) into the sharded web
format (zonos3-web-v1, see export_web_models.py) as the model directory `aligner` (src/aligner/aligner.ts).

The q8_0 matrices of both transformers are copied as is (the GGUF's README: q8_0 matches f32 accuracy). Re-laid out for
the web kernels (ggml memory order, the gemm's [K][O][IC] weight layout):
  dia.<i>.qkv         q|k|v projections concatenated along the outputs (q8)
  fa.<i>.w13          SwiGLU w1|w3 concatenated along the outputs (q8)
  fa.audio_patch      Conv1d(1, dim, 512, stride 256) as a K=2 gemm over 256-sample rows: [2][dim][256] f32
  fa.audio_bias       audio_patch.bias + mod_emb[0]; fa.text_bias = text_proj.bias + mod_emb[1]
  fa.<head>.up<s>.even / .odd   ConvTranspose1d(k4, s2, p1) as two K=2 gemms: output 2m reads rows (m-1, m) with
                      taps (3, 1), output 2m+1 rows (m, m+1) with taps (2, 0); [2][feat][feat] f32
  fa.<head>.smooth    Conv1d(k5, p2) as a K=5 gemm: [5][feat][feat] f32
  fa.mlp0             the three word MLPs' first layers concatenated (start|end|presence): [3*dim][dim] f32
The Dia embedding (BF16) is a host tensor (rows gathered on the CPU); the Dia norm weights are f32 of their bf16 values.
The Silero VAD tensors are not exported (no long-form alignment in the browser).
"""
import argparse
import os

import numpy as np
from export_web_models import Gguf, Writer
from gguf import GGMLQuantizationType

HF_REPO = "Zyphra/fusealign"
GGUF_FILE = "fusealign-multilingual-cjkft-step2500-q8_0.gguf"
UP_KERNEL, SMOOTH_KERNEL = 4, 5


def bf16_f32(g, name):
    """A BF16 tensor as float32, C order shape reversed(ne)."""
    qtype = g.tensors[name].tensor_type
    if qtype != GGMLQuantizationType.BF16:
        raise SystemExit(f"{name}: expected BF16, got {qtype.name}")
    bits = np.asarray(g.raw(name)).view(np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(list(reversed(g.ne(name))))


def f32(g, name, ne):
    qtype = g.tensors[name].tensor_type
    if qtype != GGMLQuantizationType.F32 or g.ne(name) != list(ne):
        raise SystemExit(f"{name}: expected F32 {ne}, got {qtype.name} {g.ne(name)}")
    return np.asarray(g.arr(name), dtype=np.float32)


def q8(g, name, ne):
    qtype = g.tensors[name].tensor_type
    if qtype != GGMLQuantizationType.Q8_0 or g.ne(name) != list(ne):
        raise SystemExit(f"{name}: expected Q8_0 {ne}, got {qtype.name} {g.ne(name)}")
    return np.asarray(g.raw(name))


def export_aligner(path, out):
    g = Gguf(path)
    kv = g.kv
    if kv.get("general.architecture") != "fusealign" or kv.get("fusealign.format") != "fusealign-gguf/1":
        raise SystemExit(f"{path}: not a fusealign-gguf/1 file")
    hp = {k.split(".", 1)[1]: v for k, v in kv.items() if k.startswith("fusealign.")}
    dia = {k.split(".", 1)[1]: v for k, v in kv.items() if k.startswith("dia.")}
    dim, nl, nh, ffn, feat = hp["dim"], hp["n_layers"], hp["n_heads"], hp["ffn_dim"], hp["feat_dim"]
    win, hop, nup = hp["patch_window"], hp["patch_hop"], hp["n_upsample"]
    D, dl, dnh, dhd, inter = (dia["hidden_size"], dia["num_layers"], dia["num_heads"], dia["head_dim"],
                              dia["intermediate_size"])
    HD = dnh * dhd
    if win != 2 * hop or hp["text_dim"] != D or dim % nh or dia["vocab_size"] != 256:
        raise SystemExit(f"{path}: unsupported FuseAlign hyperparameters")
    meta = {
        "general.architecture": "fusealign", "format": "fusealign-gguf/1", "source": os.path.basename(path),
        "dim": dim, "n_layers": nl, "n_heads": nh, "ffn_dim": ffn, "feat_dim": feat, "patch_window": win,
        "patch_hop": hop, "n_upsample": nup, "rope_theta": hp["rope_theta"], "norm_eps": 1e-6,
        "presence_threshold": hp["presence_threshold"], "max_text_bytes": hp["max_text_bytes"],
        "max_seconds": hp["max_seconds"], "dia_hidden": D, "dia_layers": dl, "dia_heads": dnh, "dia_head_dim": dhd,
        "dia_intermediate": inter, "dia_norm_eps": dia["norm_eps"], "dia_rope_theta": dia["rope_theta"],
        "dia_max_positions": dia["max_position_embeddings"],
    }
    w = Writer(os.path.join(out, "aligner"), "aligner", meta)

    # Dia text encoder
    w.add("dia.embedding", bf16_f32(g, "text_encoder.embedding.weight"), "f32", [D, 256], host=True)
    for i in range(dl):
        p = f"text_encoder.layers.{i}."
        w.add(f"dia.{i}.pre_norm", bf16_f32(g, p + "pre_sa_norm.weight"), "f32", [D])
        w.add(f"dia.{i}.post_norm", bf16_f32(g, p + "post_sa_norm.weight"), "f32", [D])
        qkv = np.concatenate([q8(g, p + f"self_attention.{n}_proj.weight", [D, HD]) for n in "qkv"])
        w.add_q8(f"dia.{i}.qkv", qkv, [D, 3 * HD])
        w.add_q8(f"dia.{i}.o", q8(g, p + "self_attention.o_proj.weight", [HD, D]), [HD, D])
        w.add_q8(f"dia.{i}.gate_up", q8(g, p + "mlp.gate_up_proj.weight", [D, 2 * inter]), [D, 2 * inter])
        w.add_q8(f"dia.{i}.down", q8(g, p + "mlp.down_proj.weight", [inter, D]), [inter, D])
    w.add("dia.norm", bf16_f32(g, "text_encoder.norm.weight"), "f32", [D])

    # FuseAlign input projections (mod_emb folded into the biases)
    mod = f32(g, "model.mod_emb", [dim, 2])
    patch = f32(g, "model.audio_patch.weight", [win, 1, dim]).reshape(dim, 2, hop).transpose(1, 0, 2)
    w.add("fa.audio_patch", np.ascontiguousarray(patch), "f32", [hop, dim, 2])
    w.add("fa.audio_bias", f32(g, "model.audio_patch.bias", [dim]) + mod[0], "f32", [dim])
    w.add_q8("fa.text_proj", q8(g, "model.text_proj.weight", [D, dim]), [D, dim])
    w.add("fa.text_bias", f32(g, "model.text_proj.bias", [dim]) + mod[1], "f32", [dim])

    # Joint blocks
    for i in range(nl):
        p = f"model.blocks.{i}."
        w.add(f"fa.{i}.attn_norm", f32(g, p + "attn_norm.weight", [dim]), "f32", [dim])
        w.add_q8(f"fa.{i}.qkv", q8(g, p + "attn.qkv.weight", [dim, 3 * dim]), [dim, 3 * dim])
        w.add_q8(f"fa.{i}.out", q8(g, p + "attn.out.weight", [dim, dim]), [dim, dim])
        w.add(f"fa.{i}.ffn_norm", f32(g, p + "ffn_norm.weight", [dim]), "f32", [dim])
        w13 = np.concatenate([q8(g, p + "ffn.w1.weight", [dim, ffn]), q8(g, p + "ffn.w3.weight", [dim, ffn])])
        w.add_q8(f"fa.{i}.w13", w13, [dim, 2 * ffn])
        w.add_q8(f"fa.{i}.w2", q8(g, p + "ffn.w2.weight", [ffn, dim]), [ffn, dim])
    w.add("fa.final_norm", f32(g, "model.final_norm.weight", [dim]), "f32", [dim])

    # Refinement heads
    for head in ("start", "end"):
        p = f"model.{head}_refinement."
        w.add(f"fa.{head}.in", f32(g, p + "0.weight", [1, dim, feat]).reshape(feat, dim), "f32", [dim, feat])
        w.add(f"fa.{head}.in_b", f32(g, p + "0.bias", [feat]), "f32", [feat])
        for s in range(nup):
            up = f32(g, p + f"{1 + 2 * s}.weight", [UP_KERNEL, feat, feat])  # [in][out][k]
            even = np.stack([up[:, :, 3].T, up[:, :, 1].T])                  # [k][out][in]
            odd = np.stack([up[:, :, 2].T, up[:, :, 0].T])
            w.add(f"fa.{head}.up{s}.even", np.ascontiguousarray(even), "f32", [feat, feat, 2])
            w.add(f"fa.{head}.up{s}.odd", np.ascontiguousarray(odd), "f32", [feat, feat, 2])
            w.add(f"fa.{head}.up{s}.b", f32(g, p + f"{1 + 2 * s}.bias", [feat]), "f32", [feat])
        k = 1 + 2 * nup
        smooth = f32(g, p + f"{k}.weight", [SMOOTH_KERNEL, feat, feat]).transpose(2, 0, 1)  # [out][in][k] -> [k][out][in]
        w.add(f"fa.{head}.smooth", np.ascontiguousarray(smooth), "f32", [feat, feat, SMOOTH_KERNEL])
        w.add(f"fa.{head}.smooth_b", f32(g, p + f"{k}.bias", [feat]), "f32", [feat])
        w.add(f"fa.{head}.out", f32(g, p + f"{k + 2}.weight", [1, feat, feat]).reshape(feat, feat), "f32", [feat, feat])
        w.add(f"fa.{head}.out_b", f32(g, p + f"{k + 2}.bias", [feat]), "f32", [feat])

    # Word MLPs
    mlps = ("start", "end", "presence")
    w.add("fa.mlp0", np.concatenate([f32(g, f"model.{m}_mlp.0.weight", [dim, dim]) for m in mlps]), "f32",
          [dim, 3 * dim])
    w.add("fa.mlp0_b", np.concatenate([f32(g, f"model.{m}_mlp.0.bias", [dim]) for m in mlps]), "f32", [3 * dim])
    for m in mlps:
        o = 1 if m == "presence" else feat
        w.add(f"fa.{m}_mlp2", f32(g, f"model.{m}_mlp.2.weight", [dim, o]), "f32", [dim, o])
        w.add(f"fa.{m}_mlp2_b", f32(g, f"model.{m}_mlp.2.bias", [o]), "f32", [o])
    w.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", default=None, help=f"FuseAlign q8_0 GGUF (default: {GGUF_FILE} from the local "
                                                 f"Hugging Face cache of {HF_REPO})")
    ap.add_argument("--out", required=True, help="models output directory (writes <out>/aligner)")
    a = ap.parse_args()
    path = a.gguf
    if path is None:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(HF_REPO, GGUF_FILE)
        if not isinstance(path, str):
            raise SystemExit(f"{HF_REPO}/{GGUF_FILE} is not in the Hugging Face cache: hf download {HF_REPO} "
                             f"{GGUF_FILE} (or pass --gguf)")
    export_aligner(path, a.out)


if __name__ == "__main__":
    main()
