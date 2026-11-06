#!/usr/bin/env python3
"""PyTorch golden dumps for the ggml port, one subcommand per release component.

    tools/dump_golden.py lm              RELEASE/lm/rate30_20k_r2grpo -o OUT [--text TEXT | --input-ids ids.npy] ...
    tools/dump_golden.py codec-decoder   --generator G --head-dir H codes.npy -o OUT [--compare cpp.npy]
    tools/dump_golden.py codec-encoder   --generator G --audio A -o OUT [--compare codes.npy]
    tools/dump_golden.py ssl-teacher     --teacher DIR (--audio A | --pcm44100 X | --features F) -o OUT
    tools/dump_golden.py speaker-encoder ckpt_0037.pth audio.wav -o OUT

Every subcommand writes float32 .npy captures plus (where the comparison needs one) a manifest.json whose
expected_tensors set is authoritative for tools/compare_golden.py. `tools/dump_golden.py SUB -h` documents each.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

# ------------------------------------------------------------------------------------------------------------- lm

LM_DOC = """CPU PyTorch oracle for the ZONOS3 Mini release LM (GDN/global + FullAttnRes backbone).

python tools/dump_golden.py lm RELEASE/lm/rate30_20k_r2grpo -o out/mini/golden
  [--text TEXT | --input-ids ids.npy]
  [--speaker-embedding emb.npy | --reference-codes codes.npy] [--condition-buckets a,b,c]
  [--silence-prefill N] [--silence-frame registry.json]
  [--weight-type f32|f16] [--dtype f32|bf16] [--mode full|chunked|decode]
  [--chunk-size 8] [--prefill-tokens 8] [--state-in state.npz]

Prompt rows: [speaker row | reference prefix] [loudness][fmax][speaking_rate] BOS text EOS [silence rows].
After the 11-way embedding sum (captured as emb_sum) the speaker row / condition rows are REPLACED by
speaker_projection(vec) / condition_embeddings[i][bucket_i] (captured as emb_cond), then the weightless emb_norm.
--input-ids X.npy reads X.cond.json (row indices, buckets, speaker vector); input_ids.npy and
input_ids.cond.json are always written. Chunk/decode modes address the replaced rows by absolute position.

The forward is zonos3.reference.lm.Zonos3LM (capture= dict), loaded by tools/convert_gguf.py load_lm.
Captures every sublayer's residual route, GDN intermediates and final FLA[V,K]
recurrent/causal-convolution states, not just the last token. Chunk/decode modes
also dump each invocation in chunk_NNN/; root activations concatenate all rows.
state.npz includes global KV caches and absolute position for split-run continuation.
"""


def integer_array(path, columns):
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] != columns or arr.shape[0] == 0:
        raise ValueError(f"{path}: expected nonempty [rows,{columns}], got {arr.shape}")
    if not np.isfinite(arr).all() or not np.equal(arr, np.floor(arr)).all():
        raise ValueError(f"{path}: non-integral IDs")
    return arr.astype(np.int64)


def load_speaker_embedding(path, dim):
    vec = np.load(path, allow_pickle=False)
    if vec.dtype not in (np.float32, np.float64) or vec.size != dim or vec.shape not in ((dim,), (1, dim)):
        raise ValueError(f"{path}: expected float [{dim}] speaker embedding, got {vec.dtype} {vec.shape}")
    vec = vec.astype(np.float32).reshape(dim)
    if not np.isfinite(vec).all():
        raise ValueError(f"{path}: speaker embedding contains a non-finite value")
    return vec


def parse_buckets(text, counts):
    try:
        values = [int(v) for v in text.split(",")]
    except ValueError:
        values = []
    counts = [int(n) for n in counts]
    if len(values) != len(counts) or any(not 0 <= v <= n for v, n in zip(values, counts)):
        raise ValueError(
            f"--condition-buckets needs {len(counts)} comma-separated integers, each in 0..count for the "
            f"bucket counts {counts} (count = unknown)"
        )
    return values


def build_prompt(
    params, meta, tokenizer_json, text, reference_codes=None, speaker=None, buckets=None, silence_prefill=3
):
    """Mirror of zonos3_build_prompt: returns (ids [rows,11] int64, cond dict)."""
    from tokenizers import Tokenizer

    if speaker is not None and reference_codes:
        raise ValueError("a speaker embedding and a reference prefix are mutually exclusive")
    if not 0 <= silence_prefill <= 30:
        raise ValueError("silence prefill must be 0..30 frames")
    pad, text_vocab = params["audio_pad_id"], params["text_vocab"]
    buckets = list(meta["bucket_counts"]) if buckets is None else list(buckets)
    tokenizer = Tokenizer.from_str(tokenizer_json)
    text_ids = [2] + [i + 4 for i in tokenizer.encode(text, add_special_tokens=False).ids] + [3]
    text_rows = np.full((len(text_ids), 11), pad, dtype=np.int64)
    text_rows[:, 10] = text_ids
    allpad = np.full((1, 11), pad, dtype=np.int64)
    allpad[:, 10] = text_vocab
    leading, speaker_rows, speaker_row = [], 0, -1
    if speaker is not None:
        leading, speaker_rows, speaker_row = [allpad.copy()], 1, 0
    elif reference_codes:
        codes = integer_array(reference_codes, 10)
        if np.any(codes < 0) or np.any(codes >= 4096):
            raise ValueError("reference codes must be ordinary IDs in [0,4095]")
        reference = np.full((len(codes) + 9, 11), pad, dtype=np.int64)
        reference[:, 10] = text_vocab
        for cb in range(10):
            reference[cb : cb + len(codes), cb] = codes[:, cb]
        leading, speaker_rows = [reference], len(reference)
    silence = np.full((silence_prefill, 11), pad, dtype=np.int64)
    silence[:, 10] = text_vocab
    for k in range(silence_prefill):
        for cb in range(10):
            if k - cb >= 0:
                silence[k, cb] = meta["silence_frame"][cb]
    rows = np.concatenate(leading + [np.repeat(allpad, len(meta["bucket_counts"]), 0), text_rows, silence])
    cond = {
        "speaker_row": speaker_row,
        "speaker_embedding": speaker,
        "condition_row": speaker_rows,
        "condition_buckets": buckets,
        "speaker_rows": speaker_rows,
        "prefill_frames": silence_prefill,
    }
    return rows, cond


def cond_path(ids_path):
    p = Path(ids_path)
    return p.with_suffix(".cond.json") if p.suffix == ".npy" else Path(str(p) + ".cond.json")


def save_cond(path, cond):
    doc = {"speaker_row": cond["speaker_row"]}
    if cond["speaker_row"] >= 0:
        doc["speaker_embedding"] = [float(v) for v in cond["speaker_embedding"]]
    doc.update(
        condition_row=cond["condition_row"],
        condition_buckets=[int(b) for b in cond["condition_buckets"]],
        speaker_rows=cond["speaker_rows"],
        prefill_frames=cond["prefill_frames"],
    )
    Path(path).write_text(json.dumps(doc) + "\n")


def load_cond(path, params, meta):
    path = Path(path)
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    cond = {
        "speaker_row": int(doc["speaker_row"]),
        "speaker_embedding": None,
        "condition_row": int(doc["condition_row"]),
        "condition_buckets": [int(b) for b in doc["condition_buckets"]],
        "speaker_rows": int(doc["speaker_rows"]),
        "prefill_frames": int(doc["prefill_frames"]),
    }
    if cond["speaker_row"] >= 0:
        cond["speaker_embedding"] = np.asarray(doc["speaker_embedding"], dtype=np.float32)
        if (
            cond["speaker_embedding"].shape != (meta["speaker_dim"],)
            or not np.isfinite(cond["speaker_embedding"]).all()
        ):
            raise ValueError(f"{path}: invalid speaker embedding")
    if len(cond["condition_buckets"]) != len(meta["bucket_counts"]) or any(
        not 0 <= b <= n for b, n in zip(cond["condition_buckets"], meta["bucket_counts"])
    ):
        raise ValueError(f"{path}: invalid condition buckets")
    return cond


def cond_inputs(cond):
    """Zonos3LM.forward speaker_rows / condition_rows (absolute rows) of a build_prompt/load_cond dict."""
    if cond is None:
        return None, None
    speaker_rows = None
    if cond["speaker_row"] >= 0:
        speaker_rows = {cond["speaker_row"]: torch.from_numpy(np.asarray(cond["speaker_embedding"], dtype=np.float32))}
    condition_rows = {cond["condition_row"] + i: int(b) for i, b in enumerate(cond["condition_buckets"])}
    return speaker_rows, condition_rows


def kv_rounding(kv_dtype):
    """--kv-dtype f16: the global layers attend over F16-stored keys/values (k/v rounded through kv_dtype before
    the attention, as the native KV cache). Rounding is idempotent, so rounding the SDPA key/value operands equals
    storing the cache in F16."""
    if kv_dtype == torch.float32:
        return nullcontext()
    original = torch.nn.functional.scaled_dot_product_attention

    def sdpa(query, key, value, *args, **kwargs):
        return original(query, key.to(kv_dtype).to(key.dtype), value.to(kv_dtype).to(value.dtype), *args, **kwargs)

    return mock.patch.object(torch.nn.functional, "scaled_dot_product_attention", sdpa)


def round_weights_f16(model):
    """Round the matrices exactly as `convert_gguf.py lm --outtype f16` stores them (the GDN conv and temp stay f32)."""
    for name, param in model.named_parameters():
        if param.ndim >= 2 and not name.endswith("attention.conv.weight") and not name.endswith("attention.temp"):
            rounded = param.data.half()
            if not torch.isfinite(rounded).all():
                raise ValueError(f"F16 overflow in {name}")
            param.data = rounded.float()


def load_lm_state(path, cache, config, layers, dtype, kv_dtype):
    from zonos3.reference.lm import GDN_HEAD_DIM, GDN_HEADS, GDN_KERNEL

    with np.load(path, allow_pickle=False) as state:
        position = np.asarray(state["position"])
        if position.shape != () or int(position) < 0 or int(position) != position:
            raise ValueError("invalid cache position")
        cache.position = int(position)
        expected = {"position"}

        def entry(name, shape, tensor_dtype):
            arr = state[name]
            if arr.shape != shape or not np.isfinite(arr).all():
                raise ValueError(f"invalid cache {name}: expected {shape}, got {arr.shape}")
            expected.add(name)
            return torch.from_numpy(arr.copy()).to(tensor_dtype)

        for layer in range(layers):
            if config.uses_gdn(layer):
                cache.gdn_state[layer] = entry(
                    f"gdn_state_{layer}", (GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM), torch.float32
                )
                cache.conv_state[layer] = entry(
                    f"gdn_conv_state_{layer}", (3 * config.gdn_inner, GDN_KERNEL - 1), dtype
                )
            else:
                shape = (cache.position, config.n_kv_heads, config.head_dim)
                cache.kv[layer] = (entry(f"kv_k_{layer}", shape, kv_dtype), entry(f"kv_v_{layer}", shape, kv_dtype))
        if set(state.files) != expected:
            raise ValueError(f"unexpected/missing state entries: {set(state.files) ^ expected}")


def save_lm_state(path, cache, config, layers, kv_dtype):
    arrays = {}
    for layer in range(layers):
        if config.uses_gdn(layer):
            arrays[f"gdn_conv_state_{layer}"] = cache.conv_state[layer].float().numpy()
            arrays[f"gdn_state_{layer}"] = cache.gdn_state[layer].float().numpy()
        else:
            k, v = cache.kv[layer]
            arrays[f"kv_k_{layer}"] = k.to(kv_dtype).float().numpy()
            arrays[f"kv_v_{layer}"] = v.to(kv_dtype).float().numpy()
    np.savez(path, position=np.asarray(cache.position, dtype=np.int64), **arrays)


def save_captures(out, captures, ids, cond=None):
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "input_ids.npy", ids.astype(np.float32))
    if cond is not None:
        save_cond(out / "input_ids.cond.json", cond)
    for name, array in captures.items():
        np.save(out / (name + ".npy"), array)
    # Comparison uses this authoritative set, including every critical state capture.
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "architecture": "zonos3",
                "expected_tensors": sorted(captures),
                "shapes": {k: list(v.shape) for k, v in captures.items()},
            },
            indent=2,
        )
        + "\n"
    )


def add_lm_arguments(parser):
    parser.add_argument("lm_dir", help="LM directory (release lm/rate30_20k_r2grpo or lm/mect_60k)")
    parser.add_argument("--silence-frame", help="silence-frame registry JSON (default: release registries/)")
    parser.add_argument("--text", "--prompt", dest="text", default="Hello, world.")
    parser.add_argument("--input-ids", help="NPY [tokens,11] for exact prompt/continuation replay (reads X.cond.json)")
    parser.add_argument("--reference-codes", help="ordinary NPY [frames,10] codes prepended with delays")
    parser.add_argument("--speaker-embedding", help="float32 NPY [192] speaker embedding (speaker row)")
    parser.add_argument(
        "--condition-buckets", help="a,b,c bucket ids per table (0..count-1, count = unknown); default all unknown"
    )
    parser.add_argument("--silence-prefill", type=int, help="codec-silence rows after EOS (default 3, 0..30)")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    parser.add_argument(
        "--weight-type", choices=("f32", "f16"), default="f32", help="round matrices exactly as the GGUF exporter"
    )
    parser.add_argument("--kv-dtype", choices=("f32", "f16"), default="f32")
    parser.add_argument("--mode", choices=("full", "chunked", "decode"), default="full")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--prefill-tokens", type=int, default=8)
    parser.add_argument("--state-in", help="saved state.npz; input rows start at its absolute position")
    parser.add_argument("--state-out", help="defaults to OUT/state.npz")
    parser.add_argument("--n-layer-limit", type=int)
    parser.add_argument("--threads", type=int, default=4)


def run_lm(args, parser):
    from convert_gguf import load_lm
    from zonos3.reference.lm import LMCache

    if args.chunk_size < 1 or args.prefill_tokens < 0 or args.threads < 1:
        parser.error("chunk-size/threads must be positive and prefill-tokens nonnegative")
    shaping = (args.reference_codes, args.speaker_embedding, args.condition_buckets, args.silence_prefill)
    if args.input_ids and any(v is not None for v in shaping):
        parser.error(
            "--reference-codes/--speaker-embedding/--condition-buckets/--silence-prefill are only valid with a text prompt"
        )
    if args.speaker_embedding and args.reference_codes:
        parser.error("a speaker embedding and a reference prefix are mutually exclusive")
    torch.set_num_threads(args.threads)
    src = load_lm(args.lm_dir, silence_frame=args.silence_frame)
    model, params, meta = src.model, src.params, src.meta
    config = model.config
    if args.n_layer_limit is not None and not 1 <= args.n_layer_limit <= config.n_layers:
        parser.error(f"n-layer-limit must be in [1,{config.n_layers}]")
    layers = config.n_layers if args.n_layer_limit is None else args.n_layer_limit
    dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
    kv_dtype = torch.float32 if args.kv_dtype == "f32" else torch.float16
    model.cast(torch.float32)
    if args.weight_type == "f16":
        round_weights_f16(model)
    model.cast(dtype)
    if args.n_layer_limit is not None:
        # The forward runs config.n_layers layers; the final AttnRes query stays the checkpoint's last row.
        model.config = dataclasses.replace(config, n_layers=layers)
    if args.input_ids:
        ids = integer_array(args.input_ids, 11)
        cond = load_cond(cond_path(args.input_ids), params, meta)
        if cond is None:
            print(f"warning: {cond_path(args.input_ids)} missing; no speaker/condition rows are replaced")
    else:
        speaker = (
            load_speaker_embedding(args.speaker_embedding, meta["speaker_dim"]) if args.speaker_embedding else None
        )
        buckets = parse_buckets(args.condition_buckets, meta["bucket_counts"]) if args.condition_buckets else None
        silence = 3 if args.silence_prefill is None else args.silence_prefill
        ids, cond = build_prompt(
            params, meta, src.tokenizer, args.text, args.reference_codes, speaker, buckets, silence
        )
    if (
        np.any(ids[:, :10] < 0)
        or np.any(ids[:, :10] >= config.audio_vocab)
        or np.any(ids[:, 10] < 0)
        or np.any(ids[:, 10] >= config.text_rows)
    ):
        raise ValueError("input IDs outside embedding vocabulary")
    cache = LMCache()
    if args.state_in:
        load_lm_state(args.state_in, cache, config, layers, dtype, kv_dtype)
    if cache.position + len(ids) > config.max_seqlen:
        raise ValueError("input exceeds model context")
    if args.mode == "full":
        lengths = [len(ids)]
    elif args.mode == "chunked":
        lengths = [min(args.chunk_size, len(ids) - start) for start in range(0, len(ids), args.chunk_size)]
    else:
        prefill = min(args.prefill_tokens, len(ids))
        lengths = ([prefill] if prefill else []) + [1] * (len(ids) - prefill)
    speaker_rows, condition_rows = cond_inputs(cond)
    out, start, combined = Path(args.out), 0, {}
    for index, length in enumerate(lengths):
        part = ids[start : start + length]
        captures = {}
        with kv_rounding(kv_dtype):
            model(
                torch.from_numpy(part),
                cache,
                speaker_rows=speaker_rows,
                condition_rows=condition_rows,
                capture=captures,
            )
        if args.mode != "full":
            save_captures(out / f"chunk_{index:03d}", captures, part)
        for name, array in captures.items():
            if name.startswith(("gdn_state_", "gdn_conv_state_")):
                combined[name] = array
            else:
                combined.setdefault(name, []).append(array)
        start += length
    combined = {
        name: value if isinstance(value, np.ndarray) else np.concatenate(value, 0) for name, value in combined.items()
    }
    save_captures(out, combined, ids, cond)
    save_lm_state(args.state_out or out / "state.npz", cache, config, layers, kv_dtype)
    print(
        f"Saved {len(combined)} tensors from {len(model.state_dict())} weights for {len(ids)} rows, "
        f"{len(lengths)} invocation(s), position={cache.position} -> {out}"
    )


# ---------------------------------------------------------------------------------------------------- codec-decoder

CODEC_DECODER_DOC = """Decode sheared codes with the reference ZONOS3 Mini MR-iSTFT decoder and compare a ggml waveform.

    tools/dump_golden.py codec-decoder \\
        --generator <release>/autoencoder/generator.pt --head-dir <release>/autoencoder/heads/full_mr \\
        out/codes.npy -o out/zac-golden [--compare out/cpp.npy]

codes.npy is the zonos3-cli stream output [H, 10] (sheared: codebook c of frame t sits in row t + c; any
silence-prefill rows are the first stream rows and are decoded like the rest). The frame count matches
zac_decode: min(eos, H - 9), with eos read from <codes>.eos.npy (first value) if present.

The decoder is built exactly like minisgl's tokenizer/matched_codec.py: zonos3.codec ISTFTCodec(cfg["model"],
cfg["head"]), load_pretrained(generator), load_trainable_state_dict(head checkpoint "trainable"), CPU fp32, TF32 off.
audio.npy is the raw (pre-headroom-gain) decoder output [frames * 2940]. Intermediates for debugging the C++
port are saved next to it: trunk.npy [768, 16*frames] (decoder.8 features), y0.npy [738, 16*frames] and
y1.npy [2210, 16*frames] (per-branch head linear outputs, rows = log-magnitude | phase), s0.npy / s1.npy
[16*frames*184] (per-branch iSTFT output before the FIR mixer), mixer.npy (mixer output, 44160 Hz).
"""
EXTERNAL_HOP = 2940


def load_wav_or_npy(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).reshape(-1).astype(np.float32)
    with wave.open(str(path), "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2 or f.getframerate() != 44100:
            raise SystemExit("comparison WAV must be 44.1 kHz mono PCM16")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768


def load_codec_decoder(generator: Path, head_dir: Path):
    from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec

    configs = sorted(head_dir.glob("config_*.json"))
    if len(configs) != 1:
        raise SystemExit(f"{head_dir} must contain exactly one config_*.json")
    cfg = json.loads(configs[0].read_text())
    assert cfg.get("decoder", {}).get("type") != "original"
    gen = torch.load(generator, map_location="cpu", mmap=True, weights_only=True)
    assert gen["step"] == 600000 and gen["config"]["model"] == cfg["model"]
    model = ISTFTCodec(cfg["model"], cfg["head"])
    model.load_pretrained(gen["generator"])
    ck = torch.load(head_dir / "step_0050000.pt", map_location="cpu", mmap=True, weights_only=False)
    assert int(ck["step"]) == 50000
    model.load_trainable_state_dict(ck["trainable"])
    model.eval().requires_grad_(False)
    for name in ("encoder", "to_internal", "semantic_encoder"):  # matched_codec.prune_model
        if hasattr(model, name):
            try:
                delattr(model, name)
            except Exception:
                pass
    return model


def add_codec_decoder_arguments(ap):
    ap.add_argument("--generator", type=Path, required=True)
    ap.add_argument("--head-dir", type=Path, required=True)
    ap.add_argument("codes", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path)


def run_codec_decoder(args, parser):
    raw = np.rint(np.load(args.codes)).astype(np.int64)
    if raw.ndim != 2 or raw.shape[1] != 10:
        raise SystemExit("codes must have shape [H,10]")
    H = raw.shape[0]
    eos_file = Path(str(args.codes) + ".eos.npy")
    eos = int(np.load(eos_file).reshape(-1)[0]) if eos_file.exists() else -1
    end = min(eos if eos >= 0 else H, H - 9)
    if end <= 0:
        raise SystemExit("no code frames to decode")
    idx = np.stack([raw[c : c + end, c] for c in range(10)], axis=1)  # un-shear [end, 10]
    if ((idx < 0) | (idx >= 4096)).any():
        raise SystemExit("un-sheared codes contain pad/eos ids; decode fewer frames")
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "codes_proc.npy", idx.astype(np.int32))

    model = load_codec_decoder(args.generator, args.head_dir)

    taps: dict[str, np.ndarray] = {}
    model.stages[8].register_forward_hook(lambda m, i, o: taps.__setitem__("trunk", o[0].float().numpy()))
    for i, lin in enumerate(model.head.outs):
        lin.register_forward_hook(lambda m, inp, o, i=i: taps.__setitem__(f"y{i}", o[0].float().T.numpy()))
    model.head.mixer.register_forward_pre_hook(
        lambda m, inp: [taps.__setitem__(f"s{i}", inp[0][0, i].float().numpy()) for i in range(inp[0].shape[1])]
        and None
    )
    model.head.mixer.register_forward_hook(lambda m, inp, o: taps.__setitem__("mixer", o[0, 0].float().numpy()))

    # minisgl matched_codec.decode_model: [B, frames, codebooks] -> packed [B, 10, frames], fp32, TF32 off.
    packed = torch.from_numpy(idx).unsqueeze(0).permute(0, 2, 1).to(torch.int64)
    expected = packed.shape[-1] * EXTERNAL_HOP
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode():
            out = model(packed)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
    if out.shape != (1, 1, expected):
        raise SystemExit(f"codec output shape {tuple(out.shape)}; expected {expected} samples")
    if not torch.isfinite(out).all():
        raise SystemExit("nonfinite codec audio")
    audio = out.float().reshape(-1).numpy()
    np.save(args.out / "audio.npy", audio)
    for name, a in taps.items():
        np.save(args.out / f"{name}.npy", np.ascontiguousarray(a, dtype=np.float32))
    print(
        f"audio {audio.shape} ({end} frames) rms {float(np.sqrt(np.mean(audio**2))):.4f} "
        f"peak {float(np.abs(audio).max()):.4f}"
    )
    if args.compare:
        cpp = load_wav_or_npy(args.compare)
        if cpp.shape != audio.shape:
            raise SystemExit(f"audio length mismatch: Python {audio.size}, C++ {cpp.size}")
        a, b = audio.astype(np.float64), cpp.astype(np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"cos={cos:.8f} max_abs={np.max(np.abs(a - b)):.6g}")
        if cos < 0.999:
            raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------- codec-encoder

CODEC_ENCODER_DOC = """Dump the actual frozen zonos3.codec MossAutoencoder native dual recipe.

    tools/dump_golden.py codec-encoder --generator <release>/autoencoder/generator.pt \\
        --audio speech.wav -o out/zac-encoder-golden [--teacher DIR] [--samples N] [--compare codes.npy]

Requires the production CUDA BF16 teacher/encoder path; no substitute teacher,
folded-table inversion or independent approximation of reference preprocessing.
Pass the resulting audio.npy to zac-encoder-cli --audio ... --dump ... . All
intermediates use C-order [time,channels] matching the native diagnostic callback.
The SSL teacher defaults to the generator's sibling teacher/ directory (release autoencoder/ layout).
Qualify --samples 1, 2939, 2940, 2941, 882000, 882001 and 1323017 in separate
output directories to cover short, exact-hop and independent-lane boundaries.
--compare NATIVE_CODES_NPY requires exact agreement for every frame/codebook.
"""


def add_codec_encoder_arguments(ap):
    ap.add_argument("--generator", type=Path, required=True, help="release autoencoder/generator.pt (step 600000)")
    ap.add_argument("--teacher", type=Path, help="SSL teacher directory (default: <generator dir>/teacher)")
    ap.add_argument("--audio", type=Path, required=True, help="44100-Hz F32 mono NPY or an ffmpeg-readable file")
    ap.add_argument(
        "--samples",
        type=int,
        help="trim or repeat the real source to this length for short/chunk-boundary qualification",
    )
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="frozen production default is four independent lanes; use one for constrained CUDA devices",
    )
    ap.add_argument("--compare", type=Path, help="native ordinary [frames,10] codes NPY; require exact equality")


def run_codec_encoder(args, parser):
    if not torch.cuda.is_available():
        raise SystemExit("production golden encoding requires an allocated CUDA GPU")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import torchaudio
    import transformers
    from transformers import Wav2Vec2BertModel
    from zonos3.codec.codec_adapter import MossAutoencoder
    from zonos3.codec.data_processing.independent_batcher import IndependentBatcher
    from zonos3.codec.hybrid10_dual_encoding.model import MossCodec
    from zonos3.codec.s1dac.losses import SemanticTeacher

    generator = args.generator
    teacher_path = args.teacher or generator.parent / "teacher"
    state = torch.load(generator, map_location="cpu", mmap=True, weights_only=True)
    if state["step"] != 600000 or state["config"]["teacher"]["layer"] != 16:
        raise SystemExit("golden requires matched step600000/layer16 recipe")
    cfg = state["config"]["model"]
    # Torchaudio builds real resampling kernels during construction; meta
    # initialization cannot materialize their arithmetic or copy their values.
    model = MossCodec(cfg)
    model.load_state_dict(state["generator"], strict=True, assign=True)
    model.eval().requires_grad_(False)
    # Construct the adapter without constructor-only CUDA graph warmups.
    # _prepare, _external and _codes/_dual are the actual frozen reference methods.
    adapter = MossAutoencoder.__new__(MossAutoencoder)
    adapter.device = torch.device("cuda")
    adapter.resample = model.to_internal.cpu()
    del model.decoder, model.to_external, model.to_internal
    adapter.model = model.to(adapter.device)
    adapter.teacher = (
        SemanticTeacher(str(teacher_path), layer=16, dtype=torch.bfloat16)
        .to(adapter.device)
        .eval()
        .requires_grad_(False)
    )
    adapter.optimized = None
    del state
    if args.audio.suffix == ".npy":
        audio = np.load(args.audio, allow_pickle=False)
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        if audio.ndim != 1:
            raise SystemExit("audio NPY must be mono [samples] or [1,samples]")
        audio = np.asarray(audio, dtype=np.float32)
    else:
        raw = subprocess.check_output(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(args.audio),
                "-t",
                "60",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-f",
                "f32le",
                "-",
            ]
        )
        audio = np.frombuffer(raw, dtype="<f4").copy()
    if args.samples is not None:
        if args.samples < 1 or args.samples > 60 * 44100:
            raise SystemExit("--samples must lie in 1..2646000")
        if not audio.size:
            raise SystemExit("empty source audio")
        audio = np.tile(audio, (args.samples + audio.size - 1) // audio.size)[: args.samples].copy()
    if not 0 < audio.size <= 60 * 44100 or not np.isfinite(audio).all():
        raise SystemExit("expected finite nonempty mono audio of at most 60 seconds")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "audio.npy", np.ascontiguousarray(audio), allow_pickle=False)
    expected_tensors = []
    teacher_capture_tensors, teacher_precision = [], {}
    first_attention_backends = {}

    def save(name, value, channels_first=True, expected=True):
        value = value.detach().float().cpu()
        if value.ndim == 3:
            value = value[0].T if channels_first else value[0]
        elif value.ndim == 2 and channels_first:
            value = value.T
        np.save(out / (name + ".npy"), np.ascontiguousarray(value.numpy()), allow_pickle=False)
        (expected_tensors if expected else teacher_capture_tensors).append(name)

    packer = IndependentBatcher(
        torch=torch,
        device=adapter.device,
        sample_rate=44160,
        hop_length=2944,
        feature_dim=10,
        prepare=adapter._prepare,
        run=adapter._codes,
        batch_size=args.batch_size,
        chunk_seconds=20.0,
        continuation_context_seconds=10.0,
        window=32,
        output_dtype=np.int16,
        label="frozen_dual_golden",
        input_dtype=torch.float32,
        input_channels=2,
    )
    # Use the frozen scheduler's exact full-file prepare and piece placement.
    sample = SimpleNamespace(audio=audio[None], sample_rate=44100, error=None)
    with torch.inference_mode():
        record = packer._record(sample)
        if record.sample.error is not None:
            raise RuntimeError(record.sample.error)
        prepared = adapter._prepare(audio[None], 44100)
        save("internal_audio", prepared[0, 0].reshape(-1, 1), channels_first=False)
        chunks = []
        lane = [0]
        handles = []

        def p():
            return f"lane.{lane[0]}"

        def teacher_input_hook(mod, inp):
            save(p() + ".ssl.features", inp[0], False, expected=False)
            save(p() + ".ssl.attention_mask", inp[1][0], False, expected=False)

        def teacher_norm_hook(mod, inp, output):
            save(p() + ".ssl.feature_input", inp[0], False, expected=False)
            save(p() + ".ssl.feature_norm", output, False, expected=False)
            teacher_precision[p()] = {"feature_input": str(inp[0].dtype), "feature_norm": str(output.dtype)}

        def teacher_projection_hook(mod, inp, output):
            save(p() + ".ssl.feature_projection", output, False, expected=False)
            teacher_precision[p()]["feature_projection"] = str(output.dtype)

        def teacher_output_hook(mod, inp, output):
            save(p() + ".ssl", output, False)
            teacher_precision[p()]["output"] = str(output.dtype)

        handles.append(adapter.teacher.register_forward_pre_hook(teacher_input_hook))
        handles.append(adapter.teacher.model.feature_projection.layer_norm.register_forward_hook(teacher_norm_hook))
        handles.append(
            adapter.teacher.model.feature_projection.projection.register_forward_hook(teacher_projection_hook)
        )
        handles.append(adapter.teacher.register_forward_hook(teacher_output_hook))
        handles.append(
            adapter.model.encoder.register_forward_hook(lambda mod, inp, output: save(p() + ".acoustic", output))
        )
        handles.append(
            adapter.model.encoder[0].register_forward_hook(lambda mod, inp, output: save(p() + ".enc.in", output))
        )
        for b in range(2):
            block = adapter.model.encoder[1 + b]
            block = getattr(block, "module", block)
            for r in range(3):
                handles.append(
                    block.block[r].register_forward_hook(
                        lambda mod, inp, output, b=b, r=r: save(p() + f".enc.b{b}.res{r}", output)
                    )
                )
            handles.append(
                block.block[4].register_forward_hook(
                    lambda mod, inp, output, b=b: save(p() + f".enc.b{b}.conv", output)
                )
            )
        for s in range(5):
            stage = adapter.model.encoder[4 + 2 * s]
            handles.append(stage.register_forward_hook(lambda mod, inp, output, s=s: save(p() + f".stage.{s}", output)))
            handles.append(
                stage.input_proj.register_forward_hook(
                    lambda mod, inp, output, s=s: save(p() + f".stage.{s}.in", output, False)
                )
            )
            # Eval calls layer.forward directly, bypassing Module forward hooks.
            # The next norm1 input is the unchanged previous layer's output.
            for l in range(1, len(stage.layers)):
                handles.append(
                    stage.layers[l].norm1.register_forward_pre_hook(
                        lambda mod, inp, s=s, l=l: save(p() + f".stage.{s}.l{l - 1}", inp[0], False)
                    )
                )
            handles.append(
                stage.output_proj.register_forward_pre_hook(
                    lambda mod, inp, s=s, l=len(stage.layers) - 1: save(p() + f".stage.{s}.l{l}", inp[0], False)
                )
            )
        first_layer = adapter.model.encoder[4].layers[0]
        sdpa_active = [False]

        def begin_sdpa(mod, inp):
            sdpa_active[0] = True

        def sdpa_input(mod, inp):
            save(p() + ".stage.0.l0.sdpa", inp[0], False)
            sdpa_active[0] = False

        handles.append(first_layer.norm1.register_forward_pre_hook(begin_sdpa))
        handles.append(
            first_layer.norm1.register_forward_hook(
                lambda mod, inp, output: save(p() + ".stage.0.l0.norm1", output, False)
            )
        )
        handles.append(
            first_layer.qkv.register_forward_hook(lambda mod, inp, output: save(p() + ".stage.0.l0.qkv", output, False))
        )
        handles.append(first_layer.attention_out.register_forward_pre_hook(sdpa_input))
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        def trace_sdpa(query, key, value, *inputs, **kwargs):
            if sdpa_active[0]:
                mask = kwargs.get("attn_mask")
                # CUDA autocast registers SDPA as lower_precision_fp. Query its
                # real runtime selector on those same eligible operand casts,
                # but pass the original operands unchanged to the actual SDPA.
                q, k, v, bias = query, key, value, mask
                if torch.is_autocast_enabled("cuda"):
                    compute_dtype = torch.get_autocast_dtype("cuda")
                    q, k, v = (tensor.to(compute_dtype) for tensor in (q, k, v))
                    if bias is not None:
                        bias = bias.to(compute_dtype)
                with torch.autocast("cuda", enabled=False):
                    backend = torch.ops.aten._fused_sdp_choice.default(
                        q,
                        k,
                        v,
                        bias,
                        kwargs.get("dropout_p", 0.0),
                        kwargs.get("is_causal", False),
                        scale=kwargs.get("scale"),
                        enable_gqa=kwargs.get("enable_gqa", False),
                    )
                first_attention_backends.setdefault(p() + ".stage.0.l0", []).append(
                    {
                        "backend": int(backend),
                        "query_shape": list(q.shape),
                        "key_shape": list(k.shape),
                        "query_dtype": str(q.dtype),
                        "mask_dtype": str(bias.dtype) if bias is not None else None,
                    }
                )
            return original_sdpa(query, key, value, *inputs, **kwargs)

        def sem_hook(mod, inp, output):
            save(p() + ".pooled", inp[0])
            save(p() + ".semantic", output)

        handles.append(adapter.model.semantic_encoder.register_forward_hook(sem_hook))
        handles.append(
            adapter.model.semantic_head.register_forward_hook(
                lambda mod, inp, output: save(p() + ".semantic_prediction", output)
            )
        )
        quantizers = [
            adapter.model.quantizer.semantic_quantizer.quantizers[0],
            *adapter.model.quantizer.quantizer.quantizers,
        ]
        for c, q in enumerate(quantizers):

            def qhook(mod, inp, output, c=c):
                name = p() + f".rvq.{c}"
                # Actual dac.nn.quantize.VectorQuantize.forward returns
                # (z_q, commitment_loss, codebook_loss, indices, z_e).
                quantized, _, _, indices, projected = output
                if (
                    projected.dtype != torch.float32
                    or quantized.dtype != torch.float32
                    or projected.shape != (args.batch_size, 8, 300)
                    or indices.shape != (args.batch_size, 300)
                ):
                    raise RuntimeError(f"{name}: unexpected frozen DAC quantizer outputs")
                save(name + ".projected", projected)
                save(name + ".quantized", quantized)
                save(name + ".codes", indices[0].reshape(1, 1, -1))
                if c == 0:
                    save(p() + ".q1", output[0])
                elif c == 1:
                    save(p() + ".residual", inp[0])

            handles.append(q.register_forward_hook(qhook))
        pieces = []
        torch.nn.functional.scaled_dot_product_attention = trace_sdpa
        try:
            for i, piece in enumerate(record.pieces):
                lane[0] = i
                packed = packer._prepare_batch([record], [piece]).waveform.to(adapter.device)
                save(p() + ".audio", packed[:, 0:1])
                save(p() + ".teacher_audio", adapter._external(packed))
                codes = adapter._codes(packed)  # actual frozen encoder + teacher + Q1 + nine RVQs
                start = piece.output_start_frame
                chunk = codes[0, start : start + piece.output_frames].cpu().numpy().astype(np.int32)
                chunks.append(chunk)
                pieces.append(
                    {
                        "lane": i,
                        "context_frames": piece.context_frames,
                        "output_start_frame": start,
                        "output_frames": piece.output_frames,
                    }
                )
        finally:
            torch.nn.functional.scaled_dot_product_attention = original_sdpa
            for handle in handles:
                handle.remove()
        codes = np.concatenate(chunks, axis=0)
    expected = ((audio.size + 2939) // 2940, 10)
    if codes.shape != expected or codes.min() < 0 or codes.max() >= 4096:
        raise RuntimeError(f"actual frozen reference produced invalid shape/codes: {codes.shape}, expected {expected}")
    np.save(out / "codes.npy", np.ascontiguousarray(codes), allow_pickle=False)
    expected_tensors.append("codes")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "generator": str(generator),
                "step": 600000,
                "teacher": str(teacher_path),
                "teacher_layer": 16,
                "precision": "fp32_weights_bf16_encoder_teacher_semantic_modules_fp32_rvq",
                "recipe": "zonos3.codec.codec_adapter.MossAutoencoder._prepare/_dual;IndependentBatcher._record/_prepare_batch",
                "audio_samples": int(audio.size),
                "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "expected_tensors": expected_tensors,
                "teacher_capture_tensors": teacher_capture_tensors,
                "teacher_precision": teacher_precision,
                "first_attention_backends": first_attention_backends,
                "runtime": {
                    "python": sys.executable,
                    "torch": torch.__version__,
                    "torch_file": torch.__file__,
                    "transformers": transformers.__version__,
                    "transformers_file": transformers.__file__,
                    "teacher_implementation": sys.modules[Wav2Vec2BertModel.__module__].__file__,
                    "torchaudio": torchaudio.__version__,
                    "torchaudio_file": torchaudio.__file__,
                    "quantizer_implementation": sys.modules[type(quantizers[0]).__module__].__file__,
                },
                "sample_rate": 44100,
                "batch_size": args.batch_size,
                "codes_shape": list(codes.shape),
                "pieces": pieces,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote frozen dual encoder goldens: {codes.shape[0]} frames, {len(pieces)} padded lanes to {out}")
    if args.compare:
        native = np.load(args.compare, allow_pickle=False)
        if native.shape != codes.shape or not np.issubdtype(native.dtype, np.integer):
            raise SystemExit(f"native codes must be integer {codes.shape}, got {native.dtype} {native.shape}")
        mismatches = native != codes
        if mismatches.any():
            frame, codebook = np.argwhere(mismatches)[0]
            counts = mismatches.sum(axis=0).tolist()
            raise SystemExit(
                f"code mismatch: per-codebook={counts}; first frame={frame}, codebook={codebook}, "
                f"reference={codes[frame, codebook]}, native={native[frame, codebook]}"
            )
        print(f"exact native code agreement: {codes.size} codes across {len(pieces)} independent lanes")


# ------------------------------------------------------------------------------------------------------ ssl-teacher

SSL_TEACHER_DOC = """Dump the matched SemanticTeacher frontend and all sixteen hidden-state layers.

    tools/dump_golden.py ssl-teacher --teacher <release>/autoencoder/teacher --audio speech.wav -o out/ssl-golden

Use --dtype f32 for diagnostic arithmetic comparison. --dtype bf16 requires CUDA
and runs the production BF16 model inside the frozen adapter's CUDA autocast
context (LayerNorm/softmax and layer outputs are F32, not blanket BF16).
To isolate the graph from frontend differences, pass the dumped features.npy and
attention_mask.npy to native --features/--mask.
Audio is decoded on the external 44100-Hz grid, optionally zero-padded there,
then resampled exactly with torchaudio (not ffmpeg).

The authoritative pointwise Conv1d outputs are the actual cuDNN module results.
--diagnose-convolutions observes those results alongside BF16 Linear and
BF16-GEMM/F32-output counterfactuals using identical operands. Backend arithmetic
can differ despite matching external dtype boundaries; the manifest records
cuDNN settings, but not its internal execution-plan/kernel ID or reduction tree.
Counterfactuals never replace oracle expectations. Keep comparisons strict:
consumer-code identity alone does not establish whole-lane floating parity.
"""


def add_ssl_teacher_arguments(ap):
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument(
        "--pcm44100", type=Path, help="C-order float32 mono waveform NPY, including external lane padding"
    )
    source.add_argument("--features", type=Path, help="C-order float32 [T,160] NPY")
    ap.add_argument("--mask", type=Path, help="[T] 0/1 NPY for --features")
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument(
        "--pad-seconds", type=float, default=0, help="zero-pad external PCM lane to this exact duration; never truncate"
    )
    ap.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--diagnose-projection",
        action="store_true",
        help="also capture F32-GEMM and prematurely-rounded BF16-GEMM counterfactuals; not oracle expectations",
    )
    ap.add_argument(
        "--trace-first-layer",
        action="store_true",
        help="capture actual first-layer module and attention operation boundaries",
    )
    ap.add_argument(
        "--diagnose-convolutions",
        action="store_true",
        help="compare genuine first-layer pointwise Conv1d outputs with BF16 Linear and F32-output GEMMs on identical operands",
    )


def run_ssl_teacher(args, parser):
    import torchaudio
    import transformers
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

    if args.dtype == "bf16" and (not args.device.startswith("cuda") or not torch.cuda.is_available()):
        raise ValueError("production BF16 oracle requires an allocated CUDA GPU and --device cuda")
    if args.diagnose_convolutions and args.dtype != "bf16":
        raise ValueError("--diagnose-convolutions requires the CUDA BF16 production oracle")
    config = json.loads((args.teacher / "config.json").read_text())
    if (
        config["num_hidden_layers"] != 16
        or config["hidden_size"] != 1024
        or config["add_adapter"]
        or config["apply_spec_augment"]
    ):
        raise ValueError("expected the matched frozen sixteen-layer SemanticTeacher")
    args.out.mkdir(parents=True, exist_ok=True)
    fe = SeamlessM4TFeatureExtractor(feature_size=80, sampling_rate=16000, num_mel_bins=80, padding_value=1, stride=2)
    if args.features:
        features = np.load(args.features, allow_pickle=False)
        if features.ndim != 2 or features.shape[1] != 160 or features.dtype != np.float32:
            raise ValueError("features must be float32 [T,160]")
        mask = np.load(args.mask, allow_pickle=False) if args.mask else np.ones(features.shape[0], dtype=np.int32)
        if mask.shape != (features.shape[0],) or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError("mask must be nonempty [T] 0/1")
    else:
        if args.mask:
            raise ValueError("--mask is only valid with --features")
        if args.pcm44100:
            wav = np.load(args.pcm44100, allow_pickle=False)
            if wav.ndim != 1 or wav.dtype != np.float32:
                raise ValueError("PCM must be float32 mono [samples]")
        else:
            raw = subprocess.run(
                [
                    os.environ.get("ZONOS3_FFMPEG", "ffmpeg"),
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(args.audio),
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    "-t",
                    "60",
                    "-f",
                    "f32le",
                    "-",
                ],
                check=True,
                capture_output=True,
            ).stdout
            wav = np.frombuffer(raw, dtype="<f4").copy()
        if not len(wav) or len(wav) > 60 * 44100 or not np.isfinite(wav).all():
            raise ValueError("PCM must be finite, nonempty and at most 60 seconds")
        if args.pad_seconds:
            samples = int(round(args.pad_seconds * 44100))
            if samples < len(wav) or samples > 60 * 44100:
                raise ValueError("padding duration must include all PCM and remain within 60 seconds")
            wav = np.pad(wav, (0, samples - len(wav)))
        np.save(args.out / "wav44100.npy", wav)
        wav16 = torchaudio.transforms.Resample(44100, 16000)(torch.from_numpy(wav)).numpy()
        np.save(args.out / "wav16k.npy", wav16)
        fbank = fe._extract_fbank_features(wav16)
        np.save(args.out / "fbank.npy", fbank)
        normalized = (fbank - fbank.mean(0)[None]) / np.sqrt(fbank.var(0, ddof=1)[None] + 1e-7)
        np.save(args.out / "normalized_fbank.npy", normalized)
        encoded = fe(wav16, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        features = encoded.input_features[0].numpy()
        mask = encoded.attention_mask[0].numpy()
    features = np.ascontiguousarray(features, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=np.int32)
    np.save(args.out / "features.npy", features)
    # Float32 mask is convenient for the project's minimal native NPY reader.
    np.save(args.out / "attention_mask.npy", mask.astype(np.float32))
    dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
    # The recipe disables TF32; do not let GPU qualification silently change its
    # F32 GEMM/convolution arithmetic. BF16 still executes real BF16 PyTorch ops.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = Wav2Vec2BertModel.from_pretrained(args.teacher, dtype=dtype, local_files_only=True).to(args.device)
    model.eval().requires_grad_(False)
    expected_tensors = ["features", "attention_mask", "feature_input", "feature_norm", "feature_projection"]
    projection_diagnostics = {}

    def save(name, value):
        np.save(args.out / f"{name}.npy", np.ascontiguousarray(value[0].detach().float().cpu().numpy()))

    def norm_hook(module, inputs, output):
        save("feature_input", inputs[0])
        save("feature_norm", output)
        projection_diagnostics["feature_input_dtype"] = str(inputs[0].dtype)
        projection_diagnostics["feature_norm_dtype"] = str(output.dtype)

    def projection_hook(module, inputs, output):
        save("feature_projection", output)
        projection_diagnostics["feature_projection_dtype"] = str(output.dtype)
        if args.diagnose_projection:
            # Counterfactuals use the exact authentic projection input and weights,
            # not a replaced teacher. Keep them out of expected_tensors.
            with torch.autocast(args.device.split(":")[0], enabled=False):
                x = inputs[0].to(dtype)
                f32_gemm = torch.nn.functional.linear(x.float(), module.weight.float(), module.bias.float()).to(dtype)
                prebiased = (torch.matmul(x, module.weight.T) + module.bias).to(dtype)
            save("diagnostic_projection_f32_gemm", f32_gemm)
            save("diagnostic_projection_premature_rounding", prebiased)
            for name, values in (("f32_gemm", f32_gemm), ("premature_rounding", prebiased)):
                difference = values.float() - output.float()
                projection_diagnostics[name] = {
                    "mismatched": int(torch.count_nonzero(difference)),
                    "maxabs": float(difference.abs().max()),
                    "rmse": float(difference.square().mean().sqrt()),
                }

    handles = [
        model.feature_projection.layer_norm.register_forward_hook(norm_hook),
        model.feature_projection.projection.register_forward_hook(projection_hook),
    ]
    convolution_diagnostics = {}
    if args.diagnose_convolutions:

        def convolution_probe(module, inputs, output, name):
            # These are the actual masked/activated Conv1d operands, narrowed by
            # the same CUDA-autocast policy. Never substitute a counterfactual
            # for the real module's output or later teacher inputs.
            with torch.autocast("cuda", enabled=False):
                operands = inputs[0].to(module.weight.dtype).transpose(1, 2).contiguous()
                weight = module.weight[:, :, 0]
                flat = operands.reshape(-1, operands.shape[-1])
                bf16_linear = torch.nn.functional.linear(flat, weight)
                bf16_mm_f32 = torch.mm(flat, weight.T, out_dtype=torch.float32).to(module.weight.dtype)
                f32_mm = torch.mm(flat.float(), weight.T.float()).to(module.weight.dtype)
            reference = output.transpose(1, 2)
            save(f"diagnostic_{name}.input", operands)
            stats = {
                "input_dtype": str(inputs[0].dtype),
                "operand_dtype": str(operands.dtype),
                "weight_dtype": str(weight.dtype),
                "output_dtype": str(output.dtype),
            }
            for label, values in (
                ("bf16_linear", bf16_linear),
                ("bf16_mm_f32_output", bf16_mm_f32),
                ("f32_mm", f32_mm),
            ):
                values = values.reshape(reference.shape)
                save(f"diagnostic_{name}.{label}", values)
                difference = values.float() - reference.float()
                stats[label] = {
                    "mismatched": int(torch.count_nonzero(difference)),
                    "maxabs": float(difference.abs().max()),
                    "rmse": float(difference.square().mean().sqrt()),
                }
            convolution_diagnostics[name] = stats

        conv = model.encoder.layers[0].conv_module
        for name, module in (("enc.0.conv.pw1", conv.pointwise_conv1), ("enc.0.conv.pw2", conv.pointwise_conv2)):
            handles.append(
                module.register_forward_hook(
                    lambda mod, inp, output, name=name: convolution_probe(mod, inp, output, name)
                )
            )
    operation_dtypes = {}
    attention_active, attention_matmuls = [False], [0]
    originals = (torch.matmul, torch.einsum, torch.softmax)
    if args.trace_first_layer:

        def save_operation(name, value):
            save(name, value)
            expected_tensors.append(name)
            operation_dtypes[name] = str(value.dtype)

        def module_capture(module, name, channels_first=False, before=False):
            def record(value):
                save_operation(name, value.transpose(1, 2) if channels_first else value)

            if before:
                handles.append(module.register_forward_pre_hook(lambda mod, inp: record(inp[0])))
            else:
                handles.append(module.register_forward_hook(lambda mod, inp, output: record(output)))

        layer = model.encoder.layers[0]
        for index in (1, 2):
            ffn = getattr(layer, f"ffn{index}")
            module_capture(getattr(layer, f"ffn{index}_layer_norm"), f"enc.0.ffn{index}.norm")
            module_capture(ffn.intermediate_dense, f"enc.0.ffn{index}.up")
            module_capture(ffn.intermediate_act_fn, f"enc.0.ffn{index}.act")
            module_capture(ffn.output_dense, f"enc.0.ffn{index}.down")
        module_capture(layer.self_attn_layer_norm, "enc.0.residual.ffn1", before=True)
        module_capture(layer.self_attn_layer_norm, "enc.0.attn.norm")
        for name, module in (
            ("q", layer.self_attn.linear_q),
            ("k", layer.self_attn.linear_k),
            ("v", layer.self_attn.linear_v),
            ("o", layer.self_attn.linear_out),
        ):
            module_capture(module, f"enc.0.attn.{name}")
        conv = layer.conv_module
        module_capture(conv.layer_norm, "enc.0.residual.attn", before=True)
        module_capture(conv.layer_norm, "enc.0.conv.norm")
        module_capture(conv.pointwise_conv1, "enc.0.conv.pw1", channels_first=True)
        module_capture(conv.glu, "enc.0.conv.glu", channels_first=True)
        module_capture(conv.depthwise_conv, "enc.0.conv.dw", channels_first=True)
        module_capture(conv.depthwise_layer_norm, "enc.0.conv.dw_norm")
        module_capture(conv.activation, "enc.0.conv.act", channels_first=True)
        module_capture(conv.pointwise_conv2, "enc.0.conv.pw2", channels_first=True)
        module_capture(layer.ffn2_layer_norm, "enc.0.residual.conv", before=True)
        module_capture(layer.final_layer_norm, "enc.0.residual.ffn2", before=True)

        def matrix_capture(name, value):
            save_operation(name, value.transpose(1, 2).reshape(1, len(features), -1))

        def begin_attention(mod, inp):
            attention_active[0], attention_matmuls[0] = True, 0

        def end_attention(mod, inp, output):
            attention_active[0] = False

        handles.append(layer.self_attn.register_forward_pre_hook(begin_attention))
        handles.append(layer.self_attn.register_forward_hook(end_attention))

        # Temporary identity wrappers observe genuine source calls; neither their
        # operands nor return values are replaced. Restore them even on failure.
        def trace_matmul(*inputs, **kwargs):
            output = originals[0](*inputs, **kwargs)
            if attention_active[0]:
                if attention_matmuls[0] == 0:
                    matrix_capture("enc.0.attn.content", output)
                else:
                    save_operation("enc.0.attn.value", output.transpose(1, 2).reshape(1, len(features), -1))
                attention_matmuls[0] += 1
            return output

        def trace_einsum(*inputs, **kwargs):
            output = originals[1](*inputs, **kwargs)
            if attention_active[0]:
                matrix_capture("enc.0.attn.relative", output)
            return output

        def trace_softmax(*inputs, **kwargs):
            output = originals[2](*inputs, **kwargs)
            if attention_active[0]:
                matrix_capture("enc.0.attn.masked_scores", inputs[0])
                matrix_capture("enc.0.attn.probs", output)
            return output

        torch.matmul, torch.einsum, torch.softmax = trace_matmul, trace_einsum, trace_softmax
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.dtype == "bf16" else nullcontext()
    try:
        with torch.inference_mode(), autocast:
            result = model(
                input_features=torch.from_numpy(features)[None].to(args.device, dtype=dtype),
                attention_mask=torch.from_numpy(mask)[None].to(args.device, dtype=torch.bool),
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        torch.matmul, torch.einsum, torch.softmax = originals
        for handle in handles:
            handle.remove()
    if len(result.hidden_states) != 17 or result.last_hidden_state.shape[-1] != 1024:
        raise ValueError("teacher output is not sixteen layers of 1024 per-frame channels")
    for layer, state in enumerate(result.hidden_states):
        values = state[0].float().cpu().numpy()
        np.save(args.out / f"hidden_{layer}.npy", values)
        print(f"hidden_{layer}: {values.shape}", flush=True)
        expected_tensors.append(f"hidden_{layer}")
    np.save(args.out / "output.npy", result.last_hidden_state[0].float().cpu().numpy())
    expected_tensors.append("output")
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "teacher": str(args.teacher.resolve()),
                "precision": args.dtype,
                "device": args.device,
                "recipe": "zonos3.codec.codec_adapter.MossAutoencoder._dual CUDA autocast BF16"
                if args.dtype == "bf16"
                else "F32 diagnostic without autocast",
                "expected_tensors": expected_tensors,
                "projection_diagnostics": projection_diagnostics,
                "operation_dtypes": operation_dtypes,
                "convolution_diagnostics": convolution_diagnostics,
                "hidden_dtypes": [str(state.dtype) for state in result.hidden_states],
                "runtime": {
                    "python": sys.executable,
                    "torch": torch.__version__,
                    "torch_file": torch.__file__,
                    "transformers": transformers.__version__,
                    "transformers_file": transformers.__file__,
                    "teacher_implementation": sys.modules[Wav2Vec2BertModel.__module__].__file__,
                    "torchaudio": torchaudio.__version__,
                    "torchaudio_file": torchaudio.__file__,
                    "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                },
                "convolution_runtime": {
                    "cudnn_version": torch.backends.cudnn.version(),
                    "cudnn_enabled": torch.backends.cudnn.enabled,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"features: {features.shape}, valid={mask.sum()}/{len(mask)}, dtype={args.dtype}", flush=True)


# -------------------------------------------------------------------------------------------------- speaker-encoder

SPEAKER_ENCODER_DOC = """Reference dumps of the ZONOS3 Mini speaker encoder (ownmect_s2, MECT-B2) for spk-encoder.gguf parity.

    tools/dump_golden.py speaker-encoder <ckpt_0037.pth> <audio.wav> -o OUT [--registry PATH]

Mirrors George's embed_prompt_scp.py exactly: soundfile float32 read, channel mean, torchaudio.functional.resample
(file rate -> 16 kHz), crop to 60 s, then zonos3.models.mect_speaker.MECTFbankSpeakerEmbedding (Kaldi fbank on
wav*32768, VPR with CMN, MECT_B2, ASTP, BN, linear) in eval mode, CPU fp32. The model is built from the registry's
ownmect_s2 entry (default <ckpt>/../../../registries/speaker_encoder_registry.json, the release layout) with the given
checkpoint, loaded strict=True from ckpt["modules"]["spk_model"] (sha256 checked against the registry).

Writes (float32 .npy, row-major [rows, cols]) -- the names spk-encoder-cli --dump-dir writes:
  fbank [T,80] (pre-CMN), fbank_cmn [T,80], feat_1..feat_4 [T,2560] (per MECT stage output, flattened as
  MECT.forward does: permute(0,2,1,3).reshape(B,-1,T), transposed to [T,2560]), pooled [1,5120], embedding [1,192].
Extra reference files (not in manifest.expected_tensors, so compare_golden.py ignores them):
  wav16k.npy [N] (the 16 kHz input of the fbank; torchaudio.functional.resample of the file, as
  embed_prompt_scp.py), wav16k_module.npy [N] (torchaudio.transforms.Resample(44100,16000), the kernel stored in
  spk-encoder.gguf as frontend.resample_44100; only written for 44.1 kHz inputs; differs from wav16k by ~4e-6),
  wav_in.npy [M] (mono float32 input at the file's native rate), manifest.json.
"""
SPEAKER_NAMES = ["fbank", "fbank_cmn", "feat_1", "feat_2", "feat_3", "feat_4", "pooled", "embedding"]


def add_speaker_encoder_arguments(ap):
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("audio", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--registry", type=Path, default=None)


def run_speaker_encoder(args, parser):
    import soundfile as sf
    import torchaudio
    from zonos3.models.mect_speaker import MECTFbankSpeakerEmbedding, registry_path

    entry = json.loads((args.registry or registry_path()).read_text())["ownmect_s2"]
    model = MECTFbankSpeakerEmbedding(dict(entry, codec="ownmect_s2", checkpoint=str(args.checkpoint)), device="cpu")

    sample_rate = model.sample_rate
    sig, sr = sf.read(str(args.audio), dtype="float32")
    if sig.ndim == 2:
        sig = sig.mean(axis=1)
    sig = np.ascontiguousarray(sig, dtype=np.float32)
    wav = torch.from_numpy(sig)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav[: int(model.max_seconds * sample_rate)].float()

    captured = {}
    vpr = model.model

    def flat(x):  # MECT.forward: x.permute(0,2,1,3).reshape(B, -1, T)
        return x.permute(0, 2, 1, 3).reshape(x.size(0), -1, x.size(-1))

    for s in range(1, 5):
        getattr(vpr.feature_extractor, f"mect_block{s}").register_forward_hook(
            lambda mod, inp, out, s=s: captured.__setitem__(f"feat_{s}", flat(out)[0].t().contiguous())
        )
    vpr.encoding_layers.register_forward_hook(lambda mod, inp, out: captured.__setitem__("pooled", out.detach()))

    with torch.inference_mode():
        fbank = model.fbank(wav)  # [T, 80], wav * 32768 inside
        fbank_cmn = fbank - fbank.mean(dim=0, keepdim=True)
        emb = vpr(fbank[None].transpose(1, 2))  # VPR applies the CMN itself
    captured["fbank"] = fbank
    captured["fbank_cmn"] = fbank_cmn
    captured["embedding"] = emb
    # blocks 1..4 are Sequentials of MECT blocks, so the hooks above fire once per stage with the stage output.
    args.out.mkdir(parents=True, exist_ok=True)
    for name in SPEAKER_NAMES:
        v = captured[name].detach().float().cpu().numpy()
        if v.ndim == 1:
            v = v[None]
        if not np.isfinite(v).all():
            raise SystemExit(f"nonfinite {name}")
        np.save(args.out / f"{name}.npy", np.ascontiguousarray(v))
        print(f"{name:10s} {tuple(v.shape)}")
    np.save(args.out / "wav16k.npy", wav.numpy().astype(np.float32))
    np.save(args.out / "wav_in.npy", sig)
    if sr == 44100:
        mod16 = torchaudio.transforms.Resample(44100, sample_rate)(torch.from_numpy(sig))[: wav.numel()]
        np.save(args.out / "wav16k_module.npy", mod16.numpy().astype(np.float32))
        print(f"resample module vs functional max|diff| {float((mod16 - wav).abs().max()):.3g}")
    e = captured["embedding"].reshape(-1)
    print(
        f"embedding norm {float(e.norm()):.4f}; wav16k {wav.numel()} samples @ {sample_rate}; "
        f"input {sig.size} samples @ {sr}"
    )
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "expected_tensors": SPEAKER_NAMES,
                "sample_rate_in": int(sr),
                "n_in": int(sig.size),
                "n_16k": int(wav.numel()),
                "frames": int(fbank.shape[0]),
                "checkpoint": str(args.checkpoint),
                "audio": str(args.audio),
            },
            indent=1,
        )
    )


# ------------------------------------------------------------------------------------------------------------- main

SUBCOMMANDS = {
    "lm": (LM_DOC, add_lm_arguments, run_lm),
    "codec-decoder": (CODEC_DECODER_DOC, add_codec_decoder_arguments, run_codec_decoder),
    "codec-encoder": (CODEC_ENCODER_DOC, add_codec_encoder_arguments, run_codec_encoder),
    "ssl-teacher": (SSL_TEACHER_DOC, add_ssl_teacher_arguments, run_ssl_teacher),
    "speaker-encoder": (SPEAKER_ENCODER_DOC, add_speaker_encoder_arguments, run_speaker_encoder),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    parsers = {}
    for name, (doc, add_arguments, _) in SUBCOMMANDS.items():
        parsers[name] = commands.add_parser(
            name, help=doc.splitlines()[0], description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        add_arguments(parsers[name])
    args = parser.parse_args()
    SUBCOMMANDS[args.command][2](args, parsers[args.command])


if __name__ == "__main__":
    main()
