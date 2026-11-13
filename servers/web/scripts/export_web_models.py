#!/usr/bin/env python3
"""Export the ZONOS3 Mini GGUFs (the gguf/ folder of Zyphra/ZONOS3-Mini) into the sharded web format (zonos3-web-v1).

Layout per model: <out>/<model>/manifest.json + shard-NNN.bin. Every tensor is 256-byte aligned, never straddles a
shard, and is stored in ggml memory order (ne0 fastest). bf16 tensors store raw bfloat16 bits (two per u32, element 2i
in the low half). q8/q4 tensors store their int8 quants (or nibbles) contiguously followed (at scales_offset) by f32
block scales.

Besides the models the export writes voices/ (default voices with their shipped reference codes), companions/ (the
steering bundle steer.gguf, parsed directly by the worker) and laci/ (alignment heads and their language index).
The speech-edit adapter edit/ (the edit vocabulary's grown rows and the edit/accents.gguf factors that the web backbone
applies at runtime on top of backbone/, see "speech-edit adapter" below) is checked against the MLX server's runtime
adapter by scripts/check_edit_export.py.
"""

import argparse
import hashlib
import json
import os
import shutil

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import quantize
from zonos3.tts.laci import discover_laci_heads
from zonos3.utils.hashing import sha256_bytes, sha256_file

SHARD_CAP = 64 << 20
ALIGN = 256

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- GGUF reader

FLOAT_DTYPES = {
    GGMLQuantizationType.F32: np.float32,
    GGMLQuantizationType.F16: np.float16,
    GGMLQuantizationType.F64: np.float64,
}
EXPORT_TYPES = (*FLOAT_DTYPES, GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0)


class Gguf:
    """A GGUF file (gguf.GGUFReader): ``kv`` the metadata as Python values, ``tensors`` name -> ReaderTensor in file
    order."""

    def __init__(self, path):
        self.path = path
        self.reader = GGUFReader(path)
        self.kv = {k: f.contents() for k, f in self.reader.fields.items() if not k.startswith("GGUF.")}
        self.tensors = {t.name: t for t in self.reader.tensors}

    def ne(self, name):
        """The tensor's ggml shape (ne0 first)."""
        return self.tensors[name].shape.tolist()

    def raw(self, name):
        t = self.tensors[name]
        return self.reader.data[t.data_offset : t.data_offset + t.n_bytes]

    def arr(self, name):
        """Tensor as numpy array in C order shape = reversed(ne) (f32/f16/f64 only)."""
        dt = FLOAT_DTYPES.get(self.tensors[name].tensor_type)
        if dt is None:
            raise SystemExit(f"unsupported tensor type {self.tensors[name].tensor_type.name} for {name}")
        return self.raw(name).view(dt).reshape(list(reversed(self.ne(name))))

    def check_types(self):
        for name, t in self.tensors.items():
            if t.tensor_type not in EXPORT_TYPES:
                raise SystemExit(f"unsupported tensor type {t.tensor_type.name} for {name}")

    def meta(self):
        arch = self.kv.get("general.architecture", "")
        out = {}
        for k, v in self.kv.items():
            if k == "general.alignment":
                continue
            key = k[len(arch) + 1 :] if arch and k.startswith(arch + ".") else k
            out[key] = v
        return out


# --------------------------------------------------------------------------- bf16 conversion


def bf16_bits(a):
    """ggml_fp32_to_bf16 (round to nearest, ties to even) of finite float32 values, as uint16 bits."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_round(a):
    """float32 values rounded to the nearest bf16 (what a BF16 cast followed by an F32 cast returns)."""
    return (bf16_bits(a).astype(np.uint32) << 16).view(np.float32)


# --------------------------------------------------------------------------- shard writer


class Writer:
    def __init__(self, out_dir, model, meta, shard_prefix="shard-"):
        self.dir = out_dir
        self.model = model
        self.meta = meta
        self.prefix = shard_prefix
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            if f.startswith("shard-") or f.startswith(shard_prefix) or f == "manifest.json":
                os.remove(os.path.join(out_dir, f))
        self.shards = []
        self.tensors = {}
        self.f = None
        self.pos = 0
        self.total = 0

    def _open(self):
        if self.f:
            self.f.close()
        name = f"{self.prefix}{len(self.shards):03d}.bin"
        self.shards.append(name)
        self.f = open(os.path.join(self.dir, name), "wb")
        self.pos = 0

    def _pad(self):
        p = (-self.pos) % ALIGN
        if p:
            self.f.write(b"\0" * p)
            self.pos += p

    def _reserve(self, nbytes):
        if self.f is None or (self.pos > 0 and self.pos + nbytes > SHARD_CAP):
            self._open()
        self._pad()

    def _write(self, data):
        b = memoryview(np.ascontiguousarray(data)).cast("B")
        off = self.pos
        self.f.write(b)
        self.pos += len(b)
        self.total += len(b)
        return off, len(b)

    def add(self, name, arr, dtype, shape, host=False):
        """arr: numpy array already in final memory order and dtype."""
        arr = np.ascontiguousarray(arr)
        exp = {"f32": np.float32, "f16": np.float16, "bf16": np.uint16}[dtype]
        assert arr.dtype == exp, (name, arr.dtype, dtype)
        n = 1
        for d in shape:
            n *= d
        assert arr.size == n, (name, arr.shape, shape)
        self._reserve(arr.nbytes)
        off, nb = self._write(arr)
        e = {"dtype": dtype, "shape": list(shape), "shard": len(self.shards) - 1, "offset": off, "nbytes": nb}
        if host:
            e["host"] = True
        self.tensors[name] = e

    def _add_blocks(self, name, raw, shape, dtype, bsize):
        """ggml block-quantized tensor: quants (ggml order) contiguous, then f32 scales at scales_offset."""
        n = 1
        for d in shape:
            n *= d
        blocks = np.asarray(raw).reshape(n // 32, bsize)
        d = blocks[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
        qs = np.ascontiguousarray(blocks[:, 2:]).reshape(-1)
        need = qs.nbytes + ALIGN + d.nbytes
        self._reserve(need)
        off, nb = self._write(qs)
        self._pad()
        soff, snb = self._write(d)
        self.tensors[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "shard": len(self.shards) - 1,
            "offset": off,
            "nbytes": nb,
            "scales_offset": soff,
            "scales_nbytes": snb,
        }

    def add_q8(self, name, raw, shape):
        self._add_blocks(name, raw, shape, "q8", 34)

    def add_q8_from(self, name, arr, shape):
        """Quantize a float tensor (ggml order, ne0 % 32 == 0) to q8_0 as ggml quantize-cli does."""
        self.add_q8(name, quantize(np.asarray(arr, dtype=np.float32), GGMLQuantizationType.Q8_0), shape)

    def add_bf16(self, name, arr, shape):
        """A float tensor (final memory order) rounded to bf16 as ggml_fp32_to_bf16_row does."""
        self.add(name, bf16_bits(arr), "bf16", shape)

    def add_q4(self, name, raw, shape):
        """q4_0: 16 bytes per 32-weight block, byte j = element j (low nibble) | element j+16 (high)."""
        self._add_blocks(name, raw, shape, "q4", 18)

    def finish(self):
        if self.f:
            self.f.close()
        man = {
            "format": "zonos3-web-v1",
            "model": self.model,
            "meta": self.meta,
            "shards": self.shards,
            "tensors": self.tensors,
        }
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1)
        print(f"{self.model}: {len(self.tensors)} tensors, {len(self.shards)} shards, {self.total / 1e9:.3f} GB")


def copy_plain(w, g, name):
    qtype, ne = g.tensors[name].tensor_type, g.ne(name)
    if qtype == GGMLQuantizationType.Q8_0:
        w.add_q8(name, g.raw(name), ne)
    elif qtype == GGMLQuantizationType.Q4_0:
        w.add_q4(name, g.raw(name), ne)
    elif qtype == GGMLQuantizationType.F16:
        w.add(name, g.arr(name), "f16", ne)
    else:
        w.add(name, g.arr(name).astype(np.float32), "f32", ne)


# --------------------------------------------------------------------------- backbone
#
# backbone/ is the release LM (backbone-ptq.gguf).


def check_tokenizer(meta, path):
    """Pops tokenizer.json from a backbone meta; it must be src/superbpe8192.json."""
    tok = meta.pop("tokenizer.json")
    ref_hash = sha256_file(os.path.join(HERE, "..", "src", "superbpe8192.json"))
    if sha256_bytes(tok.encode("utf-8")) != ref_hash:
        raise SystemExit(f"{path}: backbone tokenizer differs from src/superbpe8192.json")


def open_backbone(path):
    g = Gguf(path)
    g.check_types()
    if g.kv.get("general.architecture") != "zonos3" or g.kv.get("zonos3.residual.type") != "full_attnres":
        raise SystemExit(f"{path}: not a zonos3 GDN backbone GGUF")
    return g


def export_backbone(path, out):
    g = open_backbone(path)
    meta = g.meta()
    check_tokenizer(meta, path)
    ncb = g.kv["zonos3.n_codebooks"]
    nl = g.kv["zonos3.block_count"]
    w = Writer(os.path.join(out, "backbone"), "backbone", meta)
    parts = [g.arr(f"audio_embd.{k}.weight").astype(np.float16) for k in range(ncb)]
    parts.append(g.arr("text_embd.weight").astype(np.float16))
    embd = np.concatenate(parts, axis=0)
    w.add("embd_all", embd, "f16", [embd.shape[1], embd.shape[0]])
    del embd, parts
    skip = {f"audio_embd.{k}.weight" for k in range(ncb)} | {"text_embd.weight"}
    # global-attention layers: fuse q | k | v | gate into one projection (GDN layers keep gdn_in;
    # gdn_conv.weight stays f32 C-order [1536][4] via copy_plain)
    for L in range(nl):
        if f"blk.{L}.attn_q.weight" not in g.tensors:
            continue
        names = [f"blk.{L}.attn_{s}.weight" for s in ("q", "k", "v", "gate")]
        types = {g.tensors[n].tensor_type for n in names}
        if types == {GGMLQuantizationType.Q8_0}:
            # each output row is 1024/32 contiguous blocks: row concat is a byte concat (q|k|v|gate)
            raw = np.concatenate([np.asarray(g.raw(n)) for n in names])
            rows = sum(g.ne(n)[1] for n in names)
            w.add_q8(f"blk.{L}.attn_qkvg.weight", raw, [g.ne(names[0])[0], rows])
        elif types == {GGMLQuantizationType.F16}:
            cat = np.concatenate([g.arr(n).astype(np.float16) for n in names], axis=0)
            w.add(f"blk.{L}.attn_qkvg.weight", cat, "f16", [cat.shape[1], cat.shape[0]])
        else:
            raise SystemExit(f"blk.{L}: mixed attention projection types")
        skip.update(names)
    for name in g.tensors:
        if name in skip:
            continue
        if name == "output.weight":
            # web only: the 84 MB F16 logits head as q8_0 (half the bytes of every decode step)
            w.add_q8_from(name, g.arr(name), g.ne(name))
        else:
            copy_plain(w, g, name)
    w.finish()


# --------------------------------------------------------------------------- speech-edit adapter
#
# edit/ turns backbone/ into the speech-edit LM at runtime (src/backbone.ts Backbone.withEdit), as the native servers
# apply the adapter (servers/mlx/src/zonos3_mlx/companions.py AccentState, servers/ggml/src/zonos3-accent.h):
#   - the edit vocabulary: backbone-ptq-edit.gguf is backbone-ptq.gguf with grown tables (span_end_id = audio pad + 1 in
#     every audio table and logits-head block, text_extra_tokens infill-marker rows after the text pad); edit/ keeps
#     only the grown rows: embd_extra (f16 [n_codebooks audio rows, then the marker rows]) and output_extra (q8_0 as
#     backbone/'s head, one span-end row per codebook block). Every other row and tensor must equal backbone/'s;
#   - the zonos-accent-lora/3 adapter of edit/accents.gguf at strength 1, as f16 factors (rms relative rounding error
#     below 3e-4): every adapted linear <base> adds up @ (down @ x) (<base>.lora_down [K, in], .lora_up [out, K];
#     the global layers' q|k|v|gate factors stacked as blk.L.attn_qkvg: downs row-concatenated, ups block-diagonal;
#     the head factor zero-filled at the span-end rows when it predates the grown vocabulary, as AccentBank.load fits
#     it); every routed (token, expert e) of an adapted stack adds up[e] @ (down[e] @ row) to its gate/up
#     pre-activations (blk.L.ffn_gate_up_exps: downs [E, 2r, in] gate rows then up rows, ups [E, 2, out, r]) and to its
#     expert output (blk.L.ffn_down_exps: [E, r, in], [E, out, r]);
#   - each adapted router's balancing bias, replaced by router_bias + shift (f32).
# The manifest meta is the edit GGUF's (its vocabulary) plus text_extra_tokens and adapter.{base,name,rank}; adapter.base
# is backbone/'s general.name, which the runtime requires.


def edit_adapter(path, ncb, head_rows):
    """The zonos-accent-lora/3 slot of an accents.gguf: (name, rank, {base: (down, up)} f32 factors, {layer: shift})."""
    a = Gguf(path)
    a.check_types()
    if a.kv.get("general.architecture") != "zonos3_accents":
        raise SystemExit(f"{path}: not a zonos3_accents GGUF")
    names, formats = a.kv["zonos3_accents.names"], a.kv["zonos3_accents.format"]
    edits = [i for i, f in enumerate(formats) if f == "zonos-accent-lora/3"]
    if len(edits) != 1:
        raise SystemExit(f"{path}: expected one zonos-accent-lora/3 (speech-edit) adapter, found {len(edits)}")
    prefix = f"acc.{edits[0]}."
    parts, shifts = {}, {}
    for name in a.tensors:
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        if rest.endswith(".router_bias"):
            shifts[int(rest.split(".")[1])] = a.arr(name).astype(np.float32)
            continue
        base, _, part = rest.rpartition(".")
        parts.setdefault(base, {})[part] = a.arr(name).astype(np.float32)
    factors = {}
    for base, p in parts.items():
        down, up = p["lora_down"], p["lora_up"]
        if base == "output" and up.shape[0] == head_rows - ncb:
            k = up.shape[1]
            up = np.concatenate([up.reshape(ncb, -1, k), np.zeros((ncb, 1, k), np.float32)], axis=1).reshape(
                head_rows, k
            )
        factors[base] = (down, up)
    return names[edits[0]], int(a.kv["zonos3_accents.rank"][edits[0]]), factors, shifts


def grown_rows(g, ge, name, blocks):
    """The rows `name` of the edit GGUF `ge` adds per block (of `blocks`) over the release GGUF `g`, which must equal
    every other row: [blocks, extra, ...] in the edit tensor's dtype."""
    old, new = g.arr(name), ge.arr(name)
    old = old.reshape(blocks, -1, *old.shape[1:])
    new = new.reshape(blocks, -1, *new.shape[1:])
    n = old.shape[1]
    if new.shape[1] <= n or not np.array_equal(new[:, :n].view(np.uint16), old.view(np.uint16)):
        raise SystemExit(f"{name}: the edit GGUF does not grow the release tensor by trailing rows")
    return new[:, n:]


def export_edit(release, edit, adapter, out):
    g, ge = open_backbone(release), open_backbone(edit)
    meta = ge.meta()
    check_tokenizer(meta, edit)
    if "span_end_id" not in meta:
        raise SystemExit(f"{edit}: the speech-edit adapter needs the edit vocabulary (backbone-ptq-edit.gguf)")
    ncb = g.kv["zonos3.n_codebooks"]
    grown = {f"audio_embd.{k}.weight" for k in range(ncb)} | {"text_embd.weight", "output.weight"}
    if list(g.tensors) != list(ge.tensors):
        raise SystemExit(f"{edit}: tensor list differs from {release}")
    for name in g.tensors:
        if name in grown:
            if {g.tensors[name].tensor_type, ge.tensors[name].tensor_type} != {GGMLQuantizationType.F16}:
                raise SystemExit(f"{name}: expected F16")
        elif g.ne(name) != ge.ne(name) or not np.array_equal(g.raw(name), ge.raw(name)):
            raise SystemExit(f"{name}: differs between {release} and {edit}")
    audio = [grown_rows(g, ge, f"audio_embd.{k}.weight", 1)[0] for k in range(ncb)]
    text = grown_rows(g, ge, "text_embd.weight", 1)[0]
    head = grown_rows(g, ge, "output.weight", ncb).reshape(-1, g.ne("output.weight")[0])
    ax = audio[0].shape[0]
    if meta["audio_vocab"] != g.kv["zonos3.audio_vocab"] + ax or head.shape[0] != ncb * ax:
        raise SystemExit(f"{edit}: grown audio rows do not match audio_vocab")
    name, rank, factors, shifts = edit_adapter(adapter, ncb, ge.ne("output.weight")[1])
    for base in factors:
        if base + ".weight" not in g.tensors:
            raise SystemExit(f"{adapter}: adapter module {base} is not a tensor of {release}")
    for layer in shifts:
        if f"blk.{layer}.router_bias" not in g.tensors:
            raise SystemExit(f"{adapter}: router shift of layer {layer} has no router in {release}")
    meta["text_extra_tokens"] = text.shape[0]
    meta["adapter.base"] = g.kv["general.name"]
    meta["adapter.name"] = name
    meta["adapter.rank"] = rank

    # lora-NNN.bin: URLs the merged speech-edit LM never used, so a Cache Storage copy of its shard-NNN.bin cannot stand
    # in for them (CACHE_NAME, and so backbone/'s cached shards, stay as they are).
    w = Writer(os.path.join(out, "edit"), "edit", meta, shard_prefix="lora-")
    embd = np.concatenate(audio + [text], axis=0)
    w.add("embd_extra", embd, "f16", [embd.shape[1], embd.shape[0]])
    w.add_q8_from("output_extra", head, [head.shape[1], head.shape[0]])

    def f16(arr):
        return np.ascontiguousarray(arr, dtype=np.float16)

    def add(tname, arr):
        a16 = f16(arr)
        w.add(tname, a16, "f16", list(reversed(a16.shape)))

    done = set()
    for L in range(g.kv["zonos3.block_count"]):
        # q | k | v | gate as one rank-4K linear over the fused projection
        qkvg = [f"blk.{L}.attn_{s}" for s in ("q", "k", "v", "gate")]
        if any(b in factors for b in qkvg):
            rows = [g.ne(b + ".weight")[1] for b in qkvg]
            downs, k0 = [], 0
            for b in qkvg:
                if b in factors:
                    downs.append(factors[b][0])
            K = sum(d.shape[0] for d in downs)
            up = np.zeros((sum(rows), K), np.float32)
            r0 = 0
            for b, nr in zip(qkvg, rows):
                if b in factors:
                    d, u = factors[b]
                    up[r0 : r0 + nr, k0 : k0 + d.shape[0]] = u
                    k0 += d.shape[0]
                r0 += nr
            add(f"blk.{L}.attn_qkvg.lora_down", np.concatenate(downs, axis=0))
            add(f"blk.{L}.attn_qkvg.lora_up", up)
            done.update(qkvg)
        exps = {s: f"blk.{L}.ffn_{s}_exps" for s in ("gate", "up", "down")}
        if exps["gate"] in factors or exps["up"] in factors:
            if exps["gate"] not in factors or exps["up"] not in factors:
                raise SystemExit(f"blk.{L}: the adapter must shape both the gate and up experts")
            (dg, ug), (du, uu) = factors[exps["gate"]], factors[exps["up"]]
            add(f"blk.{L}.ffn_gate_up_exps.lora_down", np.concatenate([dg, du], axis=1))
            add(f"blk.{L}.ffn_gate_up_exps.lora_up", np.stack([ug, uu], axis=1))
            done.update((exps["gate"], exps["up"]))
    for base, (down, up) in factors.items():
        if base in done:
            continue
        add(base + ".lora_down", down)
        add(base + ".lora_up", up)
    for layer, shift in sorted(shifts.items()):
        bias = f"blk.{layer}.router_bias"
        w.add(bias, g.arr(bias).astype(np.float32) + shift, "f32", g.ne(bias))
    w.finish()


# --------------------------------------------------------------------------- codec decoder (mossdual_istft)

ZAC_HOST = {"resample.kernel", "head.mixer.weight", "head.window.0", "head.window.1"}


def is_zac_mat(n):
    """quantize.cpp is_zac_mat: the MOSS transformer, semantic-head and iSTFT-head matrices."""
    return (
        (n.startswith("stage.") or n.startswith("sem_head.") or n.startswith("head."))
        and n.endswith(".weight")
        and ".dw." not in n
        and "norm" not in n
        and n != "head.mixer.weight"
    )


def export_zac(path, out, name="zac"):
    """A mossdual_istft decoder GGUF into model directory `name` (the default keep2_plain head: `zac`; the full-size
    head: `zac-full_mr`). The engine reads the decoder id from the meta `head.variant`."""
    g = Gguf(path)
    g.check_types()
    if g.kv.get("general.architecture") != "mossdual_istft":
        raise SystemExit(f"{path}: not a mossdual_istft codec GGUF")
    if "mossdual_istft.head.variant" not in g.kv:
        raise SystemExit(f"{path}: no mossdual_istft.head.variant (decoder id)")
    w = Writer(os.path.join(out, name), name, g.meta())
    sem = g.arr("sem.table").astype(np.float16)
    w.add("sem.table", sem, "f16", [sem.shape[1], sem.shape[0]])
    del sem
    res = np.concatenate([g.arr(f"res.{c}.table") for c in range(9)], axis=0).astype(np.float16)
    w.add("res.tables", res, "f16", [res.shape[1], res.shape[0]])
    del res
    # Per-codebook residual biases (res.bias is their sum): early partial-frame decode drops a missing codebook's whole
    # term. Older codec GGUFs lack them; the web codec then decodes complete frames only.
    if all(f"res.{c}.bias" in g.tensors for c in range(9)):
        biases = np.concatenate([g.arr(f"res.{c}.bias") for c in range(9)], axis=0).astype(np.float32)
        w.add("res.biases", biases, "f32", [biases.size])
    for name, t in g.tensors.items():
        if name == "sem.table" or (name.startswith("res.") and name.endswith(".table")):
            continue
        if name.startswith("res.") and name.endswith(".bias") and name != "res.bias":
            continue
        ne = g.ne(name)
        if t.tensor_type == GGMLQuantizationType.Q8_0:
            if not is_zac_mat(name) or len(ne) != 2:
                raise SystemExit(f"{name}: q8_0 is only supported for codec linear matrices")
            w.add_q8(name, g.raw(name), ne)
            continue
        a = g.arr(name)  # C-order shape reversed(ne)
        if name in ZAC_HOST:
            w.add(name, a.astype(np.float32), "f32", ne, host=True)
        elif name.endswith(".dw.weight"):
            K, C = ne  # a: (C, K) -> [K][C]
            w.add(name, np.ascontiguousarray(a.T).astype(np.float32), "f32", [C, K])
        elif len(ne) == 2 and is_zac_mat(name) and ne[0] % 32 == 0:
            w.add_q8_from(name, a, ne)
        elif len(ne) >= 2 and ne[0] != 1:
            w.add(name, a.astype(np.float16), "f16", ne)
        else:
            w.add(name, a.astype(np.float32), "f32", ne)
    w.finish()


# --------------------------------------------------------------------------- reference encoder + teacher
#
# Both run the native BF16 recipe of servers/ggml/src/zac-encoder.cpp / ssl-encoder.cpp (see encoder.ts, ssl.ts):
# every matrix a BF16 GEMM reads is stored as bf16 (round to nearest even of the F32 checkpoint, exactly
# ggml_fp32_to_bf16), and the remaining tensors keep the values the native loaders give them.


def is_encoder_linear(n, ne):
    """zac_encoder_load linear_matrix: converted to a BF16 tensor (2-D stage matrices, the semantic projections and
    the ConvNeXt pointwise matrices)."""
    stage_matrix = n.startswith("stage.") and ".weight" in n and "norm" not in n
    return len(ne) == 2 and (
        stage_matrix or n in ("sem_enc.in.weight", "sem_head.out.weight") or ".pw1.weight" in n or ".pw2.weight" in n
    )


def is_encoder_autocast(n):
    """zac_encoder_load autocast_parameter: F32 storage holding bf16-rounded values (conv/depthwise weights and every
    non-norm bias). Norm parameters, Snake alphas and layer scales stay exact F32."""
    return "norm" not in n and (".weight" in n or ".bias" in n)


def export_encoder(path, out):
    """mossdual_encoder, native BF16 recipe. Linear matrices and convolution kernels (which the native encoder_conv
    casts to BF16 GEMM operands) are bf16; the acoustic front convs are re-laid out for the tiled gemm path (see
    encoder.ts). Depthwise weights and biases are f32 rounded to bf16 values; norms, Snake alphas, layer scales, the
    RVQ and the resampler are exact f32."""
    g = Gguf(path)
    g.check_types()
    if g.kv.get("general.architecture") != "mossdual_encoder":
        raise SystemExit(f"{path}: not a mossdual_encoder GGUF")
    w = Writer(os.path.join(out, "encoder"), "encoder", g.meta())
    for name, t in g.tensors.items():
        ne = g.ne(name)
        if t.tensor_type != GGMLQuantizationType.F32:
            raise SystemExit(f"{name}: the reference encoder export needs the F32 zac-encoder.gguf")
        if name == "resample.kernel" or name.startswith("rvq."):
            w.add(name, g.arr(name).astype(np.float32), "f32", ne, host=True)
        elif is_encoder_linear(name, ne):
            w.add_bf16(name, g.arr(name), ne)
        elif name == "enc.in.weight":
            # K=7 causal conv on 1 channel as a 2-tap gemm over 16-sample rows:
            # y[16t+r][c] = sum_k W[k][c] x[16t+r+k-6]; tap 0 reads row t-1, tap 1 row t.
            a = g.arr(name).astype(np.float32).reshape(7, 64)  # W[k][c]
            rep = np.zeros((2, 16, 64, 16), np.float32)  # [tap][r][c][i]
            for r in range(16):
                for i in range(16):
                    k0, k1 = i - 10 - r, i - r + 6
                    if 0 <= k0 <= 6:
                        rep[0, r, :, i] = a[k0]
                    if 0 <= k1 <= 6:
                        rep[1, r, :, i] = a[k1]
            w.add_bf16(name, rep.reshape(2, 1024, 16), [16, 1024, 2])
        elif name in ("enc.b0.conv.weight", "enc.b1.conv.weight"):
            # strided causal conv K=2s, left pad s, as a 2-tap gemm over rows of s input frames:
            # tap0[o][j*C+ic] = a[j][o][ic] (previous row), tap1[o][j*C+ic] = a[s+j][o][ic] (current row)
            a = g.arr(name).astype(np.float32)  # (2s, 2C, C)
            K2, OC, C = a.shape
            s = K2 // 2
            tap0 = np.transpose(a[:s], (1, 0, 2)).reshape(OC, s * C)
            tap1 = np.transpose(a[s:], (1, 0, 2)).reshape(OC, s * C)
            w.add_bf16(name, np.stack([tap0, tap1]), [s * C, OC, 2])
        elif name.endswith(".dw.weight"):  # already [K][C]
            w.add(name, bf16_round(g.arr(name)), "f32", ne)
        elif len(ne) >= 2 and ne[0] != 1:  # [K][OC][IC] residual-unit convs
            w.add_bf16(name, g.arr(name), ne)
        elif is_encoder_autocast(name):
            w.add(name, bf16_round(g.arr(name)), "f32", ne)
        else:
            w.add(name, g.arr(name).astype(np.float32), "f32", ne)
    w.finish()


def export_teacher(path, out):
    """w2vbert_ssl, native BF16 recipe: every linear weight and the relative-position table attn.dist (the operands
    ssl-encoder.cpp matmul() casts to BF16) are bf16 (attn.dist as f32 holding bf16 values, read by the attention
    kernel); LayerNorm parameters, biases and the depthwise conv stay exact f32; the frontend tables are host f32."""
    g = Gguf(path)
    g.check_types()
    if g.kv.get("general.architecture") != "w2vbert_ssl":
        raise SystemExit(f"{path}: not a w2vbert_ssl teacher GGUF")
    w = Writer(os.path.join(out, "teacher"), "teacher", g.meta())
    for name, t in g.tensors.items():
        ne = g.ne(name)
        if name.startswith("frontend."):
            w.add(name, g.arr(name).astype(np.float32), "f32", ne, host=True)
            continue
        if t.tensor_type != GGMLQuantizationType.F32:
            raise SystemExit(f"{name}: the teacher export needs the F32 ssl-teacher.gguf")
        if name.endswith(".conv.dw.weight"):
            K, C = ne  # a: (C, K) -> [K][C]
            w.add(name, np.ascontiguousarray(g.arr(name).T).astype(np.float32), "f32", [C, K])
        elif name.endswith(".attn.dist"):
            w.add(name, bf16_round(g.arr(name)), "f32", ne)
        elif name.endswith(".weight") and len(ne) == 2 and ne[0] != 1:
            w.add_bf16(name, g.arr(name), ne)
        else:
            w.add(name, g.arr(name).astype(np.float32), "f32", ne)
    w.finish()


# --------------------------------------------------------------------------- speaker encoder (MECT-B2 + ASTP)

SPK_HOST = {"frontend.window", "frontend.mel_banks", "frontend.resample_44100"}


def export_spk(path, out):
    """mect_spk, all f32. Grouped 3x3 conv slots w0..w{a-1} ([3,3,1,OC] each) become one conv{1,2}.weight
    [OC][a][3][3] (index ((oc*a + j)*3 + kh)*3 + kw; kw runs along time, kh along mel). pos.dw is transposed to
    [K][C] with its BatchNorm folded in (weight * scale, bias * scale + shift)."""
    g = Gguf(path)
    g.check_types()
    if g.kv.get("general.architecture") != "mect_spk":
        raise SystemExit(f"{path}: not a mect_spk speaker encoder GGUF")
    w = Writer(os.path.join(out, "spk"), "spk", g.meta())
    done = set()
    for name in g.tensors:
        if name in done:
            continue
        ne = g.ne(name)
        stem, _, last = name.rpartition(".")
        if stem.endswith((".conv1", ".conv2")) and last.startswith("w") and last[1:].isdigit():
            a = 0
            while f"{stem}.w{a}" in g.tensors:
                a += 1
            slots = []
            for j in range(a):
                sj = f"{stem}.w{j}"
                if g.ne(sj)[:3] != [3, 3, 1]:
                    raise SystemExit(f"{sj}: expected a [3,3,1,OC] grouped-conv slot")
                slots.append(g.arr(sj).astype(np.float32))  # (OC, 1, 3, 3)
                done.add(sj)
            cat = np.concatenate(slots, axis=1)  # (OC, a, 3, 3)
            w.add(stem + ".weight", cat, "f32", [3, 3, a, ne[3]])
            continue
        if name.endswith(".pos.dw.weight"):
            p = name[: -len("dw.weight")]
            K, C = ne  # a: (C, K)
            scale = g.arr(p + "bn.scale").astype(np.float32)
            wt = (g.arr(name).astype(np.float32) * scale[:, None]).T  # (K, C)
            w.add(name, np.ascontiguousarray(wt), "f32", [C, K])
            continue
        if name.endswith(".pos.dw.bias"):
            p = name[: -len("dw.bias")]
            b = g.arr(name).astype(np.float32) * g.arr(p + "bn.scale").astype(np.float32) + g.arr(
                p + "bn.shift"
            ).astype(np.float32)
            w.add(name, b.astype(np.float32), "f32", ne)
            continue
        if ".pos.bn." in name:  # folded into pos.dw
            continue
        w.add(name, g.arr(name).astype(np.float32), "f32", ne, host=name in SPK_HOST)
    w.finish()


# --------------------------------------------------------------------------- default voices

AUDIO_EXTS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}


def export_voices(voices, out):
    """Each default-voice clip with its shipped <stem>.codes.npy and <stem>.embedding.npy; ids as
    servers/ggml/src/server/speakers.cpp."""
    vdir = os.path.join(out, "voices")
    if os.path.isdir(vdir):
        shutil.rmtree(vdir)
    os.makedirs(vdir)
    root = os.path.realpath(voices)
    files = []
    for d, _, fs in os.walk(root):
        for fn in fs:
            files.append(os.path.relpath(os.path.join(d, fn), root).replace(os.sep, "/"))
    entries = []
    for rel in sorted(files):
        fn = rel.rsplit("/", 1)[-1]
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in AUDIO_EXTS:
            continue
        prefix = rel[: len(rel) - len(fn)] + stem
        codes_rel, emb_rel = prefix + ".codes.npy", prefix + ".embedding.npy"
        src_codes, src_emb = os.path.join(root, codes_rel), os.path.join(root, emb_rel)
        missing = [n for n, p in ((codes_rel, src_codes), (emb_rel, src_emb)) if not os.path.isfile(p)]
        if missing:
            print(f"voices: skipping {rel}: no {', '.join(missing)}")
            continue
        codes = np.load(src_codes)
        if codes.ndim != 2 or codes.shape[1] != 10 or not 1 <= codes.shape[0]:
            raise SystemExit(f"voice {rel}: reference codes have shape {codes.shape}, expected [T,10]")
        emb = np.load(src_emb, allow_pickle=False)
        if emb.dtype != np.float32 or emb.shape != (192,) or not np.isfinite(emb).all():
            raise SystemExit(
                f"voice {rel}: speaker embedding is {emb.dtype} {emb.shape}, expected finite float32 [192]"
            )
        os.makedirs(os.path.dirname(os.path.join(vdir, rel)), exist_ok=True)
        for r in (rel, codes_rel, emb_rel):
            shutil.copyfile(os.path.join(root, r), os.path.join(vdir, r))
        vid = "default_" + hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
        label = stem.replace("_", " ").replace("-", " ").strip() or fn
        entries.append({"id": vid, "label": label, "file": rel, "codes": codes_rel, "embedding": emb_rel})
    with open(os.path.join(vdir, "voices.json"), "w") as f:
        json.dump(entries, f)
    print(f"voices: {len(entries)} entries")


# --------------------------------------------------------------------------- companions and LACI heads

COMPANIONS = ("steer.gguf",)


def export_companions(models, out):
    """The companion GGUFs as is (the worker parses them); a stale companions/ is replaced."""
    cdir = os.path.join(out, "companions")
    if os.path.isdir(cdir):
        shutil.rmtree(cdir)
    os.makedirs(cdir)
    for fn in COMPANIONS:
        shutil.copyfile(os.path.join(models, fn), os.path.join(cdir, fn))
    print(f"companions: {', '.join(COMPANIONS)}")


def export_laci(laci, out):
    """Heads for every server language (zonos3.tts.laci.discover_laci_heads): the files and laci/index.json
    (language -> file)."""
    try:
        heads = discover_laci_heads(laci)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    ldir = os.path.join(out, "laci")
    if os.path.isdir(ldir):
        shutil.rmtree(ldir)
    os.makedirs(ldir)
    for path in sorted(set(heads.values())):
        shutil.copyfile(path, os.path.join(ldir, os.path.basename(path)))
    index = {lang: os.path.basename(path) for lang, path in heads.items()}
    with open(os.path.join(ldir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"laci: {len(index)} languages")


# --------------------------------------------------------------------------- main

EDIT_FILES = ("backbone-ptq-edit.gguf", "edit/accents.gguf")
MODEL_FILES = (
    ("backbone-ptq.gguf", "zac-keep2.gguf", "zac-encoder.gguf", "ssl-teacher.gguf", "spk-encoder.gguf")
    + COMPANIONS
    + EDIT_FILES
)
# The optional full-size decoder GGUF and its model directory (servers/web/src/worker.ts EXTRA_DECODERS).
FULL_DECODER = ("zac.gguf", "zac-full_mr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="directory with the ZONOS3 Mini GGUFs")
    ap.add_argument(
        "--voices", required=True, help="default voice directory (clips + <stem>.codes.npy + <stem>.embedding.npy)"
    )
    ap.add_argument("--laci", required=True, help="LACI alignment head directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for fn in MODEL_FILES:
        path = os.path.join(a.models, fn)
        if not os.path.isfile(path):
            raise SystemExit(
                f"missing {path}: hf download Zyphra/ZONOS3-Mini gguf/{fn} --local-dir servers/ggml"
            )
    m = a.models
    os.makedirs(a.out, exist_ok=True)
    export_backbone(os.path.join(m, "backbone-ptq.gguf"), a.out)
    export_edit(
        os.path.join(m, "backbone-ptq.gguf"), os.path.join(m, EDIT_FILES[0]), os.path.join(m, EDIT_FILES[1]), a.out
    )
    export_zac(os.path.join(m, "zac-keep2.gguf"), a.out)
    # The full-size decoder (full_mr), when the models directory has it: a request picks it with `decoder`.
    if os.path.isfile(os.path.join(m, FULL_DECODER[0])):
        export_zac(os.path.join(m, FULL_DECODER[0]), a.out, FULL_DECODER[1])
    elif os.path.isdir(os.path.join(a.out, FULL_DECODER[1])):
        shutil.rmtree(os.path.join(a.out, FULL_DECODER[1]))
    export_encoder(os.path.join(m, "zac-encoder.gguf"), a.out)
    export_teacher(os.path.join(m, "ssl-teacher.gguf"), a.out)
    export_spk(os.path.join(m, "spk-encoder.gguf"), a.out)
    export_voices(a.voices, a.out)
    export_companions(m, a.out)
    export_laci(a.laci, a.out)


if __name__ == "__main__":
    main()
