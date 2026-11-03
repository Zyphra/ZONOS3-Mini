#!/usr/bin/env python3
"""Export the ZONOS3 Mini GGUFs (hf download Zyphra/ZONOS3_Mini_GGUF) into the sharded web format (zonos3-web-v1).

Layout per model: <out>/<model>/manifest.json + shard-NNN.bin. Every tensor is 256-byte aligned, never straddles a
shard, and is stored in ggml memory order (ne0 fastest). bf16 tensors store raw bfloat16 bits (two per u32, element 2i
in the low half). q8/q4 tensors store their int8 quants (or nibbles) contiguously followed (at scales_offset) by f32
block scales.

Besides the models the export writes voices/ (default voices with their shipped reference codes), companions/ (the
steering bundle steer.gguf, parsed directly by the worker) and laci/ (alignment heads and their language index).
"""
import argparse
import hashlib
import json
import os
import shutil
import struct

import numpy as np
from zonos3.languages import CODES

SHARD_CAP = 64 << 20
ALIGN = 256

GGUF_F32, GGUF_F16, GGUF_Q4_0, GGUF_Q8_0, GGUF_F64 = 0, 1, 2, 8, 28
GGUF_TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
                   9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
                   15: "Q8_K", 20: "IQ4_NL", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
                   28: "F64", 30: "BF16"}

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- GGUF v3 reader

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}


class Tensor:
    def __init__(self, name, ne, typ, offset):
        self.name, self.ne, self.type, self.offset = name, ne, typ, offset

    @property
    def n(self):
        n = 1
        for d in self.ne:
            n *= d
        return n

    @property
    def nbytes(self):
        if self.type == GGUF_F32:
            return self.n * 4
        if self.type == GGUF_F16:
            return self.n * 2
        if self.type == GGUF_Q8_0:
            return self.n // 32 * 34
        if self.type == GGUF_Q4_0:
            return self.n // 32 * 18
        if self.type == GGUF_F64:
            return self.n * 8
        raise ValueError(self.type)


class Gguf:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(64 << 20)
        self._d, self._o = head, 0
        if self._rd("<4s") != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        ver = self._rd("<I")
        if ver != 3:
            raise SystemExit(f"{path}: unsupported GGUF version {ver}")
        nt, nkv = self._rd("<Q"), self._rd("<Q")
        self.kv = {}
        for _ in range(nkv):
            k = self._rs()
            self.kv[k] = self._rv(self._rd("<I"))
        self.tensors = {}
        self.order = []
        for _ in range(nt):
            name = self._rs()
            nd = self._rd("<I")
            ne = [self._rd("<Q") for _ in range(nd)]
            typ = self._rd("<I")
            off = self._rd("<Q")
            self.tensors[name] = Tensor(name, ne, typ, off)
            self.order.append(name)
        al = self.kv.get("general.alignment", 32)
        self.data_off = (self._o + al - 1) // al * al
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        del self._d

    def _rd(self, fmt):
        v = struct.unpack_from(fmt, self._d, self._o)
        self._o += struct.calcsize(fmt)
        return v[0]

    def _rs(self):
        n = self._rd("<Q")
        s = self._d[self._o:self._o + n].decode("utf-8", "replace")
        self._o += n
        return s

    def _rv(self, t):
        if t == 8:
            return self._rs()
        if t == 9:
            et, n = self._rd("<I"), self._rd("<Q")
            return [self._rv(et) for _ in range(n)]
        return self._rd(_SCALAR[t])

    def raw(self, name):
        t = self.tensors[name]
        o = self.data_off + t.offset
        return self.mm[o:o + t.nbytes]

    def arr(self, name):
        """Tensor as numpy array in C order shape = reversed(ne) (f32/f16/f64 only)."""
        t = self.tensors[name]
        dt = {GGUF_F32: np.float32, GGUF_F16: np.float16, GGUF_F64: np.float64}.get(t.type)
        if dt is None:
            raise SystemExit(f"unsupported tensor type {GGUF_TYPE_NAMES.get(t.type, t.type)} for {name}")
        return self.raw(name).view(dt).reshape(list(reversed(t.ne)))

    def check_types(self):
        for name in self.order:
            t = self.tensors[name].type
            if t not in (GGUF_F32, GGUF_F16, GGUF_Q4_0, GGUF_Q8_0, GGUF_F64):
                raise SystemExit(f"unsupported tensor type {GGUF_TYPE_NAMES.get(t, t)} for {name}")

    def meta(self):
        arch = self.kv.get("general.architecture", "")
        out = {}
        for k, v in self.kv.items():
            if k == "general.alignment":
                continue
            key = k[len(arch) + 1:] if arch and k.startswith(arch + ".") else k
            out[key] = v
        return out


# --------------------------------------------------------------------------- bf16 / q8_0 conversion

def bf16_bits(a):
    """ggml_fp32_to_bf16 (round to nearest, ties to even) of finite float32 values, as uint16 bits."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_round(a):
    """float32 values rounded to the nearest bf16 (what a BF16 cast followed by an F32 cast returns)."""
    return (bf16_bits(a).astype(np.uint32) << 16).view(np.float32)


def quantize_q8_0(a):
    """ggml quantize_row_q8_0_ref over a float array (ne0 % 32 == 0): 34-byte blocks (f16 d, 32 int8)."""
    x = np.ascontiguousarray(a, dtype=np.float32).reshape(-1, 32)
    amax = np.max(np.abs(x), axis=1)
    d = (amax / np.float32(127)).astype(np.float32)
    inv = np.zeros_like(d)
    np.divide(np.float32(1), d, out=inv, where=d != 0)
    v = (x * inv[:, None]).astype(np.float64)  # float32 product as ggml; |v| + 0.5 is exact in f64
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)  # roundf: half away from zero
    blocks = np.empty((x.shape[0], 34), np.uint8)
    blocks[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    blocks[:, 2:] = q.view(np.uint8)
    return blocks.reshape(-1)


# --------------------------------------------------------------------------- shard writer

class Writer:
    def __init__(self, out_dir, model, meta):
        self.dir = out_dir
        self.model = model
        self.meta = meta
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            if f.startswith("shard-") or f == "manifest.json":
                os.remove(os.path.join(out_dir, f))
        self.shards = []
        self.tensors = {}
        self.f = None
        self.pos = 0
        self.total = 0

    def _open(self):
        if self.f:
            self.f.close()
        name = f"shard-{len(self.shards):03d}.bin"
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
        self.tensors[name] = {"dtype": dtype, "shape": list(shape), "shard": len(self.shards) - 1,
                              "offset": off, "nbytes": nb, "scales_offset": soff, "scales_nbytes": snb}

    def add_q8(self, name, raw, shape):
        self._add_blocks(name, raw, shape, "q8", 34)

    def add_q8_from(self, name, arr, shape):
        """Quantize a float tensor (ggml order, ne0 % 32 == 0) to q8_0 as ggml quantize-cli does."""
        self.add_q8(name, quantize_q8_0(arr), shape)

    def add_bf16(self, name, arr, shape):
        """A float tensor (final memory order) rounded to bf16 as ggml_fp32_to_bf16_row does."""
        self.add(name, bf16_bits(arr), "bf16", shape)

    def add_q4(self, name, raw, shape):
        """q4_0: 16 bytes per 32-weight block, byte j = element j (low nibble) | element j+16 (high)."""
        self._add_blocks(name, raw, shape, "q4", 18)

    def finish(self):
        if self.f:
            self.f.close()
        man = {"format": "zonos3-web-v1", "model": self.model, "meta": self.meta,
               "shards": self.shards, "tensors": self.tensors}
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1)
        print(f"{self.model}: {len(self.tensors)} tensors, {len(self.shards)} shards, "
              f"{self.total / 1e9:.3f} GB")


def copy_plain(w, g, name):
    t = g.tensors[name]
    if t.type == GGUF_Q8_0:
        w.add_q8(name, g.raw(name), t.ne)
    elif t.type == GGUF_Q4_0:
        w.add_q4(name, g.raw(name), t.ne)
    elif t.type == GGUF_F16:
        w.add(name, g.arr(name), "f16", t.ne)
    else:
        w.add(name, g.arr(name).astype(np.float32), "f32", t.ne)


# --------------------------------------------------------------------------- backbone

def export_backbone(path, out):
    g = Gguf(path)
    g.check_types()
    if g.kv.get("general.architecture") != "zonos3" or g.kv.get("zonos3.residual.type") != "full_attnres":
        raise SystemExit(f"{path}: not a zonos3 GDN backbone GGUF")
    meta = g.meta()
    tok = meta.pop("tokenizer.json")
    with open(os.path.join(HERE, "..", "src", "superbpe8192.json"), "rb") as f:
        ref_hash = hashlib.sha256(f.read()).hexdigest()
    if hashlib.sha256(tok.encode("utf-8")).hexdigest() != ref_hash:
        raise SystemExit("backbone tokenizer differs from src/superbpe8192.json")
    w = Writer(os.path.join(out, "backbone"), "backbone", meta)
    ncb = g.kv["zonos3.n_codebooks"]
    nl = g.kv["zonos3.block_count"]
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
        types = {g.tensors[n].type for n in names}
        if types == {GGUF_Q8_0}:
            # each output row is 1024/32 contiguous blocks: row concat is a byte concat (q|k|v|gate)
            raw = np.concatenate([np.asarray(g.raw(n)) for n in names])
            rows = sum(g.tensors[n].ne[1] for n in names)
            w.add_q8(f"blk.{L}.attn_qkvg.weight", raw, [g.tensors[names[0]].ne[0], rows])
        elif types == {GGUF_F16}:
            mats = [g.arr(n).astype(np.float16) for n in names]
            cat = np.concatenate(mats, axis=0)
            w.add(f"blk.{L}.attn_qkvg.weight", cat, "f16", [cat.shape[1], cat.shape[0]])
        else:
            raise SystemExit(f"blk.{L}: mixed attention projection types")
        skip.update(names)
    for name in g.order:
        if name not in skip:
            copy_plain(w, g, name)
    w.finish()


# --------------------------------------------------------------------------- codec decoder (mossdual_istft)

DAC_HOST = {"resample.kernel", "head.mixer.weight", "head.window.0", "head.window.1"}


def is_dac_mat(n):
    """quantize.cpp is_dac_mat: the MOSS transformer, semantic-head and iSTFT-head matrices."""
    return (n.startswith("stage.") or n.startswith("sem_head.") or n.startswith("head.")) and \
        n.endswith(".weight") and ".dw." not in n and "norm" not in n and n != "head.mixer.weight"


def export_dac(path, out):
    g = Gguf(path)
    g.check_types()
    if g.kv.get("general.architecture") != "mossdual_istft":
        raise SystemExit(f"{path}: not a mossdual_istft codec GGUF")
    w = Writer(os.path.join(out, "dac"), "dac", g.meta())
    sem = g.arr("sem.table").astype(np.float16)
    w.add("sem.table", sem, "f16", [sem.shape[1], sem.shape[0]])
    del sem
    res = np.concatenate([g.arr(f"res.{c}.table") for c in range(9)], axis=0).astype(np.float16)
    w.add("res.tables", res, "f16", [res.shape[1], res.shape[0]])
    del res
    for name in g.order:
        if name == "sem.table" or (name.startswith("res.") and name.endswith(".table")):
            continue
        t = g.tensors[name]
        if t.type == GGUF_Q8_0:
            if not is_dac_mat(name) or len(t.ne) != 2:
                raise SystemExit(f"{name}: q8_0 is only supported for codec linear matrices")
            w.add_q8(name, g.raw(name), t.ne)
            continue
        a = g.arr(name)  # C-order shape reversed(ne)
        if name in DAC_HOST:
            w.add(name, a.astype(np.float32), "f32", t.ne, host=True)
        elif name.endswith(".dw.weight"):
            K, C = t.ne  # a: (C, K) -> [K][C]
            w.add(name, np.ascontiguousarray(a.T).astype(np.float32), "f32", [C, K])
        elif len(t.ne) == 2 and is_dac_mat(name) and t.ne[0] % 32 == 0:
            w.add_q8_from(name, a, t.ne)
        elif len(t.ne) >= 2 and t.ne[0] != 1:
            w.add(name, a.astype(np.float16), "f16", t.ne)
        else:
            w.add(name, a.astype(np.float32), "f32", t.ne)
    w.finish()


# --------------------------------------------------------------------------- reference encoder + teacher
#
# Both run the native BF16 recipe of servers/ggml/src/dac-encoder.cpp / ssl-encoder.cpp (see encoder.ts, ssl.ts):
# every matrix a BF16 GEMM reads is stored as bf16 (round to nearest even of the F32 checkpoint, exactly
# ggml_fp32_to_bf16), and the remaining tensors keep the values the native loaders give them.

def is_encoder_linear(n, ne):
    """dac_encoder_load linear_matrix: converted to a BF16 tensor (2-D stage matrices, the semantic projections and
    the ConvNeXt pointwise matrices)."""
    stage_matrix = n.startswith("stage.") and ".weight" in n and "norm" not in n
    return len(ne) == 2 and (stage_matrix or n in ("sem_enc.in.weight", "sem_head.out.weight") or
                             ".pw1.weight" in n or ".pw2.weight" in n)


def is_encoder_autocast(n):
    """dac_encoder_load autocast_parameter: F32 storage holding bf16-rounded values (conv/depthwise weights and every
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
    for name in g.order:
        t = g.tensors[name]
        if t.type != GGUF_F32:
            raise SystemExit(f"{name}: the reference encoder export needs the F32 dac-encoder.gguf")
        if name == "resample.kernel" or name.startswith("rvq."):
            w.add(name, g.arr(name).astype(np.float32), "f32", t.ne, host=True)
        elif is_encoder_linear(name, t.ne):
            w.add_bf16(name, g.arr(name), t.ne)
        elif name == "enc.in.weight":
            # K=7 causal conv on 1 channel as a 2-tap gemm over 16-sample rows:
            # y[16t+r][c] = sum_k W[k][c] x[16t+r+k-6]; tap 0 reads row t-1, tap 1 row t.
            a = g.arr(name).astype(np.float32).reshape(7, 64)  # W[k][c]
            rep = np.zeros((2, 16, 64, 16), np.float32)        # [tap][r][c][i]
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
            w.add(name, bf16_round(g.arr(name)), "f32", t.ne)
        elif len(t.ne) >= 2 and t.ne[0] != 1:  # [K][OC][IC] residual-unit convs
            w.add_bf16(name, g.arr(name), t.ne)
        elif is_encoder_autocast(name):
            w.add(name, bf16_round(g.arr(name)), "f32", t.ne)
        else:
            w.add(name, g.arr(name).astype(np.float32), "f32", t.ne)
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
    for name in g.order:
        t = g.tensors[name]
        if name.startswith("frontend."):
            w.add(name, g.arr(name).astype(np.float32), "f32", t.ne, host=True)
            continue
        if t.type != GGUF_F32:
            raise SystemExit(f"{name}: the teacher export needs the F32 ssl-teacher.gguf")
        if name.endswith(".conv.dw.weight"):
            K, C = t.ne  # a: (C, K) -> [K][C]
            w.add(name, np.ascontiguousarray(g.arr(name).T).astype(np.float32), "f32", [C, K])
        elif name.endswith(".attn.dist"):
            w.add(name, bf16_round(g.arr(name)), "f32", t.ne)
        elif name.endswith(".weight") and len(t.ne) == 2 and t.ne[0] != 1:
            w.add_bf16(name, g.arr(name), t.ne)
        else:
            w.add(name, g.arr(name).astype(np.float32), "f32", t.ne)
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
    for name in g.order:
        if name in done:
            continue
        t = g.tensors[name]
        stem, _, last = name.rpartition(".")
        if stem.endswith((".conv1", ".conv2")) and last.startswith("w") and last[1:].isdigit():
            a = 0
            while f"{stem}.w{a}" in g.tensors:
                a += 1
            slots = []
            for j in range(a):
                sj = f"{stem}.w{j}"
                if g.tensors[sj].ne[:3] != [3, 3, 1]:
                    raise SystemExit(f"{sj}: expected a [3,3,1,OC] grouped-conv slot")
                slots.append(g.arr(sj).astype(np.float32))  # (OC, 1, 3, 3)
                done.add(sj)
            cat = np.concatenate(slots, axis=1)            # (OC, a, 3, 3)
            w.add(stem + ".weight", cat, "f32", [3, 3, a, t.ne[3]])
            continue
        if name.endswith(".pos.dw.weight"):
            p = name[:-len("dw.weight")]
            K, C = t.ne                                     # a: (C, K)
            scale = g.arr(p + "bn.scale").astype(np.float32)
            wt = (g.arr(name).astype(np.float32) * scale[:, None]).T  # (K, C)
            w.add(name, np.ascontiguousarray(wt), "f32", [C, K])
            continue
        if name.endswith(".pos.dw.bias"):
            p = name[:-len("dw.bias")]
            b = g.arr(name).astype(np.float32) * g.arr(p + "bn.scale").astype(np.float32) + \
                g.arr(p + "bn.shift").astype(np.float32)
            w.add(name, b.astype(np.float32), "f32", t.ne)
            continue
        if ".pos.bn." in name:  # folded into pos.dw
            continue
        w.add(name, g.arr(name).astype(np.float32), "f32", t.ne, host=name in SPK_HOST)
    w.finish()


# --------------------------------------------------------------------------- default voices

AUDIO_EXTS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}


def export_voices(voices, out):
    """Each default-voice clip with its shipped <stem>.codes.npy; ids as servers/ggml/src/server/speakers.cpp."""
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
        codes_rel = rel[:len(rel) - len(fn)] + stem + ".codes.npy"
        src_codes = os.path.join(root, codes_rel)
        if not os.path.isfile(src_codes):
            print(f"voices: skipping {rel}: no {stem}.codes.npy")
            continue
        codes = np.load(src_codes)
        if codes.ndim != 2 or codes.shape[1] != 10 or not 1 <= codes.shape[0]:
            raise SystemExit(f"voice {rel}: reference codes have shape {codes.shape}, expected [T,10]")
        os.makedirs(os.path.dirname(os.path.join(vdir, rel)), exist_ok=True)
        shutil.copyfile(os.path.join(root, rel), os.path.join(vdir, rel))
        shutil.copyfile(src_codes, os.path.join(vdir, codes_rel))
        vid = "default_" + hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
        label = stem.replace("_", " ").replace("-", " ").strip() or fn
        entries.append({"id": vid, "label": label, "file": rel, "codes": codes_rel})
    with open(os.path.join(vdir, "voices.json"), "w") as f:
        json.dump(entries, f)
    print(f"voices: {len(entries)} entries")


# --------------------------------------------------------------------------- companions and LACI heads

COMPANIONS = ("steer.gguf",)
LACI_EN = "zonos3_mini_mect60k_heads.json"
LACI_ZH = "zonos3_mini_mect60k_heads_zh.json"


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
    """Heads for every language (servers/sgl/scripts/serve.sh DUB=1): the zh file for cmn, the English file else."""
    ldir = os.path.join(out, "laci")
    if os.path.isdir(ldir):
        shutil.rmtree(ldir)
    os.makedirs(ldir)
    for fn in sorted(os.listdir(laci)):
        if fn.endswith(".json"):
            shutil.copyfile(os.path.join(laci, fn), os.path.join(ldir, fn))
    for fn in (LACI_EN, LACI_ZH):
        if not os.path.isfile(os.path.join(ldir, fn)):
            raise SystemExit(f"missing {os.path.join(laci, fn)}")
    # Every server language (zonos3.languages) in its order, cmn (the zh heads) last.
    index = {lang: LACI_EN for lang in CODES if lang != "cmn"}
    index["cmn"] = LACI_ZH
    with open(os.path.join(ldir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"laci: {len(index)} languages")


# --------------------------------------------------------------------------- main

MODEL_FILES = ("zonos3-qat.gguf", "dac.gguf", "dac-encoder.gguf", "ssl-teacher.gguf", "spk-encoder.gguf") + COMPANIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="directory with the ZONOS3 Mini GGUFs")
    ap.add_argument("--voices", required=True, help="default voice directory (clips + <stem>.codes.npy)")
    ap.add_argument("--laci", required=True, help="LACI alignment head directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for fn in MODEL_FILES:
        path = os.path.join(a.models, fn)
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}: hf download Zyphra/ZONOS3_Mini_GGUF --local-dir servers/ggml/models")
    m = a.models
    os.makedirs(a.out, exist_ok=True)
    export_backbone(os.path.join(m, "zonos3-qat.gguf"), a.out)
    export_dac(os.path.join(m, "dac.gguf"), a.out)
    export_encoder(os.path.join(m, "dac-encoder.gguf"), a.out)
    export_teacher(os.path.join(m, "ssl-teacher.gguf"), a.out)
    export_spk(os.path.join(m, "spk-encoder.gguf"), a.out)
    export_voices(a.voices, a.out)
    export_companions(m, a.out)
    export_laci(a.laci, a.out)


if __name__ == "__main__":
    main()
