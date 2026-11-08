"""MLX codec decoder vs a float64 NumPy transcription of servers/ggml/src/zac.cpp (zac_decode: un-shear, trunk,
per-branch iSTFT with overlap-add, FIR mixer, 44160 -> 44100 resampler) on the dummy decoder fixture."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
from conftest import FIXTURES, dummy_codec
from gguf import GGUFReader
from zonos3_mlx.models.codec_decoder import CodecDecoder

GGUF = FIXTURES / "dummy-zac.gguf"
CODES = FIXTURES / "dummy-zac-codes.npy"


class Ref:
    """Every tensor of the GGUF in float64, numpy order."""

    def __init__(self, path):
        r = GGUFReader(str(path))
        self.t = {
            t.name: np.asarray(t.data, dtype=np.float64).reshape(tuple(reversed(t.shape.tolist()))) for t in r.tensors
        }
        self.meta = {}
        for key, f in r.fields.items():
            if key.startswith("mossdual_istft."):
                k = key.removeprefix("mossdual_istft.")
                if len(f.data) == 1 and f.types[0].name != "ARRAY":
                    v = f.parts[f.data[0]]
                    self.meta[k] = bytes(v).decode() if f.types[0].name == "STRING" else v[0].item()
                else:
                    self.meta[k] = [int(f.parts[i][0]) for i in f.data]

    def lin(self, name, x, bias=None):
        y = x @ self.t[name].T
        return y if bias is None else y + self.t[bias]


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def convnext(R, x, p, eps):
    T, C = x.shape
    w = R.t[p + ".dw.weight"]  # [C, K]
    K = w.shape[1]
    xf = np.concatenate([np.zeros((K - 1, C)), x])
    h = sum(xf[k : k + T] * w[:, k] for k in range(K)) + R.t[p + ".dw.bias"]
    h = layer_norm(h, R.t[p + ".norm.weight"], R.t[p + ".norm.bias"], eps)
    h = gelu(R.lin(p + ".pw1.weight", h, p + ".pw1.bias"))
    return x + R.lin(p + ".pw2.weight", h, p + ".pw2.bias")


def rope(x, positions):
    hd = x.shape[-1]
    freq = 10000.0 ** (-np.arange(0, hd, 2) / hd)
    ang = positions[:, None] * freq[None]
    cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2], out[..., 1::2] = a * cos - b * sin, a * sin + b * cos
    return out


def moss(R, x, p, n_head, context, eps):
    T, d = x.shape
    hd = d // n_head
    h = layer_norm(x, R.t[p + ".norm1.weight"], R.t[p + ".norm1.bias"], eps)
    qkv = R.lin(p + ".qkv.weight", h).reshape(T, 3, n_head, hd)
    pos = np.arange(T, dtype=np.float64)
    q, k, v = rope(qkv[:, 0], pos), rope(qkv[:, 1], pos), qkv[:, 2]
    o = np.zeros((T, n_head, hd))
    for t in range(T):
        lo = max(0, t - context + 1)
        for hh in range(n_head):
            s = k[lo : t + 1, hh] @ q[t, hh] / math.sqrt(hd)
            w = np.exp(s - s.max())
            o[t, hh] = (w / w.sum()) @ v[lo : t + 1, hh]
    x = x + R.lin(p + ".wo.weight", o.reshape(T, d)) * R.t[p + ".attn_scale"]
    h = layer_norm(x, R.t[p + ".norm2.weight"], R.t[p + ".norm2.bias"], eps)
    return x + R.lin(p + ".ffn2.weight", gelu(R.lin(p + ".ffn1.weight", h))) * R.t[p + ".ffn_scale"]


def reference_decode(R, codes):
    """The whole decode of ``codes``; a residual code of -1 leaves its table row and ``res.{i}.bias`` out."""
    m = R.meta
    F = len(codes)
    x = R.t["sem.table"][codes[:, 0]] + R.t["sem.bias"]
    for b in range(m["sem_head.depth"]):
        x = convnext(R, x, f"sem_head.{b}", m["convnext_eps"])
    x = R.lin("sem_head.out.weight", x, "sem_head.out.bias")
    missing = codes[:, 1:] < 0
    for c in range(1, 10):
        x = x + np.where(missing[:, c - 1 : c], 0.0, R.t[f"res.{c - 1}.table"][np.maximum(codes[:, c], 0)])
    x = x + R.t["res.bias"]
    if missing.any():
        x = x - missing @ np.stack([R.t[f"res.{i}.bias"] for i in range(9)])
    for s in range(5):
        p = f"stage.{s}"
        x = R.lin(p + ".in.weight", x)
        for layer in range(m["stage.n_layer"][s]):
            x = moss(R, x, f"{p}.l{layer}", m["stage.n_head"][s], m["stage.context"][s], m["ln_eps"])
        if m["stage.out_dim"][s] > 0:
            x = R.lin(p + ".out.weight", x)
        h = m["stage.patch"][s]
        if h > 1:  # out[t*h + j, ch] = in[t, ch*h + j]
            T, C = x.shape
            y = np.zeros((T * h, C // h))
            for t in range(T):
                for j in range(h):
                    y[t * h + j] = x[t, j::h]
            x = y
    for b in range(m["head.depth"]):
        x = convnext(R, x, f"head.{b}", m["convnext_eps"])
    x = layer_norm(x, R.t["head.norm.weight"], R.t["head.norm.bias"], m["convnext_eps"])
    hop = m["head.hop"]
    L = len(x) * hop
    branches = []
    for i, n in enumerate(m["head.n_fft"]):
        spec = R.lin(f"head.out.{i}.weight", x, f"head.out.{i}.bias")
        bins = n // 2 + 1
        win = R.t[f"head.window.{i}"]
        ola, env = np.zeros(len(x) * hop + n), np.zeros(len(x) * hop + n)
        for t, f in enumerate(spec):
            mag = np.exp(np.minimum(f[:bins], m["head.max_log_magnitude"]))
            X = mag * np.exp(1j * f[bins : 2 * bins])
            X[0], X[-1] = X[0].real, X[-1].real
            frame = np.fft.irfft(X, n=n)
            ola[t * hop : t * hop + n] += win * frame
            env[t * hop : t * hop + n] += win * win
        branches.append((ola / np.where(env > 0, env, 1.0))[n // 2 : n // 2 + L])
    mixer = R.t["head.mixer.weight"]  # [2, K]
    K = mixer.shape[1]
    hk = (K - 1) // 2
    mixed = np.zeros(L)
    for c in range(2):
        padded = np.concatenate([np.zeros(hk), branches[c], np.zeros(hk)])
        for k in range(K):
            mixed += mixer[c, k] * padded[k : k + L]
    O, N, W = m["resample.orig"], m["resample.new"], m["resample.width"]
    kernel = R.t["resample.kernel"]  # [N, 2W + O]
    total = -(-N * L // O)
    out = []
    for n in range(-(-total // N)):
        x0 = n * O - W
        idx = np.arange(x0, x0 + 2 * W + O)
        xw = np.where((idx >= 0) & (idx < L), mixed[np.clip(idx, 0, L - 1)], 0.0)
        out.extend((kernel @ xw)[: min(N, total - n * N)])
    return np.asarray(out[: F * m["hop"]])


def unshear(stream, eos):
    H = len(stream)
    E = min(eos, H - 9)
    codes = np.zeros((E, 10), dtype=np.int64)
    for c in range(10):
        for t in range(E):
            j = t + c
            codes[t, c] = np.clip(int(stream[j, c]) if j < H else 4097, 0, 4095)
    return codes


@pytest.fixture(scope="module")
def codes():
    stream = np.load(CODES).astype(np.int64)
    eos = int(np.load(str(CODES) + ".eos.npy")[0])
    return unshear(stream, eos)


@pytest.fixture(scope="module")
def biased(tmp_path_factory):
    """The dummy decoder with per-codebook residual biases (early partial-frame decode)."""
    return dummy_codec(tmp_path_factory.mktemp("zac") / "zac.gguf", residual_biases=True)


@pytest.fixture(scope="module")
def unbiased(tmp_path_factory):
    """The dummy decoder with only the summed ``res.bias`` (early partial-frame decode off)."""
    return dummy_codec(tmp_path_factory.mktemp("zac") / "zac.gguf", residual_biases=False)


def first_chunk_partial(codes, rows: int = 4):
    """``codes`` as the k = 1 first chunk after ``rows`` delayed rows leaves them: codebook j of frame f < rows is
    known once row f + j arrived, the rest missing (-1)."""
    partial = codes.copy()
    for f in range(rows):
        partial[f, rows - f :] = -1
    return partial


def test_decode_matches_the_float64_transcription(codes):
    got = CodecDecoder.load(GGUF).decode(codes)
    want = reference_decode(Ref(GGUF), codes)
    assert got.shape == want.shape == (len(codes) * 2940,)
    assert np.abs(want).max() > 1e-3, "degenerate reference"
    assert np.abs(got - want).max() <= 1e-4


def test_decode_rows_overlaps_short_prefills(codes):
    dec = CodecDecoder.load(GGUF)
    rows = np.full((len(codes) + 9, 11), 4097, dtype=np.int64)
    for c in range(10):
        rows[c : c + len(codes), c] = codes[:, c]
    prefill, gen = rows[:3], rows[3:]
    audio = dec.decode_rows(list(gen), prefill=prefill, eos_frame=None)
    full = dec.decode(codes)
    # 3 prefill rows <= the 4 overlap rows: the end-of-stream decode starts at the stream's first row, as the
    # reference (zonos3.reference.synth.decode_rows) does
    assert len(audio) == len(full)
    np.testing.assert_allclose(audio, full, atol=1e-5)


def test_stream_equals_the_whole_decode(codes):
    """CodecStream in ramp chunks + flush is the whole decode; after F frames exactly 2940*F - 1470 samples are out."""
    dec = CodecDecoder.load(GGUF)
    full = dec.decode(codes)
    stream, parts, at = dec.stream(), [], 0
    for n in (1, 2, 4, 8, 15, len(codes)):
        parts.append(stream.decode(codes[at : at + n]))
        at = min(at + n, len(codes))
        out = np.concatenate(parts)
        assert len(out) == 2940 * at - 1470
        np.testing.assert_allclose(out, full[: len(out)], atol=1e-5)
        if at == len(codes):
            break
    out = np.concatenate(parts + [stream.flush()])
    assert len(out) == len(full)
    np.testing.assert_allclose(out, full, atol=1e-5)


def test_stream_leaves_missing_residual_codes_out_of_their_frames(codes, biased):
    """Missing residual codes (-1) drop their codebooks' table rows and biases (the float64 transcription), in any
    chunking; the decode is coarser than the complete one, not the same."""
    dec = CodecDecoder.load(biased)
    assert dec.early_decode
    partial = first_chunk_partial(codes)
    want = reference_decode(Ref(biased), partial)
    stream = dec.stream()
    got = np.concatenate([stream.decode(partial[:4]), stream.decode(partial[4:9]), stream.decode(partial[9:])])
    got = np.concatenate([got, stream.flush()])
    assert got.shape == want.shape
    assert np.abs(got - want).max() <= 1e-4
    assert np.abs(got - dec.decode(codes)).max() > 1e-2


def test_complete_codes_decode_bit_identically_with_residual_biases(codes, biased, unbiased):
    """The residual biases change nothing for complete codes: the stream is bit-identical to the decoder without them,
    and in a chunk with missing codes the latent of every complete frame is the complete decode's, bit for bit."""
    dec, plain = CodecDecoder.load(biased), CodecDecoder.load(unbiased)
    assert not plain.early_decode

    def run(d):
        s = d.stream()
        return np.concatenate([s.decode(codes[:4]), s.decode(codes[4:]), s.flush()])

    assert np.array_equal(run(dec), run(plain))

    partial = first_chunk_partial(codes)
    x = mx.array(np.random.default_rng(2).standard_normal((len(codes), dec.res_biases.shape[1])).astype(np.float32))
    ids = mx.array(np.maximum(partial, 0).astype(np.int32))
    mixed = np.array(dec._residual(x, ids, partial[:, 1:] < 0))
    complete = np.array(dec._residual(x, ids))
    assert np.array_equal(mixed[4:], complete[4:])
    assert not np.allclose(mixed[:4], complete[:4])


def test_stream_refuses_non_codes_and_missing_codes_without_residual_biases(codes, biased, unbiased):
    plain = CodecDecoder.load(unbiased)
    with pytest.raises(ValueError, match="residual codes cannot be missing"):
        plain.stream().decode(first_chunk_partial(codes))
    dec = CodecDecoder.load(biased)
    for row, col, bad in ((0, 0, -1), (2, 3, -2), (1, 5, 4096), (0, 0, 4097)):
        wrong = codes.copy()
        wrong[row, col] = bad
        with pytest.raises(ValueError, match="codec ids"):
            dec.stream().decode(wrong)
