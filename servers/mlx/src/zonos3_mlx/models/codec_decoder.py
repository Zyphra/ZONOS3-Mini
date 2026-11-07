"""ZONOS3 Mini codec decoder (``mossdual_istft``) on MLX: a port of servers/ggml/src/zac.cpp.

Semantic RVQ table + causal ConvNeXt head, residual RVQ tables, five windowed-causal MOSS transformer stages with patch
upsampling (15 -> 240 Hz), the multi-resolution iSTFT head (ConvNeXt trunk, two irfft branches, overlap-add) and the
non-causal FIR mixer at 44160 Hz, then the exact 44160 -> 44100 sinc resampler. The C++ decoder streams in chunks with
conv histories and ring KV caches; over a whole utterance that equals one causal pass, which is what ``decode`` runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from zonos3.tts.stream import StreamDecoder

from zonos3_mlx.gguf_io import GGUFFile, Weight, matmul

ARCH = "mossdual_istft"
_ATTN_BLOCK = 512  # query rows per windowed-attention block


@dataclass(frozen=True)
class StageHP:
    in_dim: int
    out_dim: int
    d_model: int
    n_head: int
    n_layer: int
    n_ff: int
    context: int
    patch: int


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x * (1.0 / math.sqrt(2.0))))


def _layer_norm(x: mx.array, w: mx.array, b: mx.array, eps: float) -> mx.array:
    return mx.fast.layer_norm(x, w, b, eps)


class Params:
    """Weights of a mossdual GGUF (decoder or encoder): matrices as ``Weight``, everything else float32."""

    def __init__(self, f: GGUFFile, host: tuple[str, ...] = ()):
        self.w: dict[str, Weight] = {}
        self.t: dict[str, mx.array] = {}
        self.host: dict[str, np.ndarray] = {}
        for name in f.names():
            if name in host:
                self.host[name] = f.array(name).astype(np.float32)
            elif _is_matrix(name, f.shape(name)):
                self.w[name] = f.weight(name)
            else:
                self.t[name] = f.f32(name)
        mx.eval([w.w for w in self.w.values()] + list(self.t.values()))

    def lin(self, name: str, x: mx.array, bias: str | None = None) -> mx.array:
        y = matmul(x, self.w[name])
        return y if bias is None else y + self.t[bias]


def _is_matrix(name: str, shape: tuple[int, ...]) -> bool:
    return (
        len(shape) == 2
        and name.endswith(".weight")
        and ".dw." not in name
        and "norm" not in name
        and not name.startswith(("sem.", "res."))
        and name != "head.mixer.weight"
    )


def _identity(x: mx.array) -> mx.array:
    return x


def causal_depthwise(x: mx.array, w: mx.array, b: mx.array | None) -> mx.array:
    """x [T, C], w [C, K] (tap 0 oldest), zero history: h[t] = sum_k x[t + k - (K - 1)] * w[:, k]."""
    T, K = x.shape[0], w.shape[1]
    xf = mx.concatenate([mx.zeros((K - 1, x.shape[1]), dtype=x.dtype), x], axis=0)
    h = xf[0:T] * w[:, 0]
    for k in range(1, K):
        h = h + xf[k : k + T] * w[:, k]
    return h if b is None else h + b


def convnext_block(P: Params, x: mx.array, p: str, eps: float, rnd=_identity, round_out: bool = False) -> mx.array:
    """Causal ConvNeXt block. ``rnd`` marks the encoder's BF16 boundaries (zac.cpp ``encoder_round``; identity for the
    decoder), ``round_out`` its rounded output (the semantic encoder's blocks)."""
    h = causal_depthwise(rnd(x), P.t[p + ".dw.weight"], None)
    h = rnd(rnd(h) + P.t[p + ".dw.bias"])
    h = rnd(_layer_norm(h, P.t[p + ".norm.weight"], P.t[p + ".norm.bias"], eps))
    h = rnd(_gelu(rnd(P.lin(p + ".pw1.weight", h, p + ".pw1.bias"))))
    h = rnd(P.lin(p + ".pw2.weight", h, p + ".pw2.bias"))
    y = x + h
    return rnd(y) if round_out else y


def patch_up(x: mx.array, h: int) -> mx.array:
    """out[t*h + j, ch] = in[t, ch*h + j]."""
    T, C = x.shape
    return x.reshape(T, C // h, h).transpose(0, 2, 1).reshape(T * h, C // h)


def windowed_attention(q: mx.array, k: mx.array, v: mx.array, context: int, scale: float) -> mx.array:
    """q, k, v [heads, T, hd] at positions 0..T-1; keys with 0 <= q_pos - k_pos < context; blocked over queries."""
    T = q.shape[1]
    outs = []
    for a in range(0, T, _ATTN_BLOCK):
        e = min(a + _ATTN_BLOCK, T)
        left = max(0, a - context + 1)
        qp = mx.arange(a, e)[:, None]
        kp = mx.arange(left, e)[None, :]
        diff = qp - kp
        mask = (diff >= 0) & (diff < context)
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q[None, :, a:e], k[None, :, left:e], v[None, :, left:e], scale=scale, mask=mask
            )[0]
        )
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


def moss_layer(P: Params, x: mx.array, p: str, hp: StageHP, ln_eps: float, rnd=_identity) -> mx.array:
    """Pre-LN windowed-causal attention (adjacent-pair RoPE over absolute stage positions) + GELU FFN, layer scales.
    ``rnd``: the encoder's BF16 boundaries (identity for the decoder)."""
    T, d = x.shape
    nh = hp.n_head
    hd = d // nh
    h = _layer_norm(x, P.t[p + ".norm1.weight"], P.t[p + ".norm1.bias"], ln_eps)
    qkv = rnd(P.lin(p + ".qkv.weight", rnd(h))).reshape(T, 3, nh, hd).transpose(1, 2, 0, 3)  # [3, nh, T, hd]
    q = rnd(mx.fast.rope(qkv[0], hd, traditional=True, base=10000.0, scale=1.0, offset=0))
    k = rnd(mx.fast.rope(qkv[1], hd, traditional=True, base=10000.0, scale=1.0, offset=0))
    o = rnd(windowed_attention(q, k, qkv[2], hp.context, hd**-0.5).transpose(1, 0, 2).reshape(T, d))
    x = x + rnd(P.lin(p + ".wo.weight", o)) * P.t[p + ".attn_scale"]
    h = _layer_norm(x, P.t[p + ".norm2.weight"], P.t[p + ".norm2.bias"], ln_eps)
    h = rnd(P.lin(p + ".ffn2.weight", rnd(_gelu(rnd(P.lin(p + ".ffn1.weight", rnd(h)))))))
    return x + h * P.t[p + ".ffn_scale"]


def read_stages(f: GGUFFile, arch: str) -> list[StageHP]:
    k = lambda key: [int(v) for v in f.meta(f"{arch}.stage.{key}")]  # noqa: E731
    cols = [k(n) for n in ("in_dim", "out_dim", "d_model", "n_head", "n_layer", "n_ff", "context", "patch")]
    return [StageHP(*vals) for vals in zip(*cols)]


def sinc_resample(x: mx.array, kernel: mx.array, orig: int, new: int, width: int) -> mx.array:
    """torchaudio-style polyphase resampling: out[n*new + i] = sum_k K[i][k] x[n*orig + k - width], zero outside,
    total ceil(new * len / orig) samples (zac.cpp ``resample``)."""
    L = int(x.shape[0])
    total = -(-new * L // orig)
    blocks = -(-total // new)
    KL = 2 * width + orig
    pad_right = blocks * orig + orig - (L + width)
    xp = mx.concatenate([mx.zeros((width,), x.dtype), x, mx.zeros((max(pad_right, 0),), x.dtype)])
    a = xp[: blocks * orig].reshape(blocks, orig)
    b = xp[orig : orig + blocks * orig].reshape(blocks, orig)[:, : KL - orig]
    windows = mx.concatenate([a, b], axis=1)  # [blocks, KL]
    return (windows @ kernel.T).reshape(-1)[:total]


class CodecDecoder:
    def __init__(self, path: str | Path):
        f = GGUFFile(path)
        if f.arch != ARCH:
            raise ValueError(f"{path}: not a ZONOS3 Mini iSTFT decoder GGUF (architecture {f.arch!r})")
        k = lambda key: f.meta(f"{ARCH}.{key}")  # noqa: E731
        self.path = str(path)
        self.name = Path(path).name
        self.sample_rate = int(k("sample_rate"))
        self.hop = int(k("hop"))
        self.internal_hop = int(k("internal_hop"))
        self.n_codebooks = int(k("n_codebooks"))
        self.codebook_size = int(k("codebook_size"))
        self.sem_depth = int(k("sem_head.depth"))
        self.head_depth = int(k("head.depth"))
        self.head_hop = int(k("head.hop"))
        self.n_fft = [int(v) for v in k("head.n_fft")]
        self.max_log_mag = float(k("head.max_log_magnitude"))
        self.headroom_db = float(k("headroom_db"))
        self.ln_eps = float(k("ln_eps"))
        self.convnext_eps = float(k("convnext_eps"))
        self.rs = (int(k("resample.orig")), int(k("resample.new")), int(k("resample.width")))
        self.stages = read_stages(f, ARCH)
        if (self.sample_rate, self.hop, self.internal_hop, self.n_codebooks) != (44100, 2940, 2944, 10):
            raise ValueError(f"{path}: inconsistent decoder dimensions")
        self.P = Params(f, host=("head.window.0", "head.window.1", "head.mixer.weight", "resample.kernel"))
        host = self.P.host
        self.windows = [mx.array(host[f"head.window.{i}"]) for i in range(2)]
        self.mixer = mx.array(host["head.mixer.weight"])  # [2, K]
        self.rs_kernel = mx.array(host["resample.kernel"])  # [new, 2w + orig]

    @classmethod
    def load(cls, path: str | Path) -> "CodecDecoder":
        return cls(path)

    # -------------------------------------------------------------------------------------------------- trunk
    def _trunk(self, codes: mx.array) -> tuple[mx.array, mx.array]:
        P = self.P
        x = P.t["sem.table"][codes[:, 0]] + P.t["sem.bias"]
        for b in range(self.sem_depth):
            x = convnext_block(P, x, f"sem_head.{b}", self.convnext_eps)
        x = P.lin("sem_head.out.weight", x, "sem_head.out.bias")
        for c in range(1, self.n_codebooks):
            x = x + P.t[f"res.{c - 1}.table"][codes[:, c]]
        x = x + P.t["res.bias"]
        mx.eval(x)
        for s, hp in enumerate(self.stages):
            p = f"stage.{s}"
            x = P.lin(p + ".in.weight", x)
            for layer in range(hp.n_layer):
                x = moss_layer(P, x, f"{p}.l{layer}", hp, self.ln_eps)
                mx.eval(x)
            if hp.out_dim > 0:
                x = P.lin(p + ".out.weight", x)
            if hp.patch > 1:
                x = patch_up(x, hp.patch)
        for b in range(self.head_depth):
            x = convnext_block(P, x, f"head.{b}", self.convnext_eps)
            mx.eval(x)
        x = _layer_norm(x, P.t["head.norm.weight"], P.t["head.norm.bias"], self.convnext_eps)
        return P.lin("head.out.0.weight", x, "head.out.0.bias"), P.lin("head.out.1.weight", x, "head.out.1.bias")

    # ---------------------------------------------------------------------------------------------- iSTFT head
    def _istft(self, spec: mx.array, i: int) -> mx.array:
        """torch.istft(center=True, length=F*hop) of one branch: spec [F, n_fft + 2] = logmag | phase."""
        n, hop = self.n_fft[i], self.head_hop
        bins = n // 2 + 1
        F = spec.shape[0]
        mag = mx.exp(mx.minimum(spec[:, :bins], self.max_log_mag))
        phase = spec[:, bins : 2 * bins]
        real = mag * mx.cos(phase)
        imag = mag * mx.sin(phase)
        edge = mx.array(np.r_[0.0, np.ones(bins - 2), 0.0].astype(np.float32))
        frames = mx.fft.irfft(real + 1j * (imag * edge), n=n, axis=-1).astype(mx.float32) * self.windows[i]
        r = n // hop
        chunks = frames.reshape(F, r, hop)
        ola = mx.zeros((F + r - 1, hop), dtype=mx.float32)
        for j in range(r):
            ola = ola + mx.pad(chunks[:, j], [(j, r - 1 - j), (0, 0)])
        win2 = np.square(np.array(self.windows[i], dtype=np.float32)).reshape(r, hop)
        env = np.zeros((F + r - 1, hop), dtype=np.float32)
        for j in range(r):
            env[j : j + F] += win2[j]
        pad, length = n // 2, F * hop
        return (ola.reshape(-1) / mx.array(env.reshape(-1)))[pad : pad + length]

    def _mix(self, s0: mx.array, s1: mx.array) -> mx.array:
        K = self.mixer.shape[1]
        x = mx.stack([s0, s1], axis=-1)[None]  # [1, L, 2]
        w = self.mixer.T[None]  # [1, K, 2]
        return mx.conv1d(x, w, padding=(K - 1) // 2)[0, :, 0]

    # ------------------------------------------------------------------------------------------------- public
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Ordinary codes [frames, n_codebooks] -> float32 PCM [frames * hop] at 44100 Hz (pre-gain)."""
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self.n_codebooks:
            raise ValueError(f"codes must be [frames, {self.n_codebooks}], got {codes.shape}")
        frames = int(codes.shape[0])
        if frames == 0:
            return np.zeros(0, dtype=np.float32)
        ids = mx.array(np.clip(codes, 0, self.codebook_size - 1).astype(np.int32))
        y0, y1 = self._trunk(ids)
        mixed = self._mix(self._istft(y0, 0), self._istft(y1, 1))
        audio = sinc_resample(mixed, self.rs_kernel, *self.rs)
        mx.eval(audio)
        out = np.array(audio, dtype=np.float32)
        # every prefix length is a new set of buffer sizes: without this the MLX buffer cache grows past 9 GiB over one
        # long stream, and the memory pressure slows the next decodes and stalls the backbone's next prefill for seconds
        mx.clear_cache()
        want = frames * self.hop
        if len(out) >= want:
            return out[:want]
        return np.concatenate([out, np.zeros(want - len(out), dtype=np.float32)])

    def output_gain(self, headroom_db: float | None) -> float:
        db = self.headroom_db if headroom_db is None else float(headroom_db)
        return float(10.0 ** (-db / 20.0))

    def decode_rows(
        self,
        rows,
        *,
        prefill: np.ndarray | None,
        eos_frame: int | None,
        context_codes=None,
        audio_pad_id: int = 4097,
        overlap: int = 4,
    ) -> np.ndarray:
        """End-of-stream decode of generated rows (zonos3.reference.synth.decode_rows): the prefill rows open the
        stream, EOS is the earliest of the aligned EOS frame and every row's EOA column, the last ``overlap`` prefill
        rows are re-decoded with the rest, and the reference tail decodes as left context at the stream start."""

        def decode(codes: np.ndarray) -> np.ndarray:
            if ((codes < 0) | (codes >= self.codebook_size)).any():
                raise ValueError("decode rows hold non-codec tokens")
            return self.decode(codes)

        decoder = StreamDecoder(
            decode,
            n_codebooks=self.n_codebooks,
            audio_pad_id=audio_pad_id,
            hop=self.hop,
            prefill_rows=prefill,
            context_codes=context_codes,
            mode="whole",
            overlap_frames=overlap,
        )
        for row in rows:
            decoder.push(row, eos_frame)
        return decoder.finish()
