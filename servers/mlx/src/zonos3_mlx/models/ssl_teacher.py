"""Sixteen-layer Wav2Vec2-BERT semantic teacher (``w2vbert_ssl``) and its SeamlessM4T frontend, on MLX.

A port of servers/ggml/src/ssl-encoder.cpp (``ssl_features`` and ``build_graph``). The frontend runs on the host
exactly as the C++ does (F64 FFT materialized as complex64, NumPy's pairwise statistics). The conformer layers run in
MLX in one of the two C++ recipes: ``bf16`` (the production autocast recipe the reference encoder uses: BF16 GEMM
operands with F32 accumulation and BF16 tensor boundaries) or ``f32`` (``tools/dump_golden.py ssl-teacher --dtype
f32``, the diagnostic oracle).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from zonos3_mlx.audio_io import polyphase_resample
from zonos3_mlx.gguf_io import GGUFFile
from zonos3_mlx.numeric import bf16_round

ARCH = "w2vbert_ssl"


def _silu(x):
    return x * mx.sigmoid(x)


def _identity(x):
    return x


_BF16_MAX = 3.3895313892515355e38  # bfloat16 0x7f7f: the masked-score fill of the BF16 recipe


class SSLTeacher:
    def __init__(self, path: str | Path, dtype: str = "bf16"):
        if dtype not in ("bf16", "f32"):
            raise ValueError(f"SSL teacher dtype must be bf16 or f32, got {dtype!r}")
        f = GGUFFile(path)
        if f.arch != ARCH:
            raise ValueError(f"{path}: expected w2vbert_ssl (tools/convert_gguf.py ssl-teacher)")
        k = lambda key: f.meta(f"{ARCH}.{key}")  # noqa: E731
        self.path = str(path)
        self.n_layer, self.n_embd, self.n_head = int(k("n_layer")), int(k("n_embd")), int(k("n_head"))
        self.rel_left, self.rel_right = int(k("rel_left")), int(k("rel_right"))
        self.conv_kernel, self.n_mels = int(k("conv_kernel")), int(k("n_mels"))
        self.frame_length, self.hop_length, self.fft_length = (
            int(k("frame_length")),
            int(k("hop_length")),
            int(k("fft_length")),
        )
        self.stride = int(k("stride"))
        self.eps = float(k("eps"))
        self.preemph = float(k("preemphasis"))
        self.mel_floor = float(k("mel_floor"))
        self.padding_value = float(k("padding_value"))
        self.max_seconds = int(k("max_seconds"))
        self.window = f.array("frontend.window").astype(np.float64)  # [400]
        self.mel_filters = f.array("frontend.mel_filters").astype(np.float64)  # [257, 80]
        self.resample_kernel = f.array("frontend.resample_44100").astype(np.float32)  # [160, 475]
        self.dtype = dtype
        self._r = bf16_round if dtype == "bf16" else _identity
        self._mask_max = _BF16_MAX if dtype == "bf16" else float(np.finfo(np.float32).max)
        self.t = {}
        for name in f.names():
            if name.startswith("frontend."):
                continue
            value = mx.array(f.array(name).astype(np.float32))
            # GEMM weights are BF16 operands in the bf16 recipe; norms, biases and the depthwise taps stay F32.
            gemm = (
                name.endswith(".weight")
                and "_ln" not in name
                and not name.startswith("fp.ln")
                and not name.endswith("conv.dw.weight")
            )
            self.t[name] = self._r(value) if (gemm or name.endswith("attn.dist")) else value
        mx.eval(list(self.t.values()))

    @classmethod
    def load(cls, path: str | Path, dtype: str = "bf16") -> "SSLTeacher":
        return cls(path, dtype)

    # --------------------------------------------------------------------------------------------- frontend
    def features(self, pcm44100: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mono 44.1 kHz PCM -> (features [T, 160] float32, attention mask [T] int32)."""
        pcm = np.asarray(pcm44100, dtype=np.float32)
        if not len(pcm) or len(pcm) > 44100 * self.max_seconds:
            raise ValueError(f"SSL teacher audio must be nonempty and at most {self.max_seconds} s")
        return self.features_16k(polyphase_resample(pcm, self.resample_kernel, 441, 160, 17))

    def features_16k(self, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fl, hop, nfft, nm = self.frame_length, self.hop_length, self.fft_length, self.n_mels
        n = len(wav)
        nf = 0 if n < fl else 1 + (n - fl) // hop
        if nf < 2:
            raise ValueError("SSL teacher audio is too short for a normalized fbank")
        frames = np.asarray(wav, dtype=np.float32)[np.arange(nf)[:, None] * hop + np.arange(fl)[None, :]]
        z = frames.astype(np.float64) * 32768.0
        z = z - z.mean(axis=1, keepdims=True)
        z[:, 1:] = z[:, 1:] - 0.97 * z[:, :-1]  # the C++ runs this in reverse order, so on the original values
        z[:, 0] *= 1.0 - 0.97
        z = z * self.window
        spec = np.fft.fft(np.pad(z, ((0, 0), (0, nfft - fl))), axis=1)[:, : nfft // 2 + 1]
        re, im = spec.real.astype(np.float32).astype(np.float64), spec.imag.astype(np.float32).astype(np.float64)
        power = re * re + im * im
        fb = np.log(np.maximum(power @ self.mel_filters, self.mel_floor)).astype(np.float32)  # [nf, 80]
        # Per-mel statistics in float32 with NumPy's pairwise sums over each contiguous mel column (the C++
        # frontend_pairwise_sum transcribes them); the normalization runs in float32 too.
        cols = np.ascontiguousarray(fb.T)
        mean = cols.sum(axis=1, dtype=np.float32) / np.float32(nf)
        dev = cols - mean[:, None]
        var = np.sqrt((dev * dev).sum(axis=1, dtype=np.float32) / np.float32(nf - 1) + np.float32(1e-7))
        normalized = (fb - mean) / var
        T = -(-nf // self.stride)
        feats = np.full((T * self.stride, nm), self.padding_value, dtype=np.float32)
        feats[:nf] = normalized
        mask = np.ones(T, dtype=np.int32)
        if nf % self.stride:
            mask[-1] = 0
        return feats.reshape(T, self.stride * nm), mask

    # -------------------------------------------------------------------------------------------- network
    def _ln(self, x, p):
        return mx.fast.layer_norm(x, self.t[p + ".weight"], self.t[p + ".bias"], self.eps)

    def _lin(self, x, p, bias=True):
        """Linear at an autocast boundary: BF16 operands (inputs rounded here, weights at load), F32 accumulation,
        the output (after the bias) rounded; plain F32 in the f32 recipe."""
        y = self._r(x) @ self.t[p + ".weight"].T
        return self._r(y + self.t[p + ".bias"] if bias else y)

    def _ffn(self, x, p):
        return self._lin(self._r(_silu(self._lin(self._ln(x, p + "_ln"), p + ".up"))), p + ".down")

    def forward(self, feats: np.ndarray, mask: np.ndarray | None = None, dump: dict | None = None) -> np.ndarray:
        """features [T, 160] (+ mask [T]) -> the layer-16 hidden states [T, 1024] (ssl-encoder.cpp build_graph)."""
        T = int(feats.shape[0])
        D, H = self.n_embd, self.n_head
        hd = D // H
        r = self._r
        valid_np = np.ones(T, dtype=np.float32) if mask is None else (np.asarray(mask) != 0).astype(np.float32)
        if not valid_np.any():
            raise ValueError("SSL teacher mask has no valid frame")
        valid = mx.array(valid_np)[:, None]
        key_mask = mx.array(np.where(valid_np > 0, 0.0, -self._mask_max).astype(np.float32))[None, None, :]
        rel = np.clip(np.arange(T)[None, :] - np.arange(T)[:, None], -self.rel_left, self.rel_right) + self.rel_left
        rel_idx = mx.broadcast_to(mx.array(rel.astype(np.int32))[None], (H, T, T))
        x0 = r(mx.array(np.ascontiguousarray(feats, dtype=np.float32)))
        norm = self._ln(x0, "fp.ln")
        proj = self._lin(norm, "fp.proj")
        x = proj * valid
        if dump is not None:
            dump["feature_input"], dump["feature_norm"] = np.array(x0), np.array(norm)
            dump["feature_projection"], dump["hidden_0"] = np.array(proj), np.array(x)
        scale = 1.0 / np.sqrt(hd)
        for layer in range(self.n_layer):
            p = f"enc.{layer}."
            res = r if layer == 0 else _identity  # only the first layer's residual stream is BF16
            x = res(x + r(0.5 * self._ffn(x, p + "ffn1")))
            an = self._ln(x, p + "attn_ln")
            q, k, v = (self._lin(an, p + f"attn.{n}").reshape(T, H, hd).transpose(1, 0, 2) for n in "qkv")
            score = r(r(q @ k.transpose(0, 2, 1)) * scale)  # [H, T, T]
            rel_scores = r(r(q @ self.t[p + "attn.dist"].T) * scale)  # [H, T, 73]
            score = r(score + mx.take_along_axis(rel_scores, rel_idx, axis=-1))
            probs = mx.softmax(r(score + key_mask), axis=-1)
            attn = r(r(probs) @ v).transpose(1, 0, 2).reshape(T, D)
            x = res(x + self._lin(attn, p + "attn.o"))
            conv = self._lin(self._ln(x, p + "conv_ln") * valid, p + "conv.pw1", bias=False)
            conv = r(conv[:, :D] * mx.sigmoid(conv[:, D:]))
            K = self.conv_kernel
            padded = mx.concatenate([mx.zeros((K - 1, D)), conv], axis=0)
            conv = r(mx.conv1d(padded[None], self.t[p + "conv.dw.weight"][:, :, None], groups=D)[0])
            conv = _silu(self._ln(conv, p + "conv.dw_ln"))
            x = res(x + self._lin(conv, p + "conv.pw2", bias=False))
            x = res(x + r(0.5 * self._ffn(x, p + "ffn2")))
            x = self._ln(x, p + "final_ln")
            mx.eval(x)
            if dump is not None:
                dump[f"hidden_{layer + 1}"] = np.array(x)
        out = np.array(x, dtype=np.float32)
        if dump is not None:
            dump["output"] = out
        return out

    def encode(self, pcm44100: np.ndarray) -> np.ndarray:
        feats, mask = self.features(pcm44100)
        return self.forward(feats, mask)
