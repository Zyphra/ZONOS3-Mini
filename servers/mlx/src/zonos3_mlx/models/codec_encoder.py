"""Frozen dual reference encoder (``mossdual_encoder``) on MLX: 44.1 kHz audio -> ordinary [frames, 10] codes.

A port of servers/ggml/src/zac-encoder.cpp, including its production autocast recipe: GEMM / convolution weights and
biases hold BF16 values, operands are rounded to BF16 and accumulate in F32, and the outputs are rounded to BF16 at the
same tensor boundaries as the C++ (``encoder_round``); residual sums, Snake and the RVQ stay F32. Lanes of 300 frames
with 150 frames of left context; per lane the acoustic branch (Snake residual units, strided causal convs, patch-down,
MOSS stages), the BF16 SSL teacher pooled onto the lane's frames, the semantic encoder and RVQ 0, the semantic
prediction, and RVQ 1..9 on the residual.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from zonos3.generation import REFERENCE_AUDIO_MAX_SECONDS, SAMPLE_RATE, UPLOAD_MAX_SECONDS

from zonos3_mlx.audio_io import polyphase_resample
from zonos3_mlx.gguf_io import GGUFFile, Weight
from zonos3_mlx.models.codec_decoder import (
    Params,
    bf16_round,
    convnext_block,
    moss_layer,
    read_stages,
)
from zonos3_mlx.models.ssl_teacher import SSLTeacher

ARCH = "mossdual_encoder"
LANE, CONTEXT, HOP, INTERNAL_HOP = 300, 150, 2940, 2944


def _snake(x: mx.array, alpha: mx.array, inv_alpha: mx.array) -> mx.array:
    s = mx.sin(alpha * x)
    return x + inv_alpha * s * s


class _Quantizer:
    def __init__(self, t: dict[str, np.ndarray], p: str):
        self.in_w = mx.array(t[p + ".in_proj.weight"])  # [8, C]
        self.in_b = mx.array(t[p + ".in_proj.bias"])
        self.out_w = mx.array(t[p + ".out_proj.weight"])  # [C, 8]
        self.out_b = mx.array(t[p + ".out_proj.bias"])
        cb = t[p + ".codebook"].astype(np.float32)  # [4096, 8]
        norm = np.maximum(np.sqrt((cb * cb).sum(-1, keepdims=True)), 1e-12)
        normalized = cb / norm
        self.codebook = mx.array(cb)
        self.normalized = mx.array(normalized)
        self.norm_squared = mx.array((normalized * normalized).sum(-1))

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """x [T, C] -> (quantized [T, C], ids [T]); first index on ties (torch.max)."""
        z = x @ self.in_w.T + self.in_b
        n = z / mx.maximum(mx.sqrt((z * z).sum(-1, keepdims=True)), 1e-12)
        dist = ((n * n).sum(-1, keepdims=True) - 2.0 * (n @ self.normalized.T)) + self.norm_squared
        ids = mx.argmin(dist, axis=-1)
        lookup = z + (self.codebook[ids] - z)
        return lookup @ self.out_w.T + self.out_b, ids


class CodecEncoder:
    def __init__(self, enc_path: str | Path, ssl: SSLTeacher):
        if ssl.dtype != "bf16":
            raise ValueError("the reference encoder runs the BF16 SSL teacher recipe")
        f = GGUFFile(enc_path)
        if f.arch != ARCH:
            raise ValueError(f"{enc_path}: invalid mossdual_encoder GGUF (architecture {f.arch!r})")
        k = lambda key: f.meta(f"{ARCH}.{key}")  # noqa: E731
        if (int(k("lane_frames")), int(k("context_frames")), int(k("hop")), int(k("internal_hop"))) != (
                LANE, CONTEXT, HOP, INTERNAL_HOP):
            raise ValueError(f"{enc_path}: unexpected encoder recipe dimensions")
        self.path = str(enc_path)
        self.ssl = ssl
        self.sem_depth = int(k("semantic_depth"))
        self.window_ratio = float(k("window_ratio"))
        self.ln_eps, self.convnext_eps = float(k("ln_eps")), float(k("convnext_eps"))
        self.rs = (int(k("resample.orig")), int(k("resample.new")), int(k("resample.width")))
        self.stages = read_stages(f, ARCH)
        rvq = tuple(n for n in f.names() if n.startswith("rvq."))
        conv_weights = tuple(n for n in f.names() if n.startswith("enc.") and n.endswith(".weight"))
        self.P = Params(f, host=("resample.kernel",) + rvq + conv_weights)
        self.rs_kernel = self.P.host["resample.kernel"]
        self.rvq = [_Quantizer(self.P.host, f"rvq.{c}") for c in range(10)]
        # Linear matrices are BF16 operands (zac_encoder_load); stored as bfloat16, matmul upcasts them exactly.
        for name, W in self.P.w.items():
            if W.scales is None:
                self.P.w[name] = Weight(W.w.astype(mx.bfloat16))
        # Every other autocast parameter (weights and biases outside the norms) holds BF16-rounded values.
        for name in list(self.P.t):
            if "norm" not in name and (".weight" in name or ".bias" in name):
                self.P.t[name] = bf16_round(self.P.t[name])
        # ConvNeXt depthwise taps are stored numpy [K, C]; the shared block takes [C, K] (the decoder's layout).
        for name in [n for n in self.P.t if n.endswith(".dw.weight")]:
            self.P.t[name] = self.P.t[name].T
        # Acoustic convs are stored tap-major numpy [K, OC, IC] -> MLX conv1d [OC, K, IC], BF16 values.
        self.conv = {n.removesuffix(".weight"): bf16_round(mx.array(self.P.host.pop(n).transpose(1, 0, 2)))
                     for n in conv_weights}
        mx.eval(list(self.conv.values()) + list(self.P.t.values()) + [W.w for W in self.P.w.values()])

    @classmethod
    def load(cls, enc_path: str | Path, ssl: SSLTeacher) -> "CodecEncoder":
        return cls(enc_path, ssl)

    # --------------------------------------------------------------------------------------------- acoustic
    def _causal_conv(self, x: mx.array, p: str, dilation: int = 1, stride: int = 1) -> mx.array:
        """x [T, IC] -> [T', OC]; left pad (K-1)*d + 1 - stride; BF16 input, BF16 output before and after the bias
        (zac.cpp ``encoder_conv``)."""
        w = self.conv[p]
        K = w.shape[1]
        left = (K - 1) * dilation + 1 - stride
        x = bf16_round(x)
        xp = mx.concatenate([mx.zeros((left, x.shape[1])), x], axis=0) if left > 0 else x
        y = mx.conv1d(xp[None], w, stride=stride, dilation=dilation)[0]
        return bf16_round(bf16_round(y) + self.P.t[p + ".bias"].reshape(-1))

    def _snake(self, x: mx.array, p: str) -> mx.array:
        return _snake(x, self.P.t[p + ".alpha"].reshape(-1), self.P.t[p + ".inv_alpha"].reshape(-1))

    def _linear(self, name: str, x: mx.array) -> mx.array:
        return bf16_round(self.P.lin(name, bf16_round(x)))

    def _acoustic(self, audio: np.ndarray) -> mx.array:
        x = self._causal_conv(mx.array(audio.reshape(-1, 1)), "enc.in")
        for b in range(2):
            p = f"enc.b{b}"
            for r, dil in enumerate((1, 3, 9)):
                q = f"{p}.res{r}"
                y = self._causal_conv(self._snake(x, q + ".snake1"), q + ".conv1", dil)
                # both residual operands are BF16 convolution outputs: the sum is BF16 before the next Snake
                x = bf16_round(x + self._causal_conv(self._snake(y, q + ".snake2"), q + ".conv2"))
            x = self._causal_conv(self._snake(x, p + ".snake"), p + ".conv", 1, 2 if b == 0 else 4)
            mx.eval(x)
        for s, hp in enumerate(self.stages):
            p = f"stage.{s}"
            T, C = x.shape
            h = hp.patch
            x = x.reshape(T // h, h, C).transpose(0, 2, 1).reshape(T // h, C * h)  # out[t, ch*h + j] = in[t*h + j, ch]
            x = self._linear(p + ".in.weight", x)
            for layer in range(hp.n_layer):
                x = moss_layer(self.P, x, f"{p}.l{layer}", hp, self.ln_eps, rnd=bf16_round)
                mx.eval(x)
            x = self._linear(p + ".out.weight", x)
        return x

    # --------------------------------------------------------------------------------------------- semantic
    def _semantic(self, x: mx.array, encode: bool) -> mx.array:
        """Semantic encoder (pooled SSL -> 768) or semantic head (q1 -> 1024 prediction): zac-encoder.cpp
        ``semantic_module``; the WNConv / output projections round before and after their bias."""
        def projection(name: str, h: mx.array) -> mx.array:
            return bf16_round(bf16_round(self.P.lin(name + ".weight", bf16_round(h))) + self.P.t[name + ".bias"])

        if encode:
            x = projection("sem_enc.in", x)
        prefix = "sem_enc." if encode else "sem_head."
        for b in range(self.sem_depth):
            x = convnext_block(self.P, x, f"{prefix}{b}", self.convnext_eps, rnd=bf16_round, round_out=encode)
        if not encode:
            x = projection("sem_head.out", x)
        return x

    def _pool_ssl(self, ssl: np.ndarray) -> np.ndarray:
        """Box-window average of the T SSL frames onto the lane's 300 frames (zac-encoder.cpp pool_ssl)."""
        T = ssl.shape[0]
        prefix = np.zeros((T + 1, ssl.shape[1]), dtype=np.float32)
        np.cumsum(ssl, axis=0, dtype=np.float32, out=prefix[1:])
        ratio = np.float32(T / LANE)
        half = np.float32(0.5 * self.window_ratio) * ratio
        centers = (np.arange(LANE, dtype=np.float32) + np.float32(0.5)) * ratio
        left = np.clip(centers - half, 0, T).astype(np.float32)
        right = np.clip(centers + half, 0, T).astype(np.float32)
        li, ri = np.floor(left).astype(np.int64), np.floor(right).astype(np.int64)
        lf, rf = (left - li)[:, None], (right - ri)[:, None]
        a = prefix[li] + ssl[np.minimum(li, T - 1)] * lf
        b = prefix[ri] + ssl[np.minimum(ri, T - 1)] * rf
        return ((b - a) / (right - left)[:, None]).astype(np.float32)

    # ------------------------------------------------------------------------------------------------ public
    def encode(self, pcm44100: np.ndarray, *, max_seconds: float = REFERENCE_AUDIO_MAX_SECONDS) -> np.ndarray:
        """Mono 44.1 kHz PCM -> ordinary codes [frames, 10] int32 (frames = ceil(samples / 2940)), cut to whole
        codec frames at ``max_seconds``: at most the registry's UPLOAD_MAX_SECONDS, the limit of
        zonos3.tokenizer.dual_reference.prepare_reference and zac-encoder.h ZAC_ENCODER_MAX_SECONDS (the lanes keep
        memory independent of length; callers pass their own, shorter limit)."""
        if not 0 < max_seconds <= UPLOAD_MAX_SECONDS:
            raise ValueError(f"Codec input limit must be in (0, {UPLOAD_MAX_SECONDS}] seconds")
        audio = np.asarray(pcm44100, dtype=np.float32)
        limit = int(math.floor(max_seconds * SAMPLE_RATE / HOP)) * HOP
        audio = audio[:limit]
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("reference audio must be finite and nonempty")
        n = len(audio)
        frames = -(-n // HOP)
        internal = polyphase_resample(audio, self.rs_kernel, *self.rs)
        internal = np.concatenate([internal, np.zeros(max(0, frames * INTERNAL_HOP - len(internal)), np.float32)])
        internal = internal[: frames * INTERNAL_HOP]
        codes = np.zeros((frames, 10), dtype=np.int32)
        start = 0
        while start < frames:
            context = 0 if start == 0 else CONTEXT
            count = min(frames - start, LANE - context)
            left = start - context
            lane = np.zeros(LANE * INTERNAL_HOP, dtype=np.float32)
            lane[: (context + count) * INTERNAL_HOP] = internal[left * INTERNAL_HOP : (left + context + count) * INTERNAL_HOP]
            teacher = np.zeros(LANE * HOP, dtype=np.float32)
            t0 = left * HOP
            tn = min(n - t0, (context + count) * HOP)
            teacher[:tn] = audio[t0 : t0 + tn]
            ssl = self.ssl.encode(teacher)
            acoustic = self._acoustic(lane)
            semantic = self._semantic(mx.array(self._pool_ssl(ssl)), True)
            q1, ids = self.rvq[0](semantic)
            codes[start : start + count, 0] = np.array(ids)[context : context + count]
            residual = acoustic - self._semantic(q1, False)
            for c in range(1, 10):
                qr, ids = self.rvq[c](residual)
                codes[start : start + count, c] = np.array(ids)[context : context + count]
                residual = residual - qr
            mx.eval(residual)
            start += count
        return codes
