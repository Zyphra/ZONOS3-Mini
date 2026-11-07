"""Stateful incremental decode of an `ISTFTCodec` with the multi-resolution iSTFT head (the torch twin of zac_stream).

`ISTFTStream.decode(codes)` takes the next undelayed code frames and returns the 44100 Hz samples that are final;
`flush()` returns the rest. The causal trunk keeps its state between calls: one depthwise-conv history per ConvNeXt
block (semantic head and iSTFT head) and a sliding-window KV cache per transformer layer. The host half (overlap-add
iSTFT, FIR mixer, resampler) is `zonos3.codec.stream_host`. Concatenated outputs + flush equal `model(codes)` over the
whole sequence: after F frames exactly 2940*F - 1470 samples are out, flush brings the total to 2940*F.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

from zonos3.codec.hybrid10_dual_encoding.base_model import MossStage, Patch
from zonos3.codec.hybrid10_dual_encoding.model import SemanticConvNeXt
from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec, MultiResISTFTHead
from zonos3.codec.stream_host import IstftHeadStream, ResamplerStream

_CODEBOOK_SIZE = 4096


def _convnext(block, x, hist):
    """SemanticConvNeXt.forward with the left zero padding replaced by the previous call's last K-1 inputs."""
    k = block.dwconv.kernel_size[0]
    if hist is None:
        hist = x.new_zeros(x.shape[0], x.shape[1], k - 1)
    xf = torch.cat((hist.to(x.dtype), x), -1)
    h = block.dwconv(xf).transpose(1, 2)
    h = block.pwconv2(F.gelu(block.pwconv1(block.norm(h))))
    return x + h.transpose(1, 2), xf[..., -(k - 1) :]


def _moss_layer(layer, x, pos0, cache):
    """MossLayer.forward on positions [pos0, pos0+t) attending to the cached keys/values of the previous positions."""
    b, t, d = x.shape
    q, k, v = layer.qkv(layer.norm1(x)).reshape(b, t, 3, layer.heads, layer.head_dim).unbind(2)
    frequency = torch.exp(
        torch.arange(layer.head_dim // 2, device=x.device, dtype=torch.float32)
        * (-math.log(10000.0) * 2 / layer.head_dim)
    )
    phase = (pos0 + torch.arange(t, device=x.device)).to(torch.float32)[:, None] * frequency[None, :]
    cos, sin = phase.cos()[None, :, None, :], phase.sin()[None, :, None, :]

    def rope(z):
        zf = z.float().reshape(b, t, layer.heads, layer.head_dim // 2, 2)
        r, i = zf.unbind(-1)
        return torch.stack((r * cos - i * sin, r * sin + i * cos), -1).flatten(-2).to(z.dtype)

    q, k = rope(q), rope(k)
    q, k, v = (z.transpose(1, 2) for z in (q, k, v))
    if cache is not None:
        k = torch.cat((cache[0].to(k.dtype), k), 2)
        v = torch.cat((cache[1].to(v.dtype), v), 2)
    width = k.shape[2]
    lc = width - t
    qp = torch.arange(pos0, pos0 + t, device=x.device)
    kp = torch.arange(pos0 - lc, pos0 + t, device=x.device)
    delta = qp[:, None] - kp[None, :]
    valid = (delta >= 0) & (delta < layer.context)
    bias = x.new_full((t, (width + 7) // 8 * 8), float("-inf"))
    bias[:, :width].masked_fill_(valid, 0)
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[:, :width])
    keep = layer.context - 1
    cache = (k[:, :, -keep:], v[:, :, -keep:]) if keep > 0 else None
    x = x + layer.attention_scale * layer.attention_out(attn.transpose(1, 2).reshape(b, t, d))
    return x + layer.ffn_scale * layer.ffn(layer.norm2(x)), cache


class ISTFTStream:
    """Incremental decoder over one utterance; feed undelayed code frames [n, codebooks] in order."""

    def __init__(self, model: ISTFTCodec, max_chunk: int = 32):
        if not isinstance(model.head, MultiResISTFTHead):
            raise ValueError("ISTFTStream needs the multi-resolution iSTFT head")
        self.model = model
        self.max_chunk = int(max_chunk)
        self.n_codebooks = 1 + model.cfg["residual_codebooks"]
        self.frames = 0
        blocks = list(model.semantic_head)
        if not all(isinstance(b, SemanticConvNeXt) for b in blocks[:-1]):
            raise ValueError("ISTFTStream: unexpected semantic head layout")
        self.sem_blocks, self.sem_out = blocks[:-1], blocks[-1]
        self.sem_hist = [None] * len(self.sem_blocks)
        head = model.head
        self.head_hist = [None] * len(head.blocks)
        self.kv = [[None] * len(s.layers) if isinstance(s, MossStage) else None for s in model.stages]
        windows = [getattr(head, f"window_{i}").detach().cpu().numpy() for i in range(len(head.n_ffts))]
        self.head = IstftHeadStream(
            head.n_ffts, head.hop, windows, head.mixer.weight[0].detach().cpu().numpy(), head.max_log_magnitude
        )
        rs = model.to_external
        self.resampler = ResamplerStream(
            rs.kernel[:, 0].detach().cpu().numpy(), rs.orig_freq // rs.gcd, rs.new_freq // rs.gcd, rs.width
        )
        self.device = next(model.parameters()).device

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != self.n_codebooks or ((codes < 0) | (codes >= _CODEBOOK_SIZE)).any():
            raise ValueError("Invalid matched codec tokens")
        if len(codes) == 0:
            return np.zeros(0, np.float32)
        prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with torch.inference_mode():
                out = [self._chunk(codes[i : i + self.max_chunk]) for i in range(0, len(codes), self.max_chunk)]
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
        return np.concatenate(out)

    def flush(self) -> np.ndarray:
        return np.concatenate([self.resampler.push(self.head.flush()), self.resampler.flush()])

    def _chunk(self, codes):
        m, dev = self.model, self.device.type
        n = len(codes)
        packed = torch.as_tensor(np.ascontiguousarray(codes.T[None]), dtype=torch.int64, device=self.device)
        q = m.quantizer
        with torch.autocast(dev, enabled=False):
            x = q.semantic_quantizer.from_codes(packed[:, :1])[0]
            for i, block in enumerate(self.sem_blocks):
                x, self.sem_hist[i] = _convnext(block, x, self.sem_hist[i])
            z = self.sem_out(x).float()
            for i, vq in enumerate(q.quantizer.quantizers):
                z = z + vq.out_proj(vq.decode_code(packed[:, 1 + i]))
        x = z
        cpf = 1
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            for si, stage in enumerate(m.stages):
                if isinstance(stage, Patch):
                    x = stage(x)
                    cpf *= stage.size
                elif isinstance(stage, MossStage):
                    x = stage.input_proj(x.transpose(1, 2))
                    for li, layer in enumerate(stage.layers):
                        x, self.kv[si][li] = _moss_layer(layer, x, self.frames * cpf, self.kv[si][li])
                    x = stage.output_proj(x).transpose(1, 2)
                else:
                    raise ValueError(f"ISTFTStream: unsupported stage {type(stage).__name__}")
        head = m.head
        with torch.autocast(dev, enabled=False):
            h = x
            for i, block in enumerate(head.blocks):
                h, self.head_hist[i] = _convnext(block, h, self.head_hist[i])
            h = head.norm(h.transpose(1, 2))
            ys = [lin(h)[0].float().cpu().numpy() for lin in head.outs]
        out = self.resampler.push(self.head.push(*ys))
        self.frames += n
        return out
