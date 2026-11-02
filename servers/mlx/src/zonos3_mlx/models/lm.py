"""ZONOS3 Mini LM on MLX: the GDN / global-attention backbone with Full AttnRes and the latent MoE.

A port of ``zonos3.reference.lm.Zonos3LM.forward`` reading the ggml server's GGUF (tensor names of
servers/ggml/src/zonos3.cpp); the math follows servers/ggml/src/zonos3-graph.cpp ``build_graph``. Activations are float32,
dense weights float16 (upcast per matmul), the mixers and experts of the QAT GGUF stay Q8_0 / Q4_0 (MLX affine). One
sequence per call: the scheduler makes one call per slot per tick.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import mlx.core as mx
import numpy as np
from zonos3.reference.lm import ProbeSpec
from zonos3.tts.probe import clip_window, probe_weight, readout

from zonos3_mlx.gguf_io import GROUP_SIZE, GGUFFile, Weight, matmul
from zonos3_mlx.models.gdn_kernel import gated_delta
from zonos3_mlx.paths import CACHE_DIR

ARCH = "zonos3"
_GDN_PATTERN = re.compile(r"gdn(\d+)_global1")
_KV_BLOCK = 256


def _key(name: str) -> str:
    return f"{ARCH}.{name}"


@dataclass(frozen=True)
class LMConfig:
    n_layers: int
    max_seqlen: int
    dim: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden: int
    n_experts: int
    moe_topk: int
    n_codebooks: int
    codebook_size: int
    audio_vocab: int
    eoa_id: int
    audio_pad_id: int
    span_end_id: int | None
    text_vocab: int
    text_extra_tokens: int
    gdn_heads: int
    gdn_head_dim: int
    gdn_kernel: int
    speaker_embedding_dim: int
    condition_count: int
    condition_n_buckets: int
    condition_names: tuple[str, ...]
    condition_units: dict[str, str]
    condition_edges: dict[str, tuple[float, ...]]
    rope_theta: float
    norm_eps: float
    qk_eps: float
    loss_softcap: float
    gdn_period: int
    silence_frame: tuple[int, ...]
    text_tokenizer_spec: dict
    tag_cfg_trained: bool = False
    # Not in the GGUF: the release architecture tools/convert_gguf.py validate_config accepts.
    audio_codec: str = "opus_dual_3n_600k_sched600k_r4096_step600000"
    speaker_embedding_codec: str = "ownmect_s2"
    speaker_reference_prefix_enabled: bool = True
    speaker_reference_max_seconds: float = 60.0

    @classmethod
    def from_gguf(cls, f: GGUFFile) -> "LMConfig":
        if f.arch != ARCH:
            raise ValueError(f"{f.path}: general.architecture is {f.arch!r}, not {ARCH!r}")
        for key, wanted in (("residual.type", "full_attnres"), ("moe.balancing", "quantile")):
            if f.meta(_key(key)) != wanted:
                raise ValueError(f"{f.path}: {_key(key)} = {f.meta(_key(key))!r}, expected {wanted!r}")
        pattern = _GDN_PATTERN.fullmatch(f.meta(_key("attention.pattern")))
        if pattern is None:
            raise ValueError(f"{f.path}: unsupported attention pattern {f.meta(_key('attention.pattern'))!r}")
        names = tuple(f.meta(_key("condition.names")))
        units = f.meta(_key("condition.units"))
        text_vocab = int(f.meta(_key("text_vocab")))
        spec = json.loads(f.meta(_key("text_tokenizer_spec")))
        spec["path"] = str(_cache_tokenizer(f.meta(_key("tokenizer.json"))))
        span_end = f.meta(_key("span_end_id"), None)
        return cls(
            n_layers=int(f.meta(_key("block_count"))),
            max_seqlen=int(f.meta(_key("context_length"))),
            dim=int(f.meta(_key("embedding_length"))),
            head_dim=int(f.meta(_key("head_dim"))),
            n_heads=int(f.meta(_key("attention.head_count"))),
            n_kv_heads=int(f.meta(_key("attention.head_count_kv"))),
            ffn_hidden=int(f.meta(_key("feed_forward_length"))),
            n_experts=int(f.meta(_key("expert_count"))),
            moe_topk=int(f.meta(_key("expert_used_count_per_layer"))[1]),
            n_codebooks=int(f.meta(_key("n_codebooks"))),
            codebook_size=int(f.meta(_key("codebook_size"))),
            audio_vocab=int(f.meta(_key("audio_vocab"))),
            eoa_id=int(f.meta(_key("eoa_id"))),
            audio_pad_id=int(f.meta(_key("audio_pad_id"))),
            span_end_id=None if span_end is None else int(span_end),
            text_vocab=text_vocab,
            text_extra_tokens=f.shape("text_embd.weight")[0] - text_vocab - 1,
            gdn_heads=int(f.meta(_key("gdn.head_count"))),
            gdn_head_dim=int(f.meta(_key("gdn.head_dim"))),
            gdn_kernel=int(f.meta(_key("gdn.conv_kernel"))),
            speaker_embedding_dim=int(f.meta(_key("speaker.embedding_length"))),
            condition_count=int(f.meta(_key("condition.count"))),
            condition_n_buckets=int(f.meta(_key("condition.bucket_count"))),
            condition_names=names,
            condition_units=dict(zip(names, units)),
            condition_edges={n: tuple(float(v) for v in f.meta(_key(f"condition.edges.{n}"))) for n in names},
            rope_theta=float(f.meta(_key("rope.freq_base"))),
            norm_eps=float(f.meta(_key("attention.layer_norm_rms_epsilon"))),
            qk_eps=float(f.meta(_key("attention.qk_norm_epsilon"))),
            loss_softcap=float(f.meta(_key("logit_softcap"))),
            gdn_period=int(pattern.group(1)) + 1,
            silence_frame=tuple(int(v) for v in f.meta(_key("silence_frame"))),
            text_tokenizer_spec=spec,
            tag_cfg_trained=bool(f.meta(_key("tag_cfg_trained"), False)),
        )

    @property
    def n_sources(self) -> int:
        return 2 * self.n_layers + 1

    @property
    def codebook_sizes(self) -> tuple[int, ...]:
        return (self.codebook_size,) * self.n_codebooks

    @property
    def text_rows(self) -> int:
        return self.text_vocab + 1 + self.text_extra_tokens

    @property
    def gdn_inner(self) -> int:
        return self.gdn_heads * self.gdn_head_dim

    def uses_gdn(self, layer: int) -> bool:
        return (layer + 1) % self.gdn_period != 0

    @property
    def attention_layers(self) -> tuple[int, ...]:
        return tuple(layer for layer in range(self.n_layers) if not self.uses_gdn(layer))


def _cache_tokenizer(text: str) -> Path:
    """The embedded tokenizer as a file (zonos3.tts.superbpe reads specs by path and SHA-256)."""
    raw = text.encode("utf-8")
    path = CACHE_DIR / f"superbpe-{hashlib.sha256(raw).hexdigest()}.json"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{id(raw)}")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return path


# ----------------------------------------------------------------------------------------------------- state


@dataclass
class LMState:
    """Decoding state of one sequence: the next absolute position, the global layers' KV cache (``[n_kv_heads, capacity,
    head_dim]``, grown in blocks of 256 rows; rows ``< position`` are valid), each GDN layer's recurrent state
    (``[heads, value, key]``) and causal-conv history (``[channels, kernel - 1]``)."""

    position: int = 0
    kv: dict[int, tuple[mx.array, mx.array]] = field(default_factory=dict)
    gdn_state: dict[int, mx.array] = field(default_factory=dict)
    conv_state: dict[int, mx.array] = field(default_factory=dict)

    def fork(self) -> "LMState":
        return LMState(
            self.position,
            {k: (mx.array(a), mx.array(b)) for k, (a, b) in self.kv.items()},
            {k: mx.array(v) for k, v in self.gdn_state.items()},
            {k: mx.array(v) for k, v in self.conv_state.items()},
        )

    def arrays(self) -> list[mx.array]:
        out = [a for pair in self.kv.values() for a in pair]
        return out + list(self.gdn_state.values()) + list(self.conv_state.values())

    def kv_rows(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """The valid KV rows of a global layer as ``[rows, n_kv_heads, head_dim]`` (tools/dump_golden.py layout)."""
        k, v = self.kv[layer]
        n = self.position
        return (np.array(k[:, :n].transpose(1, 0, 2)), np.array(v[:, :n].transpose(1, 0, 2)))


class LMOutput(NamedTuple):
    logits: mx.array  # [rows, n_codebooks, audio_vocab], softcapped
    probe: list[tuple[int, float]] | None


def _to_mx(value) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(mx.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return mx.array(np.asarray(value, dtype=np.float32))


def _rms(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _softplus(x: mx.array) -> mx.array:
    return mx.logaddexp(x, mx.zeros_like(x))


def _gather_matmul(x: mx.array, W: Weight, idx: mx.array) -> mx.array:
    """Per routed pair ``x[t, r] @ W[idx[t, r]].T``: x [n, k, 1, in], idx [n, k] -> [n, k, 1, out]."""
    if W.scales is not None:
        return mx.gather_qmm(x, W.w, W.scales, W.biases, rhs_indices=idx, transpose=True, group_size=GROUP_SIZE,
                             bits=W.bits)
    if idx.size >= W.w.shape[0]:
        return mx.gather_mm(x, W.w.astype(mx.float32).swapaxes(-1, -2), rhs_indices=idx)
    picked = W.w[idx].astype(mx.float32)  # [n, k, out, in]
    return x @ picked.swapaxes(-1, -2)


# ----------------------------------------------------------------------------------------------------- model


class Zonos3MLX:
    def __init__(self, config: LMConfig, weights: dict[str, Weight], tables: dict[str, mx.array], path: str):
        self.config = config
        self.w = weights
        self.t = tables
        self.path = path
        self._capture: dict | None = None
        self._accents = None

    @classmethod
    def load(cls, path: str | Path) -> "Zonos3MLX":
        f = GGUFFile(path)
        config = LMConfig.from_gguf(f)
        tables, weights = {}, {}
        for name in f.names():
            # float32 tables: embeddings, AttnRes queries, 1-D tensors and the GDN conv taps; the rest are matmul weights
            if len(f.shape(name)) == 1 or name.endswith(".gdn_conv.weight") or name == "attnres_q.weight" \
                    or name.startswith(("audio_embd.", "text_embd.", "cond_embd.")):
                tables[name] = f.f32(name)
            else:
                weights[name] = f.weight(name)
        model = cls(config, weights, tables, str(path))
        mx.eval(list(tables.values()) + [w.w for w in weights.values()])
        return model

    # -- helpers -----------------------------------------------------------------------------------------------------

    def _save(self, name: str, value: mx.array) -> None:
        if self._capture is not None:
            self._capture[name] = np.array(value.astype(mx.float32))

    def _lin(self, base: str, x: mx.array) -> mx.array:
        y = matmul(x, self.w[base + ".weight"])
        return y if self._accents is None else self._accents.linear(base, x, y)

    # -- sublayers ---------------------------------------------------------------------------------------------------

    def _global_attention(self, x, layer: int, state: LMState, probe: ProbeSpec | None, queries: dict):
        c, p = self.config, f"blk.{layer}."
        n, nh, nkv, hd = x.shape[0], c.n_heads, c.n_kv_heads, c.head_dim
        lo = state.position
        gate = mx.sigmoid(self._lin(p + "attn_gate", x))
        q = self._lin(p + "attn_q", x).reshape(n, nh, hd)
        k = self._lin(p + "attn_k", x).reshape(n, nkv, hd)
        v = self._lin(p + "attn_v", x).reshape(n, nkv, hd)
        q = _rms(q, None, c.qk_eps) * self.t[p + "attn_temp"][:, None]
        k = _rms(k, None, c.qk_eps)
        q = mx.fast.rope(q.transpose(1, 0, 2), hd, traditional=True, base=c.rope_theta, scale=1.0, offset=lo)
        k = mx.fast.rope(k.transpose(1, 0, 2), hd, traditional=True, base=c.rope_theta, scale=1.0, offset=lo)
        v = v.transpose(1, 0, 2)
        kc, vc = state.kv.get(layer, (None, None))
        need = lo + n
        if kc is None or kc.shape[1] < need:
            cap = -(-need // _KV_BLOCK) * _KV_BLOCK
            grow = cap - (0 if kc is None else kc.shape[1])
            pad = mx.zeros((nkv, grow, hd), dtype=mx.float32)
            kc = pad if kc is None else mx.concatenate([kc, pad], axis=1)
            vc = mx.zeros((nkv, grow, hd), dtype=mx.float32) if vc is None else mx.concatenate([vc, pad], axis=1)
        kc[:, lo:need] = k
        vc[:, lo:need] = v
        state.kv[layer] = (kc, vc)
        keys, values = kc[:, :need], vc[:, :need]
        mask = None
        if n > 1:
            mask = mx.arange(need)[None, :] <= (lo + mx.arange(n))[:, None]
        out = mx.fast.scaled_dot_product_attention(q[None], keys[None], values[None], scale=hd ** -0.5, mask=mask)[0]
        if probe is not None:
            queries[layer] = q  # [nh, n, hd]
            if probe.windows:
                out = self._masked_heads(layer, q, keys, values, out, lo, probe)
        out = out.transpose(1, 0, 2) * gate[:, :, None]
        return self._lin(p + "attn_output", out.reshape(n, nh * hd))

    def _masked_heads(self, layer, q, keys, values, out, lo, probe: ProbeSpec):
        """ACI hard mask: the masked heads' output over all keys with the text keys outside their window blocked."""
        c = self.config
        rep = c.n_heads // c.n_kv_heads
        n, total = q.shape[1], keys.shape[1]
        tidx = mx.arange(total) - probe.text_start
        in_text = (tidx >= 0) & (tidx < probe.text_len)
        causal = mx.arange(total)[None, :] <= (lo + mx.arange(n))[:, None]
        heads = []
        for head in range(c.n_heads):
            window = probe.windows.get((layer, head))
            if window is None:
                heads.append(out[head])
                continue
            wlo, whi = clip_window(window, probe.text_len)
            outside = in_text & ((tidx < wlo) | (tidx > whi))
            logits = (q[head] @ keys[head // rep].T) * c.head_dim ** -0.5
            logits = mx.where(outside[None, :] | ~causal, -mx.inf, logits)
            heads.append(mx.softmax(logits, axis=-1) @ values[head // rep])
        return mx.stack(heads)

    def _gdn(self, x, layer: int, state: LMState):
        c, p = self.config, f"blk.{layer}."
        H, S, K = c.gdn_heads, c.gdn_head_dim, c.gdn_kernel
        inner = H * S
        n = x.shape[0]
        projection = self._lin(p + "gdn_in", x)
        qkv = projection[:, : 3 * inner]
        z = projection[:, 3 * inner : 4 * inner]
        a = projection[:, 4 * inner : 4 * inner + H]
        b = projection[:, 4 * inner + H : 4 * inner + 2 * H]
        history = state.conv_state.get(layer)
        if history is None:
            history = mx.zeros((3 * inner, K - 1), dtype=mx.float32)
        sequence = mx.concatenate([history.T, qkv], axis=0)  # [K - 1 + n, channels], oldest first
        weight = self.t[p + "gdn_conv.weight"]  # [channels, K], tap 0 oldest
        conv = sequence[0:n] * weight[:, 0]
        for j in range(1, K):
            conv = conv + sequence[j : j + n] * weight[:, j]
        conv = _silu(conv)
        state.conv_state[layer] = sequence[-(K - 1):].T
        q = conv[:, :inner].reshape(n, H, S)
        k = conv[:, inner : 2 * inner].reshape(n, H, S)
        v = conv[:, 2 * inner :].reshape(n, H, S)
        q = q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        decay = mx.exp(-mx.exp(self.t[p + "gdn_A_log"]) * _softplus(a + self.t[p + "gdn_dt_bias"]))
        beta = mx.sigmoid(b)
        s0 = state.gdn_state.get(layer)
        if s0 is None:
            s0 = mx.zeros((H, S, S), dtype=mx.float32)
        y, s1 = gated_delta((q * S ** -0.5)[None], k[None], v[None], decay[None], beta[None], s0[None])
        state.gdn_state[layer] = s1[0]
        recurrent = y[0]
        normalized = _rms(recurrent, self.t[p + "gdn_norm.weight"], 1e-5)
        gated = (normalized * _silu(z.reshape(n, H, S))).reshape(n, inner)
        if self._capture is not None:
            for name, tensor in (("gdn_conv", conv), ("gdn_q", q), ("gdn_k", k), ("gdn_v", v), ("gdn_decay", decay),
                                 ("gdn_beta", beta), ("gdn_recurrent", recurrent), ("gdn_gated", gated),
                                 ("gdn_state", state.gdn_state[layer]), ("gdn_conv_state", state.conv_state[layer])):
                self._save(f"{name}_{layer}", tensor)
        return self._lin(p + "attn_output", gated)

    def _feed_forward(self, x, layer: int):
        c, p = self.config, f"blk.{layer}."
        if p + "ffn_up.weight" in self.w:
            return self._lin(p + "ffn_down", self._lin(p + "ffn_up", x) * _silu(self._lin(p + "ffn_gate", x)))
        logits = self._lin(p + "router", _rms(x, self.t[p + "router_norm.weight"], c.norm_eps))
        self._save(f"router_logits_{layer}", logits)
        bias = self.t[p + "router_bias"]
        if self._accents is not None:
            bias = self._accents.router_bias(layer, bias)
        k = c.moe_topk
        choices = mx.argpartition(bias - logits, kth=k - 1, axis=-1)[:, :k]
        weights = mx.softmax(mx.take_along_axis(logits, choices, axis=-1), axis=-1)
        latent = self._lin(p + "ffn_latent_down", x)
        z = latent[:, None, None, :]
        gate = _gather_matmul(z, self.w[p + "ffn_gate_exps.weight"], choices)
        up = _gather_matmul(z, self.w[p + "ffn_up_exps.weight"], choices)
        if self._accents is not None:
            gate = self._accents.expert_shift(layer, "gate", choices, z, gate)
            up = self._accents.expert_shift(layer, "up", choices, z, up)
        hidden = _silu(gate) * up
        down = _gather_matmul(hidden, self.w[p + "ffn_down_exps.weight"], choices)
        if self._accents is not None:
            down = self._accents.expert_shift(layer, "down", choices, hidden, down)
        output = (down[:, :, 0, :] * weights[:, :, None]).sum(axis=1)
        return self._lin(p + "ffn_latent_up", output)

    def _replace_rows(self, x, lo, speaker_rows, proj_deltas, condition_rows):
        hi = lo + x.shape[0]
        for row, vec in (speaker_rows or {}).items():
            if lo <= row < hi:
                projected = matmul(_to_mx(vec).reshape(1, -1), self.w["speaker_proj.weight"])[0]
                projected = projected + self.t["speaker_proj.bias"]
                delta = (proj_deltas or {}).get(row)
                if delta is not None:
                    projected = projected + _to_mx(delta).reshape(-1)
                x[row - lo] = projected
        for table, row in enumerate(sorted(condition_rows or {})):
            if lo <= row < hi:
                x[row - lo] = self.t[f"cond_embd.{table}.weight"][int(condition_rows[row])]
        return x

    # -- forward -----------------------------------------------------------------------------------------------------

    def forward(self, ids, state: LMState, *, speaker_rows=None, proj_deltas=None, condition_rows=None,
                hidden_deltas=None, accents=None, probe: ProbeSpec | None = None, capture: dict | None = None
                ) -> LMOutput:
        """Logits of ``ids`` [rows, n_codebooks + 1] continuing ``state`` (updated in place); the keyword arguments
        have the semantics of ``zonos3.reference.lm.Zonos3LM.forward``."""
        c = self.config
        if hasattr(ids, "numpy"):
            ids = ids.detach().cpu().numpy()
        ids = np.asarray(ids).astype(np.int32)
        n = int(ids.shape[0])
        lo = state.position
        if lo + n > c.max_seqlen:
            raise ValueError(f"{lo + n} rows exceed the model context {c.max_seqlen}")
        self._capture, self._accents = capture, accents
        try:
            idx = mx.array(ids)
            x = self.t["audio_embd.0.weight"][idx[:, 0]]
            for cb in range(1, c.n_codebooks):
                x = x + self.t[f"audio_embd.{cb}.weight"][idx[:, cb]]
            x = x + self.t["text_embd.weight"][idx[:, c.n_codebooks]]
            self._save("emb_sum", x)
            if speaker_rows or condition_rows:
                x = self._replace_rows(x, lo, speaker_rows, proj_deltas, condition_rows)
            self._save("emb_cond", x)
            x = _rms(x, None, c.norm_eps)
            self._save("emb_norm", x)

            deltas = {int(k): _to_mx(v) for k, v in (hidden_deltas or {}).items()}
            queries = self.t["attnres_q.weight"]  # [n_sources, dim]
            bank: list[mx.array] = []
            scores: list[mx.array] = []

            def route(source: mx.array, index: int, query: int) -> mx.array:
                delta = deltas.get(index)
                if delta is not None:
                    source = source + delta
                bank.append(source)
                scores.append(_rms(source, None, c.norm_eps) @ queries.T)  # [rows, n_sources]
                if len(bank) == 1:
                    return source
                w = mx.softmax(mx.stack([s[:, query] for s in scores]), axis=0)  # [s, rows]
                return (mx.stack(bank) * w[:, :, None]).sum(axis=0)

            probe_queries: dict[int, mx.array] = {}
            source_index = 0
            mixed = route(x, 0, 0)
            for layer in range(c.n_layers):
                p = f"blk.{layer}."
                if layer:
                    mixed = route(x, source_index, 2 * layer)
                self._save(f"attnres_attn_{layer}", mixed)
                attn_in = _rms(mixed, self.t[p + "attn_norm.weight"], c.norm_eps)
                self._save(f"attn_in_{layer}", attn_in)
                if c.uses_gdn(layer):
                    attention = self._gdn(attn_in, layer, state)
                else:
                    attention = self._global_attention(attn_in, layer, state, probe, probe_queries)
                self._save(f"attn_out_{layer}", attention)
                mixed = route(attention, 2 * layer + 1, 2 * layer + 1)
                self._save(f"attnres_ffn_{layer}", mixed)
                ffn_in = _rms(mixed, self.t[p + "ffn_norm.weight"], c.norm_eps)
                self._save(f"ffn_in_{layer}", ffn_in)
                x = self._feed_forward(ffn_in, layer)
                self._save(f"ffn_out_{layer}", x)
                self._save(f"layer_x_{layer}", x)
                source_index = 2 * layer + 2
            mixed = route(x, source_index, 2 * c.n_layers)
            self._save("attnres_final", mixed)
            normalized = _rms(mixed, self.t["output_norm.weight"], c.norm_eps)
            self._save("out_norm", normalized)
            logits = self._lin("output", normalized).reshape(n, c.n_codebooks, c.audio_vocab)
            logits = c.loss_softcap * mx.tanh(logits / c.loss_softcap)
            self._save("logits", logits)
            state.position = lo + n
            mx.eval(logits, *state.arrays())
            readouts = None if probe is None else self._readout(probe, probe_queries, state)
            return LMOutput(logits, readouts)
        finally:
            self._capture = self._accents = None

    def _readout(self, probe: ProbeSpec, probe_queries: dict, state: LMState) -> list[tuple[int, float]]:
        import torch

        c = self.config
        layers = c.attention_layers
        weights = probe_weight(probe.heads, layers, c.n_heads)
        lo, hi = probe.text_start, probe.text_start + probe.text_len
        keys = torch.from_numpy(np.stack([np.array(state.kv[layer][0][:, lo:hi].transpose(1, 0, 2))
                                          for layer in layers]))  # [L, T, KV, D]
        q = torch.from_numpy(np.stack([np.array(probe_queries[layer].transpose(1, 0, 2)) for layer in layers]))
        # q [L, rows, H, D]
        return [readout(q[:, r], keys, layers, weights, probe.windows, c.head_dim ** -0.5) for r in range(q.shape[1])]
