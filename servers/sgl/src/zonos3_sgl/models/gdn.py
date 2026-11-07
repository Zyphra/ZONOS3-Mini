"""Original four-head GDN, with independent request-local inference state.

The small mixer is replicated across TP ranks. It neither shards wo nor
all-reduces its output; the surrounding global attention and FFNs retain TP.
All persistent model tensors use BaseOP's strict checkpoint loader. Caches
start with an underscore so they cannot masquerade as checkpoint weights.

Exactness model: the trainer runs FLA's 64-frame chunked kernels over packed
sequences. Persistent per-slot states therefore end at a multiple of 64 frames
and every forward replays the unfinished chunk (the "tail") in front of the new
rows, which keeps the training arithmetic across arbitrary prefill / decode
cuts. The tail lives in a fixed per-slot 64-row window so that decode can also
run as a static-shape CUDA graph: each request's window is processed as one
64-row sequence (real rows first, zero rows after; the kernels are causal, so
the padded rows do not change the real rows - receipt
sgl_frozen_arch_20261012/checks/graph-capturability-375k.json), the new row's
output is gathered from its position, and the states are committed only when
the window fills up.

Accent adapters (zonos3_sgl.models.accent_adapters) shift in_proj and wo per row.
Those projections run on the replayed rows too, so each forward expands the
request's [K] adapter-mask row over every row it feeds the kernels (a request
has one accent, so its replayed tail takes the same shift as when it was new).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from zonos3_sgl.core import get_global_ctx
from zonos3_sgl.layers.accent import accent_mask, apply_accent
from zonos3_sgl.layers.base import BaseOP

CHUNK = 64  # the training kernel's chunk length


class _Weight(BaseOP):
    def __init__(self, shape):
        self.weight = torch.empty(shape)


class OriginalGDN(BaseOP):
    def __init__(self, config, layer_id):
        from fla.modules.conv.causal_conv1d import causal_conv1d
        from fla.modules.fused_norm_gate import rms_norm_gated
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
        )

        self._conv_fn = causal_conv1d
        self._norm_fn = rms_norm_gated
        self._prefill_fn = chunk_gated_delta_rule
        self._layer_id = layer_id
        self._hidden = config.hidden_size
        self.in_proj = _Weight((2056, config.hidden_size))
        self.conv = _Weight((1536, 1, 4))
        self.out_weight = torch.empty(128)
        self.A_log = torch.empty(4, dtype=torch.float32)
        self.dt_bias = torch.empty(4, dtype=torch.float32)
        self.wo = _Weight((config.hidden_size, 512))
        self._capacity = 0
        # CUDA-graph decode support (see enable_graph_decode)
        self._static_mode = False
        self._graph_slots: torch.Tensor | None = None
        self._static_cu: dict[int, torch.Tensor] = {}
        self._static_rows: dict[int, torch.Tensor] = {}

    def allocate_cache(self, capacity, device, dtype):
        if self._capacity:
            raise RuntimeError("GDN cache already allocated")
        self._conv_state = torch.zeros((capacity, 1536, 4), device=device, dtype=dtype)
        self._recurrent_state = torch.zeros((capacity, 4, 128, 128), device=device, dtype=torch.float32)
        # Unfinished-chunk inputs per slot: rows [0, device_len % 64) are real, the rest are zero.
        self._window = torch.zeros((capacity, CHUNK, self._hidden), device=device, dtype=dtype)
        self._win_len = torch.zeros((capacity,), device=device, dtype=torch.int32)
        self._owners = [None] * capacity
        self._lengths = [0] * capacity
        self._capacity = capacity

    # ------------------------------------------------------------------ CUDA graphs
    @staticmethod
    def static_tables(device: torch.device, bs_list) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Per batch size: (cu_seqlens of bs 64-row windows, row offsets bs*64). Built ONCE by the engine and shared
        by every GDN layer: FLA memoizes its chunk-index preparation by tensor identity in a 4-entry cache, and that
        preparation syncs with the host, which a CUDA graph capture forbids; nine layers with private copies of the
        same cu_seqlens would thrash the cache and recompute inside the capture."""
        return {
            bs: (
                torch.arange(0, bs + 1, device=device, dtype=torch.int32) * CHUNK,
                torch.arange(bs, device=device, dtype=torch.long) * CHUNK,
            )
            for bs in bs_list
        }

    def enable_graph_decode(self, slots: torch.Tensor, tables: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
        """``slots`` is the engine's static [max_bs] int64 buffer of table indices for the padded decode batch (the
        engine fills it before every capture / replay); ``tables`` comes from ``static_tables``."""
        self._graph_slots = slots
        for bs, (cu, rows) in tables.items():
            self._static_cu[bs] = cu
            self._static_rows[bs] = rows

    @staticmethod
    def pin_fla_cache(static_tensors, pinned: dict) -> int:
        """Keep alive every FLA tensor-cache result computed for the static cu_seqlens tensors.

        FLA memoizes chunk indices / offsets / lengths per cu_seqlens object in 4-entry FIFO caches. A captured graph
        keeps raw pointers to those result tensors; once eager prefills (fresh cu_seqlens every call) push them out of
        the cache they are freed and the memory is reused, and replays read garbage or fault. Scans the caches (the
        decorator's closure deques) for entries keyed by one of ``static_tensors`` and stores their results in
        ``pinned`` (keyed by id) for the lifetime of the engine. Returns the number of newly pinned tensors."""
        import sys
        from collections import deque

        keys = {id(t) for t in static_tensors}
        added = 0
        for name, module in list(sys.modules.items()):
            if not name.startswith("fla") or module is None:
                continue
            for attr in dir(module):
                fn = getattr(module, attr, None)
                closure = getattr(fn, "__closure__", None)
                if not closure:
                    continue
                for cell in closure:
                    try:
                        val = cell.cell_contents
                    except ValueError:
                        continue
                    if not isinstance(val, deque):
                        continue
                    for entry in list(val):
                        if len(entry) != 3:
                            continue
                        args, kwargs, result = entry
                        refs = list(args) + list(kwargs.values())
                        if any(id(a) in keys for a in refs) and id(result) not in pinned:
                            pinned[id(result)] = result
                            added += 1
        return added

    def set_static_mode(self, enabled: bool) -> None:
        """While set, decode batches run the fixed-shape window path (used for graph warm-up and capture)."""
        self._static_mode = bool(enabled)

    def note_graph_decode(self, reqs) -> None:
        """Host-side mirrors after a replayed decode step (the graph advanced the device state by one row)."""
        for r in reqs:
            self._owners[r.table_idx] = (r.uid, r.is_cfg_uncond)
            self._lengths[r.table_idx] = r.device_len

    # ------------------------------------------------------------------ forward
    def forward(self, x):
        batch = get_global_ctx().batch
        if batch.is_decode and self._static_mode:
            return self._forward_decode_static(x, batch.padded_size)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("GDN ragged (prefill) path cannot be captured; decode uses the static window path")
        reqs = batch.reqs
        if len(batch.padded_reqs) != len(reqs):
            raise RuntimeError("GDN eager path does not accept dummy graph-padding requests")
        slots = [r.table_idx for r in reqs]
        lengths = [r.extend_len for r in reqs]
        if (
            not self._capacity
            or len(set(slots)) != len(slots)
            or any(s < 0 or s >= self._capacity for s in slots)
            or any(n <= 0 for n in lengths)
            or sum(lengths) != x.shape[0]
        ):
            raise RuntimeError("Invalid GDN request slots or packed lengths")
        if batch.is_decode and any(n != 1 for n in lengths):
            raise RuntimeError("GDN decode expects one new token per request")
        owners = [(r.uid, r.is_cfg_uncond) for r in reqs]
        mask = accent_mask()
        if mask is not None:  # one [K] row per request: its first new token's
            starts = torch.tensor(lengths).cumsum(0) - torch.tensor(lengths)
            mask = mask.index_select(0, starts.to(mask.device))
        # Validate the entire batch before changing any cache. A global KV-only
        # prefix hit is deliberately rejected: it has no matching recurrent state.
        for r, owner in zip(reqs, owners):
            s = r.table_idx
            if r.cached_len and (self._owners[s] != owner or self._lengths[s] != r.cached_len):
                raise RuntimeError(
                    f"GDN layer {self._layer_id}: missing/stale state at slot {s}; "
                    f"expected {(owner, r.cached_len)}, "
                    f"found {(self._owners[s], self._lengths[s])}"
                )
        # Persistent states end at a multiple of the training kernel's 64-frame
        # chunk. Replay only the unfinished chunk, never the entire request.
        # This retains training arithmetic across arbitrary prefill/decode cuts.
        pieces = x.split(lengths)
        combined, old_tail_lengths, complete_lengths = [], [], []
        for r, new in zip(reqs, pieces):
            slot = r.table_idx
            if r.cached_len == 0:
                self._conv_state[slot].zero_()
                self._recurrent_state[slot].zero_()
                tail = new[:0]
            else:
                tail = self._window[slot, : r.cached_len % CHUNK]
            old_tail_lengths.append(len(tail))
            joined = torch.cat((tail, new)) if len(tail) else new
            combined.append(joined)
            complete_lengths.append(len(joined) // CHUNK * CHUNK)

        outputs = [[] for _ in reqs]

        def process(which, values, commit):
            indices = torch.tensor([slots[i] for i in which], device=x.device, dtype=torch.long)
            conv = self._conv_state.index_select(0, indices)
            recurrent = self._recurrent_state.index_select(0, indices)
            rows = None if mask is None else torch.cat([mask[i].expand(len(v), -1) for i, v in zip(which, values)])
            out, conv_final, final = self._run_chunk_segments(values, conv, recurrent, rows)
            for i, y in zip(which, out.split([len(v) for v in values])):
                outputs[i].append(y)
            if commit:
                self._conv_state.index_copy_(0, indices, conv_final)
                self._recurrent_state.index_copy_(0, indices, final.float())

        whole = [i for i, n in enumerate(complete_lengths) if n]
        if whole:
            process(whole, [combined[i][: complete_lengths[i]] for i in whole], True)
        partial = [i for i, v in enumerate(combined) if complete_lengths[i] < len(v)]
        if partial:
            process(partial, [combined[i][complete_lengths[i] :] for i in partial], False)
        for i, (r, owner) in enumerate(zip(reqs, owners)):
            slot = r.table_idx
            tail = combined[i][complete_lengths[i] :]
            n = len(tail)
            self._window[slot, :n] = tail
            self._window[slot, n:].zero_()
            self._win_len[slot] = n
            self._owners[slot] = owner
            self._lengths[slot] = r.device_len
        return torch.cat([torch.cat(parts)[n:] for parts, n in zip(outputs, old_tail_lengths)])

    def _forward_decode_static(self, x, bs):
        """One new row per padded request through fixed 64-row windows; every shape depends on ``bs`` only."""
        if self._graph_slots is None or bs not in self._static_cu:
            raise RuntimeError(f"GDN static decode is not prepared for batch size {bs}")
        slots = self._graph_slots[:bs]
        rows = self._static_rows[bs]
        lens = self._win_len.index_select(0, slots)
        pos = rows + lens.to(torch.long)
        win = self._window.index_select(0, slots).view(bs * CHUNK, self._hidden)
        win.index_copy_(0, pos, x.contiguous())
        conv = self._conv_state.index_select(0, slots)
        recurrent = self._recurrent_state.index_select(0, slots)
        mask = accent_mask()
        if mask is not None:  # [bs, K] -> one row per window row
            mask = mask.repeat_interleave(CHUNK, dim=0)
        out, conv_final, final = self._run_packed(win, self._static_cu[bs], conv, recurrent, mask)
        y = out.index_select(0, pos)
        new_len = (lens + 1) % CHUNK
        complete = new_len == 0
        self._conv_state.index_copy_(0, slots, torch.where(complete.view(bs, 1, 1), conv_final, conv))
        self._recurrent_state.index_copy_(0, slots, torch.where(complete.view(bs, 1, 1, 1), final.float(), recurrent))
        win = win.view(bs, CHUNK, self._hidden)
        self._window.index_copy_(0, slots, torch.where(complete.view(bs, 1, 1), torch.zeros_like(win), win))
        self._win_len.index_copy_(0, slots, new_len)
        return y

    def _run_chunk_segments(self, values, conv, recurrent, mask=None):
        x = torch.cat(values)
        boundaries = [0]
        for v in values:
            boundaries.append(boundaries[-1] + len(v))
        cu = torch.tensor(boundaries, device=x.device, dtype=torch.int32)
        return self._run_packed(x, cu, conv, recurrent, mask)

    def _run_packed(self, x, cu, conv, recurrent, mask=None):
        """``mask`` [rows of x, K]: per-row accent adapter mask (None: no adapters)."""
        mixed = F.linear(x, self.in_proj.weight)
        if mask is not None:
            mixed = apply_accent(self.in_proj, x, mixed, mask)
        qkv, z, a, b = mixed.unsqueeze(0).split([1536, 512, 4, 4], dim=-1)
        qkv, conv_final = self._conv_fn(
            qkv.contiguous(),
            self.conv.weight.squeeze(1),
            initial_state=conv,
            output_final_state=True,
            activation="silu",
            backend="triton",
            cu_seqlens=cu,
        )
        q, k, v = [t.unflatten(-1, (4, 128)) for t in qkv.split(512, dim=-1)]
        output, final = self._prefill_fn(
            q,
            k,
            v,
            g=a,
            beta=b,
            initial_state=recurrent,
            output_final_state=True,
            cu_seqlens=cu,
            state_v_first=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=self.A_log.float(),
            dt_bias=self.dt_bias.float(),
        )
        output = self._norm_fn(
            output.reshape(-1, 128), z.reshape(-1, 128), self.out_weight, None, activation="swish", eps=1e-5
        )
        output = output.reshape(x.shape[0], 512)
        result = F.linear(output, self.wo.weight)
        if mask is not None:
            result = apply_accent(self.wo, output, result, mask)
        return result, conv_final, final
