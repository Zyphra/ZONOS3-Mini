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

Prefill CUDA graphs (zonos3_sgl.engine.prefill_graph) pad each fresh request to its complete chunks plus one 64-row
window and run ``_forward_prefill_static``: the complete chunks as the eager path's committed pass, the window (the
request's last ``n % 64`` rows, zeros after) as its uncommitted tail pass, laid out like the decode window.

Every pass calls FLA's forward functions directly (what causal_conv1d / chunk_gated_delta_rule / rms_norm_gated run
under their autograd wrappers) with cu_seqlens and chunk indices built on the host and copied once without blocking:
FLA derives the indices from device cu_seqlens with a host sync (repeat_interleave) per call.

Accent adapters (zonos3_sgl.models.accent_adapters) shift in_proj and wo per row.
Those projections run on the replayed rows too, so each forward expands the
request's [K] adapter-mask row over every row it feeds the kernels (a request
has one accent, so its replayed tail takes the same shift as when it was new).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from zonos3_sgl.core import get_global_ctx
from zonos3_sgl.layers.accent import accent_mask, apply_accent
from zonos3_sgl.layers.base import BaseOP
from zonos3_sgl.utils.torch_utils import make_device_tensor

CHUNK = 64  # the training kernel's chunk length


def chunk_tables(lengths: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(cu_seqlens [n + 1], FLA chunk indices [chunks, 2] = (sequence, chunk within it)) of packed sequences of
    ``lengths`` rows, int32 on ``device`` in one non-blocking copy (the values FLA's prepare_chunk_indices derives)."""
    cu = [0]
    chunks: list[int] = []
    for i, n in enumerate(lengths):
        cu.append(cu[-1] + n)
        for j in range(-(-n // CHUNK)):
            chunks += (i, j)
    packed = make_device_tensor(cu + chunks, torch.int32, device)
    return packed[: len(cu)], packed[len(cu) :].view(-1, 2)


@dataclass(frozen=True)
class PrefillLayout:
    """Static GDN tables of a padded prefill: request i owns ``CHUNK * (whole[i] + 1)`` rows of the batch from
    ``starts[i]``, its ``whole[i]`` complete chunks, then a 64-row window for its remaining ``n_i % 64`` rows."""

    whole_rows: torch.Tensor | None  # [sum 64 * whole] long: batch rows of the complete chunks
    whole_reqs: torch.Tensor | None  # [k] long: batch index of each request with complete chunks
    whole_cu: torch.Tensor | None
    whole_chunks: torch.Tensor | None
    window_rows: torch.Tensor  # [b * 64] long: batch rows of the windows
    window_base: torch.Tensor  # [b] long: 64 * whole[i], the rows before request i's window
    window_cu: torch.Tensor
    window_chunks: torch.Tensor

    @classmethod
    def build(cls, whole: list[int], device: torch.device) -> PrefillLayout:
        starts, start = [], 0
        for w in whole:
            starts.append(start)
            start += CHUNK * (w + 1)
        with_whole = [i for i, w in enumerate(whole) if w]
        whole_rows = [r for i in with_whole for r in range(starts[i], starts[i] + CHUNK * whole[i])]
        window_rows = [
            r for i, w in enumerate(whole) for r in range(starts[i] + CHUNK * w, starts[i] + CHUNK * (w + 1))
        ]
        whole_cu = whole_chunks = None
        if with_whole:
            whole_cu, whole_chunks = chunk_tables([CHUNK * whole[i] for i in with_whole], device)
        window_cu, window_chunks = chunk_tables([CHUNK] * len(whole), device)
        return cls(
            whole_rows=make_device_tensor(whole_rows, torch.long, device) if with_whole else None,
            whole_reqs=make_device_tensor(with_whole, torch.long, device) if with_whole else None,
            whole_cu=whole_cu,
            whole_chunks=whole_chunks,
            window_rows=make_device_tensor(window_rows, torch.long, device),
            window_base=make_device_tensor([CHUNK * w for w in whole], torch.long, device),
            window_cu=window_cu,
            window_chunks=window_chunks,
        )

    def tensors(self) -> list[torch.Tensor]:
        """The cu_seqlens / chunk-index tensors FLA memoizes results by (OriginalGDN.pin_fla_cache)."""
        return [t for t in (self.whole_cu, self.whole_chunks, self.window_cu, self.window_chunks) if t is not None]


@dataclass
class PrefillInputs:
    """A padded prefill's GDN inputs (TTSBatch.gdn_prefill): the layout and the graph's per-replay buffers."""

    layout: PrefillLayout
    slots: torch.Tensor  # [b] long: the requests' table slots
    lengths: torch.Tensor  # [b] long: the requests' real rows


class _Weight(BaseOP):
    def __init__(self, shape):
        self.weight = torch.empty(shape)


class OriginalGDN(BaseOP):
    def __init__(self, config, layer_id):
        from fla.modules.conv.triton.ops import causal_conv1d_fwd
        from fla.modules.fused_norm_gate import layer_norm_gated_fwd
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.common.gate import fused_beta_sigmoid_fwd
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

        self._conv_fwd = causal_conv1d_fwd
        self._norm_fwd = layer_norm_gated_fwd
        self._l2norm_fwd = l2norm_fwd
        self._beta_fwd = fused_beta_sigmoid_fwd
        self._delta_rule_fwd = chunk_gated_delta_rule_fwd
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
        self._static_chunks: dict[int, torch.Tensor] = {}

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
    def static_tables(device: torch.device, bs_list) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Per batch size: (cu_seqlens of bs 64-row windows, their chunk indices, row offsets bs*64). Built ONCE by the
        engine and shared by every GDN layer: FLA memoizes results derived from cu_seqlens by tensor identity in
        4-entry caches, and a captured graph keeps pointers to them (pin_fla_cache)."""
        tables = {}
        for bs in bs_list:
            cu, chunks = chunk_tables([CHUNK] * bs, device)
            tables[bs] = (cu, chunks, torch.arange(bs, device=device, dtype=torch.long) * CHUNK)
        return tables

    def enable_graph_decode(
        self, slots: torch.Tensor, tables: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> None:
        """``slots`` is the engine's static [max_bs] int64 buffer of table indices for the padded decode batch (the
        engine fills it before every capture / replay); ``tables`` comes from ``static_tables``."""
        self._graph_slots = slots
        for bs, (cu, chunks, rows) in tables.items():
            self._static_cu[bs] = cu
            self._static_chunks[bs] = chunks
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

    def note_graph_forward(self, reqs) -> None:
        """Host-side mirrors after a replayed decode step or prefill (the graph advanced the device state)."""
        for r in reqs:
            self._owners[r.table_idx] = (r.uid, r.is_cfg_uncond)
            self._lengths[r.table_idx] = r.device_len

    # ------------------------------------------------------------------ forward
    def forward(self, x):
        batch = get_global_ctx().batch
        if batch.is_decode and self._static_mode:
            return self._forward_decode_static(x, batch.padded_size)
        if batch.gdn_prefill is not None:
            return self._forward_prefill_static(x, batch.gdn_prefill)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("GDN ragged path cannot be captured; graphs use the static window / prefill paths")
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
            starts = [sum(lengths[:i]) for i in range(len(lengths))]
            mask = mask.index_select(0, make_device_tensor(starts, torch.long, mask.device))
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
            indices = make_device_tensor([slots[i] for i in which], torch.long, x.device)
            conv = self._conv_state.index_select(0, indices)
            recurrent = self._recurrent_state.index_select(0, indices)
            rows = None if mask is None else torch.cat([mask[i].expand(len(v), -1) for i, v in zip(which, values)])
            cu, chunks = chunk_tables([len(v) for v in values], x.device)
            out, conv_final, final = self._run_packed(torch.cat(values), cu, chunks, conv, recurrent, rows)
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
        tail_lengths = []
        for i, (r, owner) in enumerate(zip(reqs, owners)):
            slot = r.table_idx
            tail = combined[i][complete_lengths[i] :]
            n = len(tail)
            self._window[slot, :n] = tail
            self._window[slot, n:].zero_()
            tail_lengths.append(n)
            self._owners[slot] = owner
            self._lengths[slot] = r.device_len
        self._win_len.index_copy_(
            0,
            make_device_tensor(slots, torch.long, x.device),
            make_device_tensor(tail_lengths, torch.int32, x.device),
        )
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
        out, conv_final, final = self._run_packed(
            win, self._static_cu[bs], self._static_chunks[bs], conv, recurrent, mask
        )
        y = out.index_select(0, pos)
        new_len = (lens + 1) % CHUNK
        complete = new_len == 0
        self._conv_state.index_copy_(0, slots, torch.where(complete.view(bs, 1, 1), conv_final, conv))
        self._recurrent_state.index_copy_(0, slots, torch.where(complete.view(bs, 1, 1, 1), final.float(), recurrent))
        win = win.view(bs, CHUNK, self._hidden)
        self._window.index_copy_(0, slots, torch.where(complete.view(bs, 1, 1), torch.zeros_like(win), win))
        self._win_len.index_copy_(0, slots, new_len)
        return y

    def _forward_prefill_static(self, x, inputs: PrefillInputs):
        """A padded prefill of fresh requests (PrefillLayout): the eager path's committed pass over the complete
        chunks and its uncommitted pass over each request's last ``n % 64`` rows, the latter as a 64-row window with
        zeros after the real rows (the decode window); every shape depends on the layout only."""
        if accent_mask() is not None:
            raise RuntimeError("GDN static prefill does not take accent adapters")
        layout, slots = inputs.layout, inputs.slots
        bs = slots.numel()
        self._conv_state.index_fill_(0, slots, 0)
        self._recurrent_state.index_fill_(0, slots, 0)
        y = torch.empty_like(x)
        if layout.whole_rows is not None:
            whole_slots = slots.index_select(0, layout.whole_reqs)
            out, conv_final, final = self._run_packed(
                x.index_select(0, layout.whole_rows),
                layout.whole_cu,
                layout.whole_chunks,
                self._conv_state.index_select(0, whole_slots),
                self._recurrent_state.index_select(0, whole_slots),
            )
            self._conv_state.index_copy_(0, whole_slots, conv_final)
            self._recurrent_state.index_copy_(0, whole_slots, final.float())
            y.index_copy_(0, layout.whole_rows, out)
        tail = inputs.lengths - layout.window_base  # real rows in each window
        real = torch.arange(CHUNK, device=x.device).view(1, CHUNK) < tail.view(bs, 1)
        win = x.index_select(0, layout.window_rows).masked_fill(~real.view(-1, 1), 0)
        out, _, _ = self._run_packed(
            win,
            layout.window_cu,
            layout.window_chunks,
            self._conv_state.index_select(0, slots),
            self._recurrent_state.index_select(0, slots),
        )
        y.index_copy_(0, layout.window_rows, out)
        self._window.index_copy_(0, slots, win.view(bs, CHUNK, self._hidden))
        self._win_len.index_copy_(0, slots, tail.to(torch.int32))
        return y

    def _run_packed(self, x, cu, chunks, conv, recurrent, mask=None):
        """Packed sequences ``cu`` (chunk indices ``chunks``) from states ``conv`` / ``recurrent``. ``mask``
        [rows of x, K]: per-row accent adapter mask (None: no adapters). The FLA forward calls receive contiguous
        inputs, as the autograd wrappers' input_guard makes them."""
        mixed = F.linear(x, self.in_proj.weight)
        if mask is not None:
            mixed = apply_accent(self.in_proj, x, mixed, mask)
        qkv, z, a, b = mixed.unsqueeze(0).split([1536, 512, 4, 4], dim=-1)
        qkv, conv_final = self._conv_fwd(
            x=qkv.contiguous(),
            weight=self.conv.weight.squeeze(1),
            bias=None,
            residual=None,
            initial_state=conv,
            output_final_state=True,
            activation="silu",
            cu_seqlens=cu,
            chunk_indices=chunks,
            BT=CHUNK,
        )
        q, k, v = [t.unflatten(-1, (4, 128)).contiguous() for t in qkv.split(512, dim=-1)]
        q, _ = self._l2norm_fwd(q)
        k, _ = self._l2norm_fwd(k)
        _, output, _, final, _, _ = self._delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=a.contiguous(),
            beta=self._beta_fwd(b.contiguous(), 1.0),
            scale=128**-0.5,
            initial_state=recurrent,
            output_final_state=True,
            state_v_first=True,
            cu_seqlens=cu,
            chunk_indices=chunks,
            use_gate_in_kernel=True,
            A_log=self.A_log.float(),
            dt_bias=self.dt_bias.float(),
            chunk_size=CHUNK,
        )
        output = self._norm_fwd(
            x=output.to(q.dtype).reshape(-1, 128),
            g=z.reshape(-1, 128).contiguous(),
            weight=self.out_weight,
            bias=None,
            activation="swish",
            eps=1e-5,
            residual=None,
            residual_dtype=None,
            is_rms_norm=True,
        )[0]
        output = output.reshape(x.shape[0], 512)
        result = F.linear(output, self.wo.weight)
        if mask is not None:
            result = apply_accent(self.wo, output, result, mask)
        return result, conv_final, final
