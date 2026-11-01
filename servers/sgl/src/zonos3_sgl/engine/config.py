from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from zonos3_sgl.distributed import DistributedInfo
from zonos3.utils.hf import cached_load_checkpoint_config

if TYPE_CHECKING:
    from zonos3.models.config import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "fused_moe"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # Directory of accent adapters (<accent>.pt) served per request; see
    # zonos3_sgl.models.accent_adapters. None serves the base model only.
    accent_adapters: str | None = None
    # Alignment probes (zonos3_sgl.tts.align_probe) inside the decode CUDA graphs: text keys read per request (0 = graphs
    # carry no probe and any decode batch with a probe runs eager), masked requests per batch, keys per masked request.
    align_probe_max_text: int = 0
    align_probe_max_masked: int = 8
    align_probe_max_keys: int = 12288

    @cached_property
    def checkpoint_config(self):
        return cached_load_checkpoint_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from zonos3.models.config import ModelConfig

        return ModelConfig.from_checkpoint_config(self.checkpoint_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        import os

        # Several offline engines on one node (evaluations, parity checks) must not share the rendezvous port.
        return os.environ.get("ZONOS3_DISTRIBUTED_ADDR", "tcp://127.0.0.1:23333")
