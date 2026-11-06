"""TTS sampler for multi-codebook audio generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch
from zonos3.tts.sampler import active_repetition_window, clamp_sampling, sample_tts

if TYPE_CHECKING:
    from zonos3_sgl.core import TTSBatch


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create a tensor on device from a list."""
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


@dataclass
class TTSBatchSamplingArgs:
    """Sampling arguments for a TTS batch."""

    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    min_ps: torch.Tensor
    stop_token_ids: torch.Tensor  # (B,) the end token each request may sample
    text_vocab: int
    repetition_token_ids: torch.Tensor | None = None
    repetition_penalties: torch.Tensor | None = None
    # (B, n_codebooks) bool: the stop token is masked there (requests below their min_frames); None = no row is.
    stop_blocked: torch.Tensor | None = None
    generators: list[torch.Generator | None] = field(default_factory=list)


@dataclass
class TTSSampler:
    """Sampler for multi-codebook TTS output.

    end_token_ids lists the model's end tokens (EOA, plus span-end for edit
    checkpoints); each request samples only its own ``stop_id`` among them.
    """

    device: torch.device
    n_codebooks: int
    codebook_size: int
    text_vocab: int
    audio_pad_id: int
    end_token_ids: tuple[int, ...]
    codebook_sizes: tuple[int, ...] | None = None

    def prepare(self, batch: TTSBatch, token_pool: torch.Tensor | None = None) -> TTSBatchSamplingArgs:
        """Prepare sampling arguments from a TTS batch.

        Args:
            batch: TTS batch with requests
            token_pool: Optional device-side token pool for recent generated history

        Returns:
            TTSBatchSamplingArgs with per-sequence sampling parameters
        """
        params = [r.sampling_params for r in batch.reqs]
        stop_ids = [int(r.stop_id) for r in batch.reqs]
        for stop_id in stop_ids:
            if stop_id not in self.end_token_ids:
                raise ValueError(f"Request stop id {stop_id} is not an end token of this model {self.end_token_ids}")

        clamped = [clamp_sampling(p, n_codebooks=self.n_codebooks, codebook_size=self.codebook_size) for p in params]
        temps = [c.temperature for c in clamped]
        top_ks = [c.topk for c in clamped]
        top_ps = [c.top_p for c in clamped]
        min_ps = [c.min_p for c in clamped]
        repetition_penalties = [c.repetition_penalty for c in clamped]
        repetition_codebooks = [c.repetition_codebooks for c in clamped]
        generators = [r.rng for r in batch.reqs]
        repetition_token_ids = None
        repetition_penalties_tensor = None

        active_windows = [
            active_repetition_window(c, min(r.total_generated + r.history_rows, r.device_len))
            for r, c in zip(batch.reqs, clamped)
        ]
        max_window = max(active_windows, default=0)
        if max_window > 0:
            repetition_token_ids = torch.full(
                (len(batch.reqs), self.n_codebooks, max_window),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
            for i, (req, window) in enumerate(zip(batch.reqs, active_windows)):
                if window <= 0:
                    continue

                start = req.device_len - window
                if token_pool is None:
                    history = req.input_ids[start : req.device_len, : self.n_codebooks].to(
                        device=self.device, dtype=torch.int64, non_blocking=True
                    )
                else:
                    history = token_pool[req.table_idx, start : req.device_len, : self.n_codebooks].to(
                        dtype=torch.int64
                    )

                history = history.transpose(0, 1).contiguous()
                valid = (history >= 0) & (history < self.codebook_size)
                valid[repetition_codebooks[i] :] = False
                repetition_token_ids[i, :, -window:] = torch.where(valid, history, torch.full_like(history, -1))

            repetition_penalties_tensor = make_device_tensor(repetition_penalties, torch.float32, self.device)

        # sampling_params.min_frames: a stop token in codebook c at generated frame s ends the
        # generation at aligned frame s - c, so it is masked while s < min_frames + c.
        stop_blocked = None
        if any(p.min_frames > 0 for p in params):
            stop_blocked = make_device_tensor(
                [
                    [r.total_generated < p.min_frames + c for c in range(self.n_codebooks)]
                    for r, p in zip(batch.reqs, params)
                ],
                torch.bool,
                self.device,
            )
        return TTSBatchSamplingArgs(
            temperatures=make_device_tensor(temps, torch.float32, self.device),
            top_ks=make_device_tensor(top_ks, torch.int32, self.device),
            top_ps=make_device_tensor(top_ps, torch.float32, self.device),
            min_ps=make_device_tensor(min_ps, torch.float32, self.device),
            stop_token_ids=make_device_tensor(stop_ids, torch.int64, self.device),
            text_vocab=self.text_vocab,
            repetition_token_ids=repetition_token_ids,
            repetition_penalties=repetition_penalties_tensor,
            generators=generators,
            stop_blocked=stop_blocked,
        )

    def sample(self, logits: torch.Tensor, args: TTSBatchSamplingArgs) -> List[List[int]]:
        """Sample from multi-codebook logits.

        Args:
            logits: Shape (B, n_codebooks, vocab_size)
            args: Sampling arguments

        Returns:
            List of unpacked tokens [cb0, cb1, ..., cb8, text_placeholder]
        """
        return sample_tts(
            logits=logits,
            temperatures=args.temperatures,
            top_ks=args.top_ks,
            top_ps=args.top_ps,
            min_ps=args.min_ps,
            codebook_size=self.codebook_size,
            audio_pad_id=self.audio_pad_id,
            stop_token_ids=args.stop_token_ids,
            repetition_token_ids=args.repetition_token_ids,
            repetition_penalties=args.repetition_penalties,
            codebook_sizes=self.codebook_sizes,
            text_vocab=args.text_vocab,
            generators=args.generators,
            stop_blocked=args.stop_blocked,
        )

    def sample_to_tensor(self, logits: torch.Tensor, args: TTSBatchSamplingArgs) -> torch.Tensor:
        """Sample and return as tensor instead of list.

        Args:
            logits: Shape (B, n_codebooks, vocab_size)
            args: Sampling arguments

        Returns:
            Tensor of shape (B, n_codebooks + 1) with sampled tokens
        """
        tokens_list = self.sample(logits, args)
        return torch.tensor(tokens_list, dtype=torch.int32, device=self.device)
