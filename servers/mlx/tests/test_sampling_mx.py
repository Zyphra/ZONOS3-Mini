"""The device sampler (zonos3_mlx.sampling) against the reference sampler (zonos3.reference.synth.sample_row): same
random stream, the same rows up to float rounding."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch
from zonos3.reference.synth import Sampling, sample_row
from zonos3.speaker_cfg import guided_logits
from zonos3.tts.delay import apply_cold_start_delay
from zonos3.tts.sampler import clamp_sampling
from zonos3_mlx.sampling import guided_logits_mx, noise_for, repetition_ids, sample_row_mx

N_CB, CODEBOOK, EOA, PAD, TEXT_VOCAB = 10, 4096, 4096, 4097, 8196
CONFIG = SimpleNamespace(
    n_codebooks=N_CB,
    codebook_size=CODEBOOK,
    codebook_sizes=(CODEBOOK,) * N_CB,
    audio_pad_id=PAD,
    text_vocab=TEXT_VOCAB,
)


def device_row(logits, sampling, history, total_generated, generator, *, has_audio_prefix=True):
    s = clamp_sampling(sampling, n_codebooks=N_CB, codebook_size=CODEBOOK)
    available = min(total_generated, len(history))
    stop_blocked = None
    if sampling.min_frames > 0:
        stop_blocked = np.array([total_generated < sampling.min_frames + cb for cb in range(N_CB)])
    row = sample_row_mx(
        mx.array(logits.numpy()),
        mx.array(noise_for(generator, N_CB, max(EOA, PAD) + 1)),
        s,
        repetition_ids(s, history, available, n_codebooks=N_CB, codebook_size=CODEBOOK),
        codebook_size=CODEBOOK,
        codebook_sizes=CONFIG.codebook_sizes,
        audio_pad_id=PAD,
        stop_id=EOA,
        stop_blocked=stop_blocked,
        total_generated=total_generated,
        has_audio_prefix=has_audio_prefix,
        text_vocab=TEXT_VOCAB,
    )
    return np.array(row)


def reference_row(logits, sampling, history, total_generated, generator, *, has_audio_prefix=True):
    row = sample_row(CONFIG, logits, sampling, history, total_generated, 0, EOA, generator)
    apply_cold_start_delay(row, total_generated, n_codebooks=N_CB, pad_id=PAD, has_audio_prefix=has_audio_prefix)
    return row.numpy()


def random_logits(rng: np.random.Generator) -> torch.Tensor:
    logits = rng.normal(0.0, 1.0, (N_CB, CODEBOOK + 2)) * rng.uniform(0.5, 4.0, (N_CB, 1))
    logits[:, rng.integers(0, CODEBOOK, (N_CB, 8))] += 6.0  # a few peaked codes, so top-p / min-p cut
    logits[:, EOA] = rng.normal(0.0, 4.0, N_CB)
    return torch.from_numpy(logits.astype(np.float32))


def test_device_sampler_matches_reference_stream():
    rng = np.random.default_rng(11)
    sampling = Sampling(
        temperature=1.0,
        topk=100,
        top_p=0.9,
        min_p=0.05,
        repetition_window=96,
        repetition_penalty=2.4,
        repetition_codebooks=6,
        min_frames=20,
    )
    history = torch.from_numpy(rng.integers(0, CODEBOOK, (160, N_CB + 1)).astype(np.int64))
    history[::7, 3] = EOA  # specials in the history never count as repeats
    g_device, g_ref = torch.Generator().manual_seed(1234), torch.Generator().manual_seed(1234)
    rows, agree = 500, 0
    for k in range(rows):
        logits = random_logits(rng)
        total = k % 40  # min_frames blocks the stop id on the early rows only
        prefix = k % 5 != 0  # every fifth row runs the cold-start delay
        got = device_row(logits, sampling, history, total, g_device, has_audio_prefix=prefix)
        want = reference_row(logits, sampling, history, total, g_ref, has_audio_prefix=prefix)
        agree += int(np.array_equal(got, want))
    assert agree >= 0.99 * rows, f"{agree}/{rows} rows agree"
    assert torch.equal(g_device.get_state(), g_ref.get_state())


def test_greedy_fallback_and_cfg_mix():
    rng = np.random.default_rng(5)
    cond, uncond = random_logits(rng), random_logits(rng)
    for scale in (0.0, 1.0, 2.5):
        want = guided_logits(cond, uncond, scale).numpy()
        got = np.array(guided_logits_mx(mx.array(cond.numpy()), mx.array(uncond.numpy()), scale))
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    # min_p > 1 empties every row: both samplers fall back to the argmax (and still draw the noise)
    sampling = Sampling(min_p=1.5, repetition_penalty=1.0)
    history = torch.zeros((1, N_CB + 1), dtype=torch.int64)
    g_device, g_ref = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    got = device_row(cond, sampling, history, 30, g_device)
    want = reference_row(cond, sampling, history, 30, g_ref)
    np.testing.assert_array_equal(got, want)
    assert torch.equal(g_device.get_state(), g_ref.get_state())
