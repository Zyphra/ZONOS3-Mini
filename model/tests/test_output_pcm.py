"""The served PCM's output stage (zonos3.tts.audio.output_pcm): the decoder's gain, then the soft limiter above the
generation registry's output_limit_dbfs knee."""

import numpy as np
from zonos3.generation import OUTPUT_LIMIT_DBFS
from zonos3.tts.audio import OUTPUT_LIMIT_KNEE, output_pcm


def test_knee_is_the_registry_level():
    assert abs(20 * np.log10(float(OUTPUT_LIMIT_KNEE)) - OUTPUT_LIMIT_DBFS) < 1e-5


def test_samples_at_or_below_the_knee_are_only_gained():
    x = np.random.default_rng(0).uniform(-1.2, 1.2, 4096).astype(np.float32)
    gain = 0.7
    quiet = np.abs(x * np.float32(gain)) <= OUTPUT_LIMIT_KNEE
    y = output_pcm(x, gain)
    assert y.dtype == np.float32
    gained = (x * np.float32(gain))[quiet]
    assert (y[quiet] == gained).all()
    assert np.array_equal(output_pcm(gained, 1.0), gained)


def test_peaks_bend_into_full_scale_monotonically_and_symmetrically():
    x = np.linspace(float(OUTPUT_LIMIT_KNEE), 8.0, 2000, dtype=np.float32)
    y = output_pcm(x, 1.0)
    assert (np.diff(y) >= 0).all() and (y <= 1.0).all() and (y <= x).all()
    assert output_pcm(np.array([1.2], np.float32), 1.0)[0] < 1.0  # a typical overshoot stays inside full scale
    assert np.array_equal(output_pcm(-x, 1.0), -y)
    # continuous with slope 1 at the knee
    eps = np.float32(1e-4)
    assert (
        abs(float(output_pcm(np.array([OUTPUT_LIMIT_KNEE + eps], np.float32), 1.0)[0]) - float(OUTPUT_LIMIT_KNEE + eps))
        < 1e-6
    )


def test_chunks_equal_the_whole():
    x = np.random.default_rng(1).normal(0, 0.6, 10000).astype(np.float32)
    whole = output_pcm(x, 0.8)
    parts = np.concatenate([output_pcm(x[i : i + 777], 0.8) for i in range(0, len(x), 777)])
    assert np.array_equal(whole, parts)
