"""ffmpeg PCM decode: every encoder input goes through here, so the encoders only see 44100 Hz mono float32."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np

_ERROR = "Reference file must contain an audio stream supported by ffmpeg."


def ffmpeg_binary() -> str:
    path = os.environ.get("ZONOS3_FFMPEG") or shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found: install it or set ZONOS3_FFMPEG")
    return path


def decode_pcm(audio_bytes: bytes, sample_rate: int = 44100, max_seconds: float | None = None) -> np.ndarray:
    """Mono float32 PCM at ``sample_rate`` (ggml audio-decode.cpp's ffmpeg command). The bytes go through a temporary
    file, not a pipe: ffmpeg applies an MP3's gapless trim (encoder delay / padding) only to a seekable input."""
    with tempfile.NamedTemporaryFile(prefix="zonos3-audio-", suffix=".bin") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        cmd = [ffmpeg_binary(), "-nostdin", "-v", "error", "-i", tmp.name, "-ac", "1", "-ar", str(int(sample_rate))]
        if max_seconds is not None:
            cmd += ["-t", f"{float(max_seconds):g}"]
        cmd += ["-f", "f32le", "-"]
        proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise ValueError(_ERROR + (f" ({stderr})" if stderr else ""))
    pcm = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if not len(pcm):
        raise ValueError(_ERROR)
    return pcm


def polyphase_resample(x: np.ndarray, kernel: np.ndarray, orig: int, new: int, width: int) -> np.ndarray:
    """torchaudio's polyphase sinc resampler with its ``[new, 2*width + orig]`` kernel (the GGUFs store it):
    ``out[n*new + i] = sum_k K[i][k] * x[n*orig + k - width]``, zero outside, ``ceil(new * len / orig)`` samples
    (ggml resample-16k.h ``run``, dac.cpp / dac-encoder.cpp ``resample``). Accumulates in float64."""
    x = np.asarray(x, dtype=np.float64)
    L = len(x)
    total = -(-new * L // orig)
    if total == 0:
        return np.zeros(0, dtype=np.float32)
    blocks = -(-total // new)
    taps = 2 * width + orig
    padded = np.zeros(width + blocks * orig + taps, dtype=np.float64)
    padded[width : width + L] = x
    windows = np.lib.stride_tricks.sliding_window_view(padded, taps)[: blocks * orig : orig]
    out = windows @ np.asarray(kernel, dtype=np.float64).T
    return out.reshape(-1)[:total].astype(np.float32)
