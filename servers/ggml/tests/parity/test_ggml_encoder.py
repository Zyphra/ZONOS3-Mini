"""Reference audio longer than 60 seconds through the native codec encoder (zac-encoder-cli) against the Python
contract, zonos3.tokenizer.dual_reference.encode_dual_reference (the frozen MossAutoencoder on CUDA).

The native encoder runs the same independent 20-second lanes with 10 s of left context as the Python batcher, so a
clip past 60 s must encode (it was capped at 60 s while /v2/edit accepts 150 s) and its frames past 60 s must agree
with the reference as well as earlier frames do. Codes are not bit-exact against the reference even below 60 s
(docs/INTERNALS.md "Fidelity limits": BF16 rounding decisions of cuDNN / fused SDPA flip RVQ boundary choices; about
82% of all codes and 97% of codebook 0 agree on the default voices), so the gate is agreement, measured separately
before and after the 60 s mark.

Inputs (skipped when absent): the release GGUFs ``ZONOS3_ZAC_ENCODER`` / ``ZONOS3_SSL_TEACHER`` (default
servers/ggml/models/{zac-encoder,ssl-teacher}.gguf) and CUDA for the reference. The build's encoder runs on the GPU
when it was configured with GGML_CUDA (a CPU build takes about 4 s per audio second).

    uv run --package zonos3-tools pytest servers/ggml/tests/parity/test_ggml_encoder.py --ggml-build servers/ggml/build-cuda -s
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
MODELS = REPO / "servers" / "ggml" / "models"
ZAC_ENCODER = Path(os.environ.get("ZONOS3_ZAC_ENCODER", MODELS / "zac-encoder.gguf"))
SSL_TEACHER = Path(os.environ.get("ZONOS3_SSL_TEACHER", MODELS / "ssl-teacher.gguf"))
SR, HOP, FRAMES_60S = 44100, 2940, 900
SECONDS = 66.1  # five full lanes and a final lane wholly past 60 s
MAX_SECONDS = 600  # zac-encoder.h ZAC_ENCODER_MAX_SECONDS = prepare_reference's limit


def _cli(ggml_build: Path) -> Path:
    exe = ggml_build / "zac-encoder-cli"
    if not exe.exists():
        pytest.fail(f"{exe} is not built")
    return exe


def _uses_cuda(ggml_build: Path) -> bool:
    cache = ggml_build / "CMakeCache.txt"
    return cache.is_file() and "GGML_CUDA:BOOL=ON" in cache.read_text(errors="replace")


def _clip(seconds: float) -> np.ndarray:
    """The default voices at 44.1 kHz mono, concatenated and repeated to ``seconds``; peak at most 1 (the reference
    normalizes louder input, the native encoder takes it as is)."""
    parts = []
    for mp3 in sorted((REPO / "default_voices").glob("*.mp3")):
        raw = subprocess.check_output(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(mp3), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
        )
        parts.append(np.frombuffer(raw, np.float32))
    one = np.concatenate(parts)
    n = int(round(seconds * SR))
    audio = np.tile(one, n // one.size + 1)[:n].astype(np.float32)
    return audio / max(1.0, float(np.abs(audio).max()))


def test_cli_limit(ggml_build, tmp_path):
    """Up to MAX_SECONDS reaches the models (the --dump audio is written; missing GGUF paths fail at load); one sample
    more is rejected before the dump."""
    exe = _cli(ggml_build)
    for samples, accepted in [(MAX_SECONDS * SR + 1, False), (MAX_SECONDS * SR, True)]:
        np.save(tmp_path / "a.npy", np.zeros(samples, np.float32))
        dump = tmp_path / f"dump{samples}"
        run = subprocess.run(
            [
                str(exe),
                str(tmp_path / "missing.gguf"),
                str(tmp_path / "missing.gguf"),
                "--audio",
                str(tmp_path / "a.npy"),
                str(tmp_path / "codes.npy"),
                "--cpu",
                "--dump",
                str(dump),
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode != 0
        assert (dump / "audio.npy").is_file() == accepted, run.stderr


def test_encode_past_60_seconds(ggml_build, tmp_path):
    if not (ZAC_ENCODER.is_file() and SSL_TEACHER.is_file()):
        pytest.skip(f"no release encoder GGUFs at {ZAC_ENCODER} / {SSL_TEACHER}")
    if not torch.cuda.is_available():
        pytest.skip("the reference encoder needs CUDA")
    audio = _clip(SECONDS)
    np.save(tmp_path / "audio.npy", audio)
    run = subprocess.run(
        [
            str(_cli(ggml_build)),
            str(ZAC_ENCODER),
            str(SSL_TEACHER),
            "--audio",
            str(tmp_path / "audio.npy"),
            str(tmp_path / "codes.npy"),
            "--gpu" if _uses_cuda(ggml_build) else "--cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    native = np.load(tmp_path / "codes.npy")

    from zonos3.tokenizer.dual_reference import encode_dual_reference

    reference = encode_dual_reference(torch.from_numpy(audio), SR, max_seconds=150, device="cuda:0").numpy()
    assert native.shape == reference.shape == (-(-audio.size // HOP), 10)
    equal = native == reference
    before, after = equal[:FRAMES_60S], equal[FRAMES_60S:]
    print(
        f"code agreement: <60 s {before.mean():.4f} (codebook 0 {before[:, 0].mean():.4f}), "
        f">60 s {after.mean():.4f} (codebook 0 {after[:, 0].mean():.4f}); per codebook "
        + " ".join(f"{v:.3f}" for v in equal.mean(0))
    )
    # Per lane the agreement varies with content (0.72..0.89 overall, 0.95..0.99 codebook 0 over the lanes of a 150 s
    # clip of these voices, CPU and CUDA alike); a broken lane plan or truncation agrees near chance.
    for part in (before, after):
        assert part.mean() >= 0.65 and part[:, 0].mean() >= 0.9
