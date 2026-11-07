"""The speaker encoder registry + the vendored MECT-B2 encoder (ownmect_s2) behind speaker_mode=embedding."""

import json
from pathlib import Path

import pytest
import torch

sc = pytest.importorskip("zonos3.models.speaker_cloning")
ms = pytest.importorskip("zonos3.models.mect_speaker")
from zonos3.utils.release import resolve_file  # noqa: E402

ENTRY = {
    "type": "mect_fbank",
    "backbone_type": "MECT_B2",
    "backbone_config": {
        "moe_type": "token",
        "num_experts_shard": 2,
        "num_experts_token": 4,
        "top_k_shard": None,
        "top_k_token": None,
        "shard_scale": 1.0,
    },
    "pooling": "ASTP",
    "embed_dim": 192,
    "sample_rate": 16000,
    "max_seconds": 60,
    "fbank": {
        "num_mel_bins": 80,
        "frame_length": 25,
        "frame_shift": 10,
        "dither": 0.0,
        "energy_floor": 0.0,
        "window_type": "hamming",
        "scale": 32768,
    },
}


def _registry(tmp_path, monkeypatch, **entries):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(entries))
    monkeypatch.setenv(ms.REGISTRY_ENV, str(path))
    return path


def test_builtin_aliases_and_unknown_codecs_are_unchanged(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch)
    assert sc.resolve_speaker_embedding_backend("w2vbert_sv", 256) == "w2vbert_sv"
    assert sc.resolve_speaker_embedding_backend("qwen3", 2048) == "qwen3"
    assert sc.resolve_speaker_embedding_backend(None, 256) == "w2vbert_sv"
    with pytest.raises(ValueError):
        sc.resolve_speaker_embedding_backend("ownmect_s2", 192)


def test_registry_codecs_resolve_to_their_own_backend_and_check_the_dimension(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, ownmect_s2=dict(ENTRY, checkpoint=str(tmp_path / "missing.pth")))
    assert sc.resolve_speaker_embedding_backend("ownmect_s2", 192) == "ownmect_s2"
    with pytest.raises(ValueError):
        sc.resolve_speaker_embedding_backend("ownmect_s2", 256)
    with pytest.raises(FileNotFoundError):
        sc.create_speaker_embedder("ownmect_s2", 192, device="cpu")


def test_vendored_mect_b2_builds_and_embeds_on_cpu(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, ownmect_s2=dict(ENTRY))  # no checkpoint: random weights, architecture only
    torch.manual_seed(0)
    embedder = sc.create_speaker_embedder("ownmect_s2", 192, device="cpu")
    assert isinstance(embedder, ms.MECTFbankSpeakerEmbedding)
    assert embedder.fbank(torch.zeros(16000) + 1e-3).shape == (98, 80)  # 1 s -> 98 frames of 80 mel bins
    wav = torch.randn(2, 24_000) * 0.1  # stereo, 24 kHz: averaged to mono and resampled to 16 kHz
    with torch.inference_mode():
        a = embedder(wav, 24_000)
        b = embedder(wav, 24_000)
    assert a.shape == (1, 192) and a.dtype == torch.float32
    assert torch.equal(a, b)
    keys = set(embedder.state_dict())
    assert "model.linear.weight" in keys and "model.bn.running_mean" in keys  # George's MECTFbank key layout


def test_sha256_mismatch_is_refused(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt.pth"
    torch.save({"modules": {"spk_model": {}}}, ckpt)
    _registry(tmp_path, monkeypatch, ownmect_s2=dict(ENTRY, checkpoint=str(ckpt), sha256="00" * 32))
    with pytest.raises(ValueError):
        sc.create_speaker_embedder("ownmect_s2", 192, device="cpu")


@pytest.mark.skipif(not ms.DEFAULT_REGISTRY.is_file(), reason="no default speaker encoder registry")
def test_default_registry_checkpoint_loads_strictly():
    entry = ms.registry_entry("ownmect_s2")
    try:
        available = entry is not None and Path(resolve_file(entry.get("checkpoint", ""))).is_file()
    except Exception:  # noqa: BLE001  (no HF access)
        available = False
    if not available:
        pytest.skip("ownmect_s2 checkpoint not available here")
    embedder = ms.MECTFbankSpeakerEmbedding(entry, device="cpu")
    torch.manual_seed(1)
    with torch.inference_mode():
        e = embedder(torch.randn(16000 * 2) * 0.05, 16000)
    assert e.shape == (1, 192) and torch.isfinite(e).all()
