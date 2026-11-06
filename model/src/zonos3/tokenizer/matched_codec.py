"""Decode the frozen codec used to tokenize each matched LM's training data.

Decoding goes through the causal multi-resolution iSTFT head declared by the registry entry's ``decoder`` block: the
KEEP-2 pruned head (``codec_registry_keep2_plain.json``, istft_head_a_20261002 full-unfreezeall-mr-drop30-4x10).
The 600k conv-tail decoder is never used. The 600k generator.pt only supplies the frozen encoder/quantizer and the
initial decoder stages that the head checkpoint then overrides."""

import functools
import json
import os
import time
from pathlib import Path

import torch

from zonos3.utils.hashing import sha256_file
from zonos3.utils.release import hf_join, resolve_file

REGISTRY = Path(__file__).resolve().parents[1] / "registries" / "codec_registry_keep2_plain.json"
_CACHE = {}
_INFO = {}


@functools.cache
def registry():
    return json.loads(REGISTRY.read_text())


def codec_names():
    return {v["codec"] for v in registry().values()}


def entry(codec):
    values = [v for v in registry().values() if v["codec"] == codec]
    if len(values) != 1:
        raise ValueError(f"Unknown matched codec {codec}")
    return values[0]


def decoder_spec(codec):
    """The registry's decoder block; every matched codec must declare a causal iSTFT decoder (no conv-tail fallback)."""
    e = entry(codec)
    spec = e.get("decoder")
    if not spec or spec.get("kind") != "istft_mr":
        raise RuntimeError(f"matched codec {codec!r} declares no iSTFT decoder; the conv-tail decoder is disabled")
    return spec


def decoder_headroom_db(codec):
    return float(decoder_spec(codec).get("headroom_db", 0.0))


def decoder_info(codec):
    spec = decoder_spec(codec)
    info = {
        "kind": spec["kind"],
        "label": spec.get("label"),
        "checkpoint": hf_join(spec["campaign"], spec["checkpoint"]),
        "expect_step": spec.get("expect_step"),
        "headroom_db": float(spec.get("headroom_db", 0.0)),
        "causal": True,
    }
    info.update(_INFO.get(codec, {}))
    return info


def load_model(codec, device, prune=True):
    """Build the iSTFT-head decoder exactly as the listening viewer's decode.py does: ISTFTCodec(cfg['model'], cfg['head']),
    pretrained generator weights, then the run's trainable state (decoder stages + head) from the pinned checkpoint."""
    e = entry(codec)
    spec = decoder_spec(codec)
    from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec

    campaign = spec["campaign"]
    cfg = json.loads(Path(resolve_file(hf_join(campaign, spec["config"]))).read_text())
    assert cfg.get("decoder", {}).get("type") != "original", (
        "the registry points at a conv-tail (original decoder) config"
    )
    generator_path = Path(resolve_file(cfg["init"]["generator_checkpoint"]))
    assert str(generator_path) == resolve_file(e["checkpoint"]), (str(generator_path), e["checkpoint"])
    generator_state = torch.load(generator_path, map_location="cpu", mmap=True, weights_only=True)
    assert generator_state["step"] == e["codec_step"] and generator_state["config"]["model"] == cfg["model"]
    model = ISTFTCodec(cfg["model"], cfg["head"])
    init = model.load_pretrained(generator_state["generator"])
    checkpoint = Path(resolve_file(hf_join(campaign, spec["checkpoint"])))
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if spec.get("expect_step") is not None:
        assert int(state["step"]) == int(spec["expect_step"]), (
            str(checkpoint),
            int(state["step"]),
            spec["expect_step"],
        )
    if "trainable" in state:
        model.load_trainable_state_dict(state["trainable"])
    else:
        assert not model.trainable_stages
        model.head.load_state_dict(state["head"], strict=True)
    model.eval().requires_grad_(False)
    if prune:
        prune_model(model, e)
    _INFO[codec] = {
        "step": int(state["step"]),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(checkpoint.stat().st_mtime)),
        "model": type(model).__name__,
        "init": init if isinstance(init, dict) else None,
    }
    del state, generator_state
    return model.to(device)


def prune_model(model, e):
    # ISTFTCodec never holds the encoder (its __init__ keeps only stages/quantizer/semantic_head/to_external of the base codec);
    # semantic_head is used by features() and must stay.
    for name in ("encoder", "to_internal", "semantic_encoder"):
        if hasattr(model, name):
            try:
                delattr(model, name)
            except Exception:
                pass


def decode_model(model, codes, e):
    """[B, frames, codebooks] int -> float32 [B, samples]; fp32 with TF32 off, like the viewers' decode service."""
    device = next(model.parameters()).device
    packed = codes.permute(0, 2, 1).to(device=device, dtype=torch.int64)
    expected = packed.shape[-1] * e["hop"]
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.inference_mode():
            audio = model(packed)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
    if audio.shape != (packed.shape[0], 1, expected):
        raise RuntimeError(f"Codec output shape {tuple(audio.shape)}; expected {expected} samples")
    if not torch.isfinite(audio).all():
        raise RuntimeError("Nonfinite codec audio")
    return audio.float().squeeze(1).cpu()


def decode(codes, codec):
    e = entry(codec)
    if codes.ndim != 3 or codes.shape[-1] != e["codebooks"] or bool(((codes < 0) | (codes >= 4096)).any()):
        raise ValueError("Invalid matched codec tokens")
    device = os.environ.get("ZONOS3_AUDIO_CODEC_DEVICE", "cuda:0")
    key = (codec, device)
    if key not in _CACHE:
        actual = sha256_file(resolve_file(e["checkpoint"]))
        assert actual == e["checkpoint_sha256"], ("generator checkpoint sha256", actual, e["checkpoint_sha256"])
        _CACHE[key] = load_model(codec, device)
    return decode_model(_CACHE[key], codes, e)
