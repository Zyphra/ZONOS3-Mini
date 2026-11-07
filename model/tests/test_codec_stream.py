"""The stateful ISTFTStream decode equals the whole ISTFTCodec decode (tiny random model, CPU fp32)."""

import numpy as np
import pytest
import torch
from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec
from zonos3.codec.istft_head_a_20261002.stream import ISTFTStream
from zonos3.codec.stream_host import ResamplerStream

FRAMES = 40
HOP = 2940


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)

    def moss(inp, out, ctx, rate):
        return {
            "type": "MossProjectedTransformer",
            "input_dimension": inp,
            "output_dimension": out,
            "d_model": 32,
            "num_heads": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "context_frames": ctx,
            "frame_rate": rate,
            "layer_scale": 1.0,
            "attention_dropout": 0.0,
            "ffn_dropout": 0.0,
        }

    def up(size):
        return {"type": "PatchedPretransform", "patch_size": size, "direction": "up"}

    stages = [
        moss(1024, 64, 6, 15),
        up(2),
        moss(32, 64, 12, 30),
        up(2),
        moss(32, 64, 24, 60),
        up(2),
        moss(32, 64, 48, 120),
        up(2),
        moss(32, 0, 96, 240),
        up(23),
    ]
    cfg = {
        "decoder_stages": stages,
        "encoder_stages": [],
        "dual_encoding": {"width": 32, "hidden": 64, "depth": 4},
        "residual_codebooks": 9,
        "residual_codebook_size": 4096,
        "semantic_codebook_size": 4096,
        "codebook_dim": 8,
        "quantizer_dropout": 0.0,
        "checkpoint_transformers": False,
        "internal_sample_rate": 44160,
        "frame_rate": 15,
        "internal_hop_length": 2944,
        "external_samples_per_frame": HOP,
    }
    m = ISTFTCodec(cfg, {"type": "mr_istft", "depth": 2, "hidden": 64, "n_ffts": [736, 2208], "fir_kernel": 127}).eval()
    with torch.no_grad():
        m.head.mixer.weight.normal_(0, 0.05)  # both branches through every tap (the init is a delta on branch 0)
    return m


@pytest.fixture(scope="module")
def codes():
    return np.random.default_rng(0).integers(0, 4096, (FRAMES, 10))


@pytest.fixture(scope="module")
def whole(model, codes):
    with torch.inference_mode():
        return model(torch.as_tensor(codes.T[None], dtype=torch.int64))[0, 0].numpy()


@pytest.mark.parametrize("sizes", [[1, 2, 4, 8, 15, 10], [40]])
def test_stream_equals_whole_decode(model, codes, whole, sizes):
    stream, parts, at = ISTFTStream(model), [], 0
    for n in sizes:
        parts.append(stream.decode(codes[at : at + n]))
        at += n
    out = np.concatenate(parts + [stream.flush()])
    assert len(out) == len(whole) == FRAMES * HOP
    np.testing.assert_allclose(out, whole, atol=1e-5, rtol=0)


def test_stream_emits_only_final_samples(model, codes, whole):
    stream, out = ISTFTStream(model), np.zeros(0, np.float32)
    for f in range(1, 12):
        out = np.concatenate([out, stream.decode(codes[f - 1 : f])])
        assert len(out) == HOP * f - HOP // 2
        np.testing.assert_allclose(out, whole[: len(out)], atol=1e-5, rtol=0)


def test_resampler_stream_equals_torchaudio(model):
    x = np.random.default_rng(1).standard_normal(20000).astype(np.float32)
    rs = model.to_external
    stream = ResamplerStream(rs.kernel[:, 0].numpy(), rs.orig_freq // rs.gcd, rs.new_freq // rs.gcd, rs.width)
    parts, at = [], 0
    for n in (100, 2944, 5000, len(x)):
        parts.append(stream.push(x[at : at + n]))
        at += n
    out = np.concatenate(parts + [stream.flush()])
    with torch.inference_mode():
        ref = rs(torch.from_numpy(x)).numpy()
    assert len(out) == len(ref)
    np.testing.assert_allclose(out, ref, atol=1e-6, rtol=0)
