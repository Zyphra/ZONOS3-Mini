from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import dequantize, quantize
from zonos3_mlx.gguf_io import GGUFFile, matmul

SHAPES = {"a": (64, 32), "b": (4, 64, 96)}
QTYPES = (GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0)


@pytest.fixture(scope="module")
def quantized(tmp_path_factory):
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("gguf") / "q.gguf"
    writer = GGUFWriter(str(path), "test")
    raw = {}
    for qtype in QTYPES:
        for key, shape in SHAPES.items():
            name = f"{key}.{qtype.name}"
            data = rng.standard_normal(shape).astype(np.float32)
            raw[name] = (quantize(data, qtype), qtype, shape)
            writer.add_tensor(name, raw[name][0], raw_shape=raw[name][0].shape, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return GGUFFile(path), raw


@pytest.mark.parametrize("qtype", QTYPES, ids=lambda q: q.name)
@pytest.mark.parametrize("key", list(SHAPES))
def test_repacked_blocks_dequantize_exactly(quantized, qtype, key):
    f, raw = quantized
    blocks, _, shape = raw[f"{key}.{qtype.name}"]
    W = f.weight(f"{key}.{qtype.name}")
    assert W.bits == (4 if qtype == GGMLQuantizationType.Q4_0 else 8)
    got = np.array(mx.dequantize(W.w, W.scales, W.biases, group_size=32, bits=W.bits))
    want = dequantize(blocks, qtype).reshape(shape)
    np.testing.assert_array_equal(got, want)

    lead = shape[:-2]  # stacked (expert) weights take a batched x
    x = np.random.default_rng(1).standard_normal((*lead, 5, shape[-1])).astype(np.float32)
    y = np.array(matmul(mx.array(x), W))
    np.testing.assert_allclose(y, x @ np.swapaxes(want, -1, -2), atol=1e-5, rtol=1e-5)
