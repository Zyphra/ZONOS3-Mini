"""Every advertised stream_decode mode returns audio."""

from __future__ import annotations

import numpy as np
import pytest

from .support import MAX_TOKENS, TEXT


@pytest.mark.parametrize("mode", ["prefix", "whole"])
def test_mode(client, caps, mode):
    if mode not in caps["stream_decode_modes"]:
        pytest.skip(f"stream_decode={mode} is not offered")
    body = {"text": TEXT, "stream": True, "stream_decode": mode, "max_tokens": MAX_TOKENS, "seed": 3}
    with client.stream("POST", "/v2/generate", json=body) as r:
        assert r.status_code == 200
        data = b"".join(r.iter_bytes())
    assert len(data) % 4 == 0 and len(data) > 0
    assert np.isfinite(np.frombuffer(data, dtype="<f4")).all()
