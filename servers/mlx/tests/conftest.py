from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

# Apple Silicon only: the whole directory is skipped elsewhere (a module-level importorskip in a conftest aborts a
# root `pytest` run, which loads this file at startup through testpaths).
if importlib.util.find_spec("mlx") is None:
    collect_ignore_glob = ["test_*.py"]

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "servers" / "ggml" / "tests" / "fixtures"
sys.path.insert(0, str(REPO / "tools"))  # convert_gguf (tiny GGUFs), as servers/ggml/tests/parity does


def dummy_codec(dst: Path, residual_biases: bool) -> Path:
    """The dummy decoder fixture at ``dst``, with or without the per-codebook residual biases ``res.{i}.bias`` (seeded,
    adding up to its ``res.bias``) that early partial-frame decode needs."""
    from convert_gguf import write_gguf
    from gguf import GGUFReader, GGUFValueType, GGUFWriter

    r = GGUFReader(str(FIXTURES / "dummy-zac.gguf"))
    w = GGUFWriter(str(dst), r.fields["general.architecture"].contents())
    for name, f in r.fields.items():
        if not name.startswith("GGUF.") and name != "general.architecture":
            sub = f.types[-1] if f.types[0] == GGUFValueType.ARRAY else None
            w.add_key_value(name, f.contents(), f.types[0], sub)
    tensors = {
        t.name: np.asarray(t.data).reshape(tuple(reversed(t.shape.tolist())))
        for t in r.tensors
        if not re.fullmatch(r"res\.\d+\.bias", t.name)
    }
    for name, arr in tensors.items():
        w.add_tensor(name, np.ascontiguousarray(arr))
    if residual_biases:
        total = tensors["res.bias"].astype(np.float64)
        biases = np.random.default_rng(1).standard_normal((9, total.size)) * 0.5
        biases[-1] = total - biases[:-1].sum(0)
        for i, b in enumerate(biases):
            w.add_tensor(f"res.{i}.bias", b.astype(np.float32))
    write_gguf(w)
    return dst
