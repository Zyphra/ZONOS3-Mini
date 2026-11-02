from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")  # Apple Silicon only: the whole directory skips elsewhere

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "servers" / "ggml" / "tests" / "fixtures"
sys.path.insert(0, str(REPO / "tools"))  # convert_gguf (tiny GGUFs), as servers/ggml/tests/parity does
