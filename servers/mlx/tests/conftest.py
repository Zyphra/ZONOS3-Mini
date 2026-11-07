from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Apple Silicon only: the whole directory is skipped elsewhere (a module-level importorskip in a conftest aborts a
# root `pytest` run, which loads this file at startup through testpaths).
if importlib.util.find_spec("mlx") is None:
    collect_ignore_glob = ["test_*.py"]

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "servers" / "ggml" / "tests" / "fixtures"
sys.path.insert(0, str(REPO / "tools"))  # convert_gguf (tiny GGUFs), as servers/ggml/tests/parity does
