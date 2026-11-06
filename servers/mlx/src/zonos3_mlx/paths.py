"""Checkout paths the server defaults to."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
STUDIO_DIR = REPO_ROOT / "studio" / "dist"
PYTHON_SRC = REPO_ROOT / "model" / "src"
DEFAULT_VOICES = REPO_ROOT / "assets" / "default_voices"
CACHE_DIR = Path(os.environ.get("ZONOS3_MLX_CACHE", "~/.cache/zonos3-mlx")).expanduser()
