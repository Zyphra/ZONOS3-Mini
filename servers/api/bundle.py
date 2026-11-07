"""Write openapi.json, the bundled (anchor-free) form of openapi.yaml.

uv run --package zonos3-api python servers/api/bundle.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def bundle() -> str:
    """The JSON text of openapi.yaml: YAML anchors and merge keys resolved, two-space indent, trailing newline."""
    return json.dumps(yaml.safe_load((HERE / "openapi.yaml").read_text()), indent=2) + "\n"


if __name__ == "__main__":
    (HERE / "openapi.json").write_text(bundle())
