"""Parity of the ggml server's native ports against the Python reference (model/src/zonos3):

    uv run --package zonos3-tools pytest servers/ggml/tests/parity --ggml-build servers/ggml/build

Every test is skipped without --ggml-build (the CMake build directory holding the test binaries).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]


def pytest_addoption(parser):
    parser.addoption("--ggml-build", default=None, help="servers/ggml CMake build directory")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ggml-build"):
        return
    skip = pytest.mark.skip(reason="pass --ggml-build <servers/ggml build dir>")
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def ggml_build(pytestconfig) -> Path:
    return Path(pytestconfig.getoption("--ggml-build")).resolve()


@pytest.fixture(scope="session")
def tool(ggml_build):
    """``tool(name, payload)``: run a build-tree binary with a JSON document on stdin; returns its stdout JSON."""

    def run(name: str, payload, *args: str):
        exe = ggml_build / name
        if not exe.exists():
            pytest.fail(f"{exe} is not built")
        out = subprocess.run([str(exe), *args], input=json.dumps(payload), capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    return run
