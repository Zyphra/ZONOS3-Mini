"""Conformance suite of the ZONOS3 v2 HTTP API, run against a live server:

    uv run --package zonos3-api pytest servers/api/tests/conformance --base-url http://127.0.0.1:1919 [--api-token T]

Every test is skipped without --base-url (or ZONOS3_API_URL); feature tests skip when the server's capabilities say
the feature is absent.
"""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path

import httpx
import pytest

from .support import REPO, check, needs


def pytest_addoption(parser):
    parser.addoption("--base-url", default=os.environ.get("ZONOS3_API_URL"), help="server under test")
    parser.addoption(
        "--api-token", default=os.environ.get("ZONOS3_API_TOKEN"), help="ZONOS3_ACCESS_TOKEN of the server"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--base-url"):
        return
    # the hook sees every collected item of the session: skip only this suite's live-server tests
    here = Path(__file__).resolve().parent
    skip = pytest.mark.skip(reason="set --base-url or ZONOS3_API_URL")
    for item in items:
        if here in Path(str(item.path)).resolve().parents:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def base_url(pytestconfig) -> str:
    return pytestconfig.getoption("--base-url").rstrip("/")


@pytest.fixture(scope="session")
def api_token(pytestconfig) -> str | None:
    return pytestconfig.getoption("--api-token")


@pytest.fixture(scope="session")
def client(base_url, api_token):
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    with httpx.Client(base_url=base_url, timeout=900.0, headers=headers) as c:
        yield c


@pytest.fixture
def validate():
    """``validate(obj, "#/components/schemas/X")``: jsonschema Draft 7 against the spec."""
    return check


@pytest.fixture(scope="session")
def caps(client) -> dict:
    r = client.get("/v2/capabilities")
    assert r.status_code == 200, r.text
    body = r.json()
    check(body, "#/components/schemas/Capabilities")
    return body


@pytest.fixture
def session() -> dict:
    return {"X-TTS-Session-ID": f"conformance-{uuid.uuid4().hex}"}


@pytest.fixture(scope="session")
def voice_bytes() -> bytes:
    return (REPO / "assets" / "default_voices" / "AmericanFemale.mp3").read_bytes()


@pytest.fixture(scope="session")
def voice_b64(voice_bytes) -> str:
    return base64.b64encode(voice_bytes).decode("ascii")


@pytest.fixture(scope="session")
def speakers(client) -> list[dict]:
    r = client.get("/v2/speakers")
    assert r.status_code == 200, r.text
    check(r.json(), "#/components/schemas/SpeakerList")
    return r.json()["speakers"]


@pytest.fixture(scope="session")
def default_voice(speakers) -> dict:
    """A default voice (an audio one when there is any)."""
    defaults = [s for s in speakers if s["is_default"]]
    needs(defaults, "the server has no default voices")
    audio = [s for s in defaults if s["source_type"] == "audio"]
    return (audio or defaults)[0]
