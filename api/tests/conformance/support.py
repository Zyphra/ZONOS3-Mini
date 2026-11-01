"""Shared helpers of the conformance suite: the spec, schema checks, capability gates and request helpers."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from jsonschema import Draft7Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

REPO = Path(__file__).resolve().parents[3]
SPEC = json.loads((REPO / "api" / "openapi.json").read_text())
_SPEC_URI = "urn:zonos3:openapi"
_REGISTRY = Registry().with_resource(_SPEC_URI, Resource.from_contents(SPEC, default_specification=DRAFT7))

# Short utterances keep the suite fast; a frame is 2940 samples (44100 Hz / 15 fps).
TEXT = "Hello there, this is a short test."
MAX_TOKENS = 90
SAMPLES_PER_FRAME = 2940
# Request fields of the legacy checkpoints that v2 dropped.
LEGACY_FIELDS = {
    "prepend_silence": True,
    "speaking_rate_enabled": True,
    "speed": 1.0,
    "speaking_rate": 15.0,
    "speaking_rate_bucket": 3,
    "quality_enabled": True,
    "quality_buckets": {"trailing_silence_s": 3},
    "quality_values": {"trailing_silence_s": 0.3},
    "clean_speaker_background": True,
    "accurate_mode": True,
    "speaker_wav_base64": "UklGRg==",
}


def check(obj, ref: str) -> None:
    """Validate ``obj`` against a schema of the spec, e.g. ``"#/components/schemas/Capabilities"``."""
    validator = Draft7Validator({"$ref": _SPEC_URI + ref}, registry=_REGISTRY)
    errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.absolute_path))
    assert not errors, "\n".join(f"{list(e.absolute_path)}: {e.message}" for e in errors[:10])


def needs(cond, reason: str) -> None:
    """Skip the calling test unless ``cond`` holds (a capability of the server under test)."""
    if not cond:
        pytest.skip(reason)


def generate(client: httpx.Client, body: dict, headers: dict | None = None, path: str = "/v2/generate") -> httpx.Response:
    """POST a generation with the suite's short defaults (stream false, max_tokens, seed 1) under ``body``."""
    payload = {"max_tokens": MAX_TOKENS, "seed": 1, **body}
    if path == "/v2/generate":
        payload.setdefault("text", TEXT)
        payload.setdefault("stream", False)
    return client.post(path, json=payload, headers=headers or {})


def pcm(response: httpx.Response) -> np.ndarray:
    """The float32 samples of a 200 pcm response (finite, whole samples)."""
    assert response.status_code == 200, response.text[:500]
    assert len(response.content) % 4 == 0
    audio = np.frombuffer(response.content, dtype="<f4")
    assert audio.size > 0 and np.isfinite(audio).all()
    return audio


def error(response: httpx.Response, status: int) -> dict:
    """The body of an expected error response, checked against Error (400/404/...) or ValidationError (422)."""
    assert response.status_code == status, f"expected {status}, got {response.status_code}: {response.text[:500]}"
    body = response.json()
    check(body, "#/components/schemas/ValidationError" if status == 422 else "#/components/schemas/Error")
    return body
