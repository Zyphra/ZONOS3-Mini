"""Offline checks of the API spec: the committed bundle, OpenAPI validity, strict request bodies."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from openapi_spec_validator import validate

API = Path(__file__).resolve().parents[1]


def _bundle_module():
    spec = importlib.util.spec_from_file_location("zonos3_api_bundle", API / "bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spec() -> dict:
    return json.loads((API / "openapi.json").read_text())


def test_openapi_json_is_the_bundled_yaml():
    assert (API / "openapi.json").read_text() == _bundle_module().bundle(), (
        "api/openapi.json is stale: run `uv run --package zonos3-api python api/bundle.py`"
    )


def test_spec_is_valid_openapi():
    validate(_spec())


def _resolve(spec: dict, schema: dict) -> dict:
    while "$ref" in schema:
        node = spec
        for part in schema["$ref"].removeprefix("#/").split("/"):
            node = node[part]
        schema = node
    return schema


def test_every_object_request_body_forbids_unknown_fields():
    spec = _spec()
    loose = []
    for path, item in spec["paths"].items():
        for method, op in item.items():
            if not isinstance(op, dict) or "requestBody" not in op:
                continue
            for media, content in op["requestBody"]["content"].items():
                schema = _resolve(spec, content["schema"])
                if schema.get("type") == "object" and schema.get("additionalProperties") is not False:
                    loose.append(f"{method.upper()} {path} ({media})")
    assert not loose, f"request bodies without additionalProperties: false: {loose}"


def test_paths_are_v1_or_v2():
    assert all(p.startswith(("/v1/", "/v2/")) for p in _spec()["paths"])
