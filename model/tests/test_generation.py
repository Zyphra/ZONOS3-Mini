"""The generation registry (zonos3.generation, registries/generation.json) against servers/api/openapi.json: the spec repeats
the sampling defaults and request limits, the registry is what every server reads."""

import json
from pathlib import Path

import pytest
from zonos3 import generation

SCHEMAS = json.loads((Path(__file__).resolve().parents[2] / "servers" / "api" / "openapi.json").read_text())[
    "components"
]["schemas"]


@pytest.mark.parametrize("schema", ["GenerationOptions", "GenerateRequest", "SpeechRequest", "EditRequest"])
def test_openapi_sampling_defaults_are_the_registry(schema):
    properties = SCHEMAS[schema]["properties"]
    assert {name: properties[name]["default"] for name in generation.SAMPLING} == dict(generation.SAMPLING)


@pytest.mark.parametrize("schema", ["GenerationOptions", "GenerateRequest", "SpeechRequest"])
def test_openapi_request_limits_are_the_registry(schema):
    properties = SCHEMAS[schema]["properties"]
    tag_cfg = properties["tag_cfg_scale"]
    assert (tag_cfg["minimum"], tag_cfg["maximum"]) == (generation.TAG_CFG_MIN_SCALE, generation.TAG_CFG_MAX_SCALE)
    assert properties["decode_context_frames"]["default"] == generation.DECODE_CONTEXT_FRAMES
    assert properties["stream_codebooks"]["default"] == generation.STREAM_CODEBOOKS
    assert properties["stream_codebooks_scope"]["default"] == generation.STREAM_CODEBOOKS_SCOPE
