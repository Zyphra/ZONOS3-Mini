"""GET /v2/controls and steering controls on /v2/generate."""

from __future__ import annotations

import pytest

from .support import TEXT, check, error, generate, needs, pcm


@pytest.fixture(scope="module")
def controls(client):
    r = client.get("/v2/controls")
    if r.status_code == 404:
        return None
    assert r.status_code == 200, r.text[:300]
    return r.json()


def test_document(controls):
    needs(controls is not None, "the server has no steering bundle")
    check(controls, "#/components/schemas/ControlsDocument")


def test_raw_hidden_and_span(client, caps, controls, default_voice):
    needs(controls is not None, "the server has no steering bundle")
    raw = [d for d, v in controls["directions"].items() if "vector" in v]
    hidden = [d for d, v in controls["directions"].items() if "vector" not in v]
    needs(hidden, "the bundle has no hidden-space direction")
    body = {"controls": {hidden[0]: 0.5}, "control_spans": [{"start": 0, "end": 5, "controls": {hidden[0]: 1.0}}]}
    if raw and "embedding" in caps["speaker_modes"]:
        body["controls"][raw[0]] = 0.5
        body.update(speaker_mode="embedding", speaker_embedding_id=default_voice["id"])
    r = generate(client, body)
    pcm(r)
    assert r.headers["x-control-span-follow"] == "rate"


def test_controls_cfg(client, caps, controls, default_voice):
    needs(controls is not None, "the server has no steering bundle")
    needs("embedding" in caps["speaker_modes"], "controls CFG needs speaker_mode=embedding")
    hidden = [d for d, v in controls["directions"].items() if "vector" not in v]
    needs(hidden, "the bundle has no hidden-space direction")
    pcm(
        generate(
            client,
            {
                "speaker_mode": "embedding",
                "speaker_embedding_id": default_voice["id"],
                "controls": {hidden[0]: 0.5},
                "controls_cfg_scale": 1.5,
            },
        )
    )


def test_disabled_controls_are_calibrated_directions(client, controls):
    speakers = client.get("/v2/speakers").json()["speakers"]
    for speaker in speakers:
        for cid in speaker["disabled_controls"]:
            assert controls is not None, f"{speaker['id']}: disabled_controls without a steering bundle"
            did = cid.removesuffix("-")
            assert did in controls["directions"], f"{speaker['id']}: unknown control {cid!r}"
            assert "calibration" in controls["sets"][controls["directions"][did]["set"]], cid


def test_unknown_control_is_400(client, controls):
    needs(controls is not None, "the server has no steering bundle")
    error(generate(client, {"controls": {"nope.nothing": 1.0}}), 400)


def test_without_bundle(client, controls):
    needs(controls is None, "the server has a steering bundle")
    assert client.get("/v2/controls").status_code == 404
    error(generate(client, {"controls": {"style.tempo": 1.0}}), 400)


def test_span_outside_the_text_is_400(client, controls):
    needs(controls is not None, "the server has no steering bundle")
    hidden = [d for d, v in controls["directions"].items() if "vector" not in v]
    needs(hidden, "the bundle has no hidden-space direction")
    error(generate(client, {"control_spans": [{"start": 0, "end": len(TEXT) + 50, "controls": {hidden[0]: 1.0}}]}), 400)
