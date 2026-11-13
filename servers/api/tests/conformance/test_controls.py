"""GET /v2/controls and steering controls on /v2/generate."""

from __future__ import annotations

import numpy as np
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



# --------------------------------------------------------------------------------------------- steer then hold


def test_steer_release_is_reported(caps, controls):
    """Capabilities and the controls document report the same steer-then-hold defaults (null without them)."""
    if controls is None:
        assert caps["steer_release"] is None
        return
    assert caps["steer_release"] == controls["steer_release"]
    release = controls["steer_release"]
    if release is not None:
        hidden = {d for d, v in controls["directions"].items() if "vector" not in v}
        assert set(release["exempt"]) <= hidden, "exempt ids must be hidden-space directions"


def _released_control(controls) -> dict:
    """A strong dose of a hidden-space direction the bundle releases."""
    exempt = set((controls["steer_release"] or {}).get("exempt", ()))
    hidden = [d for d, v in controls["directions"].items() if "vector" not in v and d not in exempt]
    needs(hidden, "the bundle has no released hidden-space direction")
    return {hidden[0]: 4.0}


def _deterministic(client, body) -> np.ndarray:
    """The render of ``body``, skipping the test when the server does not repeat a seeded render exactly."""
    first, second = pcm(generate(client, body)), pcm(generate(client, body))
    needs(np.array_equal(first, second), "the server does not repeat a seeded render bit for bit")
    return first


def test_steer_hold_releases_the_controls(client, controls):
    """hold < 1 changes a render once the onset has passed; a render that ends before the onset is the constant push."""
    needs(controls is not None and controls["steer_release"] is not None, "the bundle has no steer_release")
    body = {"controls": _released_control(controls), "max_tokens": 150}
    constant = _deterministic(client, {**body, "steer_hold": 1.0})
    released = pcm(generate(client, {**body, "steer_hold": 0.0, "steer_onset_seconds": 0.5, "steer_ramp_seconds": 0}))
    assert not np.array_equal(constant, released), "steer_hold 0 after 0.5 s left the render unchanged"
    late = pcm(generate(client, {**body, "steer_hold": 0.25, "steer_onset_seconds": 600.0}))
    assert np.array_equal(constant, late), "an onset past the end of the render must keep the constant push"


def test_steer_hold_one_is_the_constant_push(client, controls):
    """Without a bundle steer_release, steer_hold 1 (and onset / ramp) are the request without them; below 1 is a 400."""
    needs(controls is not None and controls["steer_release"] is None, "the bundle has a steer_release")
    hidden = [d for d, v in controls["directions"].items() if "vector" not in v]
    needs(hidden, "the bundle has no hidden-space direction")
    body = {"controls": {hidden[0]: 2.0}}
    plain = _deterministic(client, body)
    held = pcm(generate(client, {**body, "steer_hold": 1.0, "steer_onset_seconds": 1.0, "steer_ramp_seconds": 1.0}))
    assert np.array_equal(plain, held)
    error(generate(client, {**body, "steer_hold": 0.5}), 400)


def test_steer_hold_without_bundle_is_400(client, controls):
    needs(controls is None, "the server has a steering bundle")
    error(generate(client, {"steer_hold": 0.5}), 400)
    pcm(generate(client, {"steer_hold": 1.0}))


@pytest.mark.parametrize(
    "fields",
    [{"steer_hold": 1.5}, {"steer_hold": -0.5}, {"steer_onset_seconds": -1}, {"steer_ramp_seconds": 1000}],
)
def test_steer_release_fields_out_of_range_are_422(client, fields):
    error(generate(client, fields), 422)