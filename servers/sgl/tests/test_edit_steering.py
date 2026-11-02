"""Steering doses of /v2/edit regions: `controls` + `region_controls[k]`, validated before any work."""
from __future__ import annotations

import json

import numpy as np
import pytest

api_server = pytest.importorskip("zonos3_sgl.server.api_server")
from zonos3.server.schemas import EditRequest  # noqa: E402


def _set(root, name, space, vectors, source=None):
    d = root / name
    d.mkdir()
    for k, v in vectors.items():
        np.save(d / f"{k}.npy", np.array(v, dtype=np.float32))
    manifest = {"dim": len(next(iter(vectors.values()))), "space": space,
                "directions": {k: {"file": f"{k}.npy", "kind": "named"} for k in vectors}}
    if source is not None:
        manifest["source_index"] = source
    (d / "manifest.json").write_text(json.dumps(manifest))


class _ModelConfig:
    def __init__(self, speaker_enabled):
        self.speaker_enabled = speaker_enabled


class _Config:
    def __init__(self, bundle, speaker_enabled=True):
        self.steer_bundle = bundle
        self.model_config = _ModelConfig(speaker_enabled)


@pytest.fixture
def config(tmp_path):
    _set(tmp_path, "who", "raw", {"gender": [0.0, 2.0, 0.0]})
    _set(tmp_path, "style", "attnres_source", {"whisper": [1.0, 0.0], "angry": [0.0, 1.0]}, source=19)
    (tmp_path / "bundle.json").write_text(json.dumps({"name": "t", "sets": {"who": "who", "style": "style"}}))
    return _Config(str(tmp_path / "bundle.json"))


def _doses(config, **fields):
    return api_server._edit_region_doses(config, EditRequest(audio_base64="", target_text="x", **fields))


SPANS = [[0.1, 0.4], [1.0, 1.5], [2.0, 2.5]]


def test_region_doses_add_to_the_shared_controls_per_id(config):
    doses = _doses(config, spans=SPANS, controls={"style.whisper": 1, "who.gender": 1},
                   region_controls=[{"style.whisper": 1}, None, {"who.gender": -1, "style.angry": 4}])
    assert doses == [{"style.whisper": 2.0, "who.gender": 1.0}, {"style.whisper": 1.0, "who.gender": 1.0},
                     {"style.whisper": 1.0, "style.angry": 4.0}]
    assert _doses(config, spans=SPANS) is None
    assert _doses(config, spans=SPANS, region_controls=[None, {"style.angry": 0}, None]) is None
    # Diffed regions (no explicit spans) all take `controls`.
    assert _doses(config, controls={"style.angry": 2}) == [{"style.angry": 2.0}]


@pytest.mark.parametrize("fields, message", [
    (dict(region_controls=[{"style.whisper": 1}]), "needs explicit spans"),
    (dict(spans=SPANS, region_controls=[None, {"style.whisper": 1}]), "2 entries for 3 spans"),
    (dict(spans=SPANS, region_controls=[None, None, {"style.shout": 1}]), "Unknown control"),
    (dict(controls={"who.gender": 1}, speaker_embedding=False), "need the speaker row"),
    (dict(spans=SPANS, region_controls=[None, None, {"style.whisper": float("inf")}]), "finite"),
])
def test_bad_region_controls_are_refused(config, fields, message):
    with pytest.raises(ValueError, match=message):
        _doses(config, **fields)


def test_raw_controls_need_a_speaker_row_checkpoint_and_hidden_ones_do_not(config, tmp_path):
    no_row = _Config(config.steer_bundle, speaker_enabled=False)
    with pytest.raises(ValueError, match="need the speaker row"):
        _doses(no_row, controls={"who.gender": 1})
    assert _doses(no_row, controls={"style.whisper": 1}) == [{"style.whisper": 1.0}]
    assert _doses(config, controls={"style.whisper": 1}, speaker_embedding=False) == [{"style.whisper": 1.0}]


def test_controls_need_a_steering_bundle():
    with pytest.raises(ValueError, match="no steering bundle"):
        _doses(_Config(None), controls={"style.whisper": 1})
