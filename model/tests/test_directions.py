"""Direction sets (zonos3.tts.directions) and the set calibration a steering bundle exposes (zonos3.tts.steering)."""

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
directions = pytest.importorskip("zonos3.tts.directions")
steering = pytest.importorskip("zonos3.tts.steering")

Calibration, DirectionSet = directions.Calibration, directions.DirectionSet


def _write_set(root, *, space="raw", dim=4, source_index=6, calibration=None):
    """A tiny set: 'happy' named + 'valence' axis (+ 'tempo' of kind axis)."""
    root.mkdir(parents=True, exist_ok=True)
    entries = {"happy": {"kind": "named"}, "valence": {}, "tempo": {"kind": "axis"}}
    for i, name in enumerate(entries):
        vec = np.zeros(dim, dtype=np.float32)
        vec[i] = 1.0
        np.save(root / f"{name}.npy", vec)
    manifest = {
        "dim": dim,
        "space": space,
        "directions": {name: {"file": f"{name}.npy", **entry} for name, entry in entries.items()},
    }
    if space == "attnres_source":
        manifest["source_index"] = source_index
    (root / "manifest.json").write_text(json.dumps(manifest))
    if calibration is not None:
        (root / "calibration.json").write_text(json.dumps(calibration))
    return root


def test_load_classifies_named_and_axes(tmp_path):
    d = DirectionSet.load(_write_set(tmp_path))
    assert d.dim == 4 and d.space == "raw" and d.source_index is None and d.calibration is None
    assert list(d.named) == ["happy"] and list(d.axes) == ["valence", "tempo"]
    assert d.names == ["happy", "valence", "tempo"]
    assert not d.is_empty()
    assert torch.equal(d.axes["tempo"], torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_load_missing_returns_none(tmp_path):
    assert DirectionSet.load(tmp_path / "nope") is None


def test_load_dim_mismatch_raises(tmp_path):
    _write_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["dim"] = 3
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        DirectionSet.load(tmp_path)


@pytest.mark.parametrize("space", ["lda", "proj", "bogus"])
def test_load_rejects_spaces_other_than_raw_and_attnres_source(tmp_path, space):
    _write_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["space"] = space
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        DirectionSet.load(tmp_path)


def test_attnres_source_needs_a_valid_source_index(tmp_path):
    d = DirectionSet.load(_write_set(tmp_path, space="attnres_source", dim=5))
    assert d.space == "attnres_source" and d.source_index == 6
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["source_index"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(KeyError):
        DirectionSet.load(tmp_path)
    manifest["source_index"] = -1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        DirectionSet.load(tmp_path)


def test_set_loads_its_calibration(tmp_path):
    d = DirectionSet.load(_write_set(tmp_path, calibration={"default": {"happy": 2.5}}))
    assert d.calibration == Calibration(by_speaker={}, default={"happy": 2.5}, global_default=1.0)


def test_calibration_load_and_lookup(tmp_path):
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "global_default": 4.0,
                "default": {"happy": 5.0, "sad": 6.0},
                "by_speaker": {"spkA": {"happy": 3.0, "angry": 8.0}},
                "labels": {},
            }
        )
    )
    c = Calibration.load(tmp_path)
    assert c.strength("spkA", "happy") == 3.0  # per-speaker value wins
    assert c.strength("spkA", "angry") == 8.0
    assert c.strength("spkA", "sad") == 6.0  # then the per-direction default
    assert c.strength("spkZ", "happy") == 5.0
    assert c.strength("spkZ", "fearful") == 4.0  # then the global default
    assert c.strength(None, "happy") == 5.0
    assert c.strength(None, "whatever") == 4.0


def test_calibration_missing_returns_none(tmp_path):
    assert Calibration.load(tmp_path) is None


def test_calibration_zero_dose_disables_and_negative_pole(tmp_path):
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "global_default": 4.0,
                "default": {"happy": 5.0, "fearful": 0.0, "tempo-": 1.0},
                "by_speaker": {"spkA": {"happy": 0.0, "clarity-": 0.0}},
            }
        )
    )
    c = Calibration.load(tmp_path)
    names = ["happy", "fearful", "sad", "tempo", "clarity"]
    # per-speaker 0 wins over a non-zero default; a default 0 applies to everyone; "<name>-": 0 lists the pole alone
    assert c.disabled("spkA", names) == ["happy", "fearful", "clarity-"]
    assert c.disabled("spkZ", names) == ["fearful"]
    assert c.disabled(None, names) == ["fearful"]


def test_calibration_document_is_plain_json(tmp_path):
    (tmp_path / "calibration.json").write_text(
        json.dumps(
            {
                "objective": "ignored",
                "global_default": 4,
                "default": {"valence": 4, "valence-": 2},
                "by_speaker": {"spkA": {"arousal": 0}},
            }
        )
    )
    doc = Calibration.load(tmp_path).document()
    assert doc == {
        "default": {"valence": 4.0, "valence-": 2.0},
        "by_speaker": {"spkA": {"arousal": 0.0}},
        "global_default": 4.0,
    }
    assert all(isinstance(v, float) for v in doc["default"].values()) and isinstance(doc["global_default"], float)
    assert json.loads(json.dumps(doc)) == doc


def _bundle(tmp_path, calibration):
    _write_set(tmp_path / "who", dim=4)
    _write_set(tmp_path / "mood", space="attnres_source", dim=5, source_index=3, calibration=calibration)
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"name": "tiny", "sets": {"who": "who", "mood": "mood"}}))
    return steering.load_bundle(path)


def test_controls_document_exposes_the_set_calibration(tmp_path):
    cal = {"default": {"valence": 4.0}, "by_speaker": {"spkA": {"valence-": 0.0}}, "global_default": 2.0}
    doc = steering.controls_document(_bundle(tmp_path, cal), speaker_embedding_dim=4)
    assert doc["sets"]["mood"] == {
        "space": "attnres_source",
        "modes": ["embedding", "reference"],
        "source_index": 3,
        "calibration": cal,
    }
    assert "calibration" not in doc["sets"]["who"]


def test_disabled_controls(tmp_path):
    cal = {"default": {"tempo": 0.0}, "by_speaker": {"spkA": {"valence-": 0.0, "happy": 0.0}}}
    bundle = _bundle(tmp_path, cal)
    assert steering.disabled_controls(bundle, "spkA") == ["mood.happy", "mood.valence-", "mood.tempo"]
    assert steering.disabled_controls(bundle, "spkZ") == ["mood.tempo"]
    assert steering.disabled_controls(bundle, None) == ["mood.tempo"]
    assert steering.disabled_controls(None, "spkA") == []
    assert steering.disabled_controls(_bundle(tmp_path / "plain", {"default": {"happy": 4.0}}), "spkA") == []
