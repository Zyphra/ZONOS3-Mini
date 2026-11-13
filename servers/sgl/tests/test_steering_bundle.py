"""Steering bundles (zonos3.tts.steering): several direction sets driven by one `controls` field,
and hidden-space controls over character spans."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from zonos3.tts.steering import load_bundle, resolve_controls, schedule_delta


def _set(root, name, space, vectors, source=None, axes=()):
    d = root / name
    d.mkdir()
    entries = {}
    for k, v in vectors.items():
        np.save(d / f"{k}.npy", np.array(v, dtype=np.float32))
        entries[k] = {"file": f"{k}.npy", "kind": "axis" if k in axes else "named"}
    manifest = {"dim": len(next(iter(vectors.values()))), "space": space, "directions": entries}
    if source is not None:
        manifest["source_index"] = source
    (d / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def bundle(tmp_path):
    _set(tmp_path, "who", "raw", {"gender": [0.0, 2.0, 0.0]}, axes=("gender",))
    _set(tmp_path, "accent", "raw", {"british": [0.0, 0.0, 1.0]})
    _set(tmp_path, "style", "attnres_source", {"whisper": [1.0, 0.0], "tempo": [0.0, 1.0]}, source=19, axes=("tempo",))
    _set(tmp_path, "deep", "attnres_source", {"cmn": [0.0, 3.0]}, source=15)
    spec = {
        "name": "t",
        "sets": {"who": "who", "accent": "accent", "style": "style", "deep": "deep"},
        "timing": {"style.tempo": [[-1, 2.0], [0, 1.0], [1, 0.5]]},
        "ui": {"controls": []},
    }
    (tmp_path / "bundle.json").write_text(json.dumps(spec))
    return load_bundle(tmp_path / "bundle.json")


def resolve(bundle, controls=None, spans=None, emb=torch.tensor([4.0, 0.0, 0.0]), text="x" * 20, rate=None):
    return resolve_controls(
        bundle, controls, spans, text=text, speaker_embedding=emb, sources=(15, 19), hidden_size=2, frames_per_char=rate
    )


def test_raw_controls_from_several_sets_move_the_embedding_once_keeping_its_norm(bundle):
    r = resolve(bundle, {"who.gender": 1.5, "accent.british": 4.0})
    # [4, 0, 0] + 1.5 x [0, 2, 0] + 4 x [0, 0, 1] = [4, 3, 4], rescaled to norm 4
    assert torch.allclose(r.speaker_embedding, torch.tensor([4.0, 3.0, 4.0]) * (4.0 / float(np.sqrt(41.0))))
    assert r.hidden_delta is None and r.schedule is None


def test_hidden_controls_land_in_their_own_bank_source_slot(bundle):
    r = resolve(bundle, {"style.whisper": 2.0, "deep.cmn": 1.0})
    # sources (15, 19): slot 0 = deep.cmn, slot 1 = style.whisper
    assert torch.equal(r.hidden_delta, torch.tensor([0.0, 3.0, 2.0, 0.0]))
    assert torch.equal(r.speaker_embedding, torch.tensor([4.0, 0.0, 0.0]))


def test_raw_controls_need_an_embedding_and_unknown_controls_are_refused(bundle):
    with pytest.raises(ValueError):
        resolve(bundle, {"who.gender": 1.0}, emb=None)
    assert resolve(bundle, {"style.whisper": 1.0}, emb=None).hidden_delta is not None  # hidden works for clones
    with pytest.raises(ValueError):
        resolve(bundle, {"style.shout": 1.0})


def test_spans_are_placed_by_character_at_their_own_pace(bundle):
    # 2 frames per character; the global tempo +1 halves it; the span's tempo -1 doubles it inside the span
    r = resolve(
        bundle,
        {"style.tempo": 1.0},
        [{"start": 4, "end": 8, "controls": {"style.tempo": -1.0, "style.whisper": 1.0}}],
        rate=2.0,
    )
    start, end, ramp = r.schedule[0, :3].tolist()
    assert (start, end, ramp) == (4.0, 4.0 + 4 * 2.0 * 0.5 * 2.0, 3.0)
    assert torch.equal(r.schedule[0, 3:], torch.tensor([0.0, 0.0, 1.0, -1.0]))


@pytest.mark.parametrize(
    "spans, message",
    [
        (
            [
                {"start": 0, "end": 5, "controls": {"style.whisper": 1}},
                {"start": 3, "end": 9, "controls": {"style.whisper": 1}},
            ],
            "overlap",
        ),
        ([{"start": 0, "end": 5, "controls": {"who.gender": 1}}], "cannot vary"),
        ([{"start": 5, "end": 25, "controls": {"style.whisper": 1}}], "outside the text"),
    ],
)
def test_bad_spans_are_refused(bundle, spans, message):
    with pytest.raises(ValueError):
        resolve(bundle, None, spans)


def test_schedule_ramps_in_and_out_around_the_span():
    schedule = torch.tensor([[10.0, 20.0, 4.0, 2.0, -2.0]])
    weight = lambda f: None if schedule_delta(schedule, f) is None else float(schedule_delta(schedule, f)[0]) / 2.0
    assert weight(5) is None and weight(25) is None
    assert weight(8) == pytest.approx(0.5) and weight(22) == pytest.approx(0.5)
    assert weight(10) == 1.0 and weight(19) == 1.0


def test_request_steer_delta_adds_the_active_spans_to_the_constant_delta():
    from zonos3_sgl.core import TTSReq

    req = TTSReq(
        input_ids=torch.zeros((2, 10), dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=50,
        uid=0,
        sampling_params=None,
        cache_handle=None,
        audio_pad_id=1025,
        stop_id=1024,
        steering_delta=torch.tensor([1.0, 1.0]),
        steer_schedule=torch.tensor([[3.0, 5.0, 1.0, 0.0, 4.0]]),
    )
    assert torch.equal(req.steer_delta(), torch.tensor([1.0, 1.0]))  # frame 0: outside the span
    req.total_generated = 4
    assert torch.equal(req.steer_delta(), torch.tensor([1.0, 5.0]))


def test_literal_angle_text_contributes_to_span_timing(bundle):
    result = resolve(
        bundle,
        spans=[{"start": 11, "end": 13, "controls": {"style.whisper": 1.0}}],
        text="ab <laugh/>cd",
        rate=2.0,
    )
    assert result.schedule[0, :2].tolist() == [22.0, 26.0]
