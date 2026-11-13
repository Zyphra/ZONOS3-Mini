"""Steer then hold (zonos3.tts.steering ``steer_release``): the bundle key, the request fields, the split of the
request-level hidden controls into a constant and a released delta, the per-frame weight and what the documents report."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from zonos3.server.schemas import EditRequest, GenerateRequest
from zonos3.tts import plan
from zonos3.tts.edit_runtime import FillJob, fill_steering
from zonos3.tts.steering import (
    ReleaseSchedule,
    controls_document,
    load_bundle,
    release_document,
    release_weight,
    released_delta,
    request_release,
    resolve_controls,
)

from test_request_rules import Context  # noqa: E402

RELEASE = {"onset_seconds": 4.0, "ramp_seconds": 1.0, "hold": 0.25, "exempt": ["style.tempo"]}


def _set(root, name, space, vectors, source=None):
    d = root / name
    d.mkdir()
    for k, v in vectors.items():
        np.save(d / f"{k}.npy", np.array(v, dtype=np.float32))
    manifest = {
        "dim": len(next(iter(vectors.values()))),
        "space": space,
        "directions": {k: {"file": f"{k}.npy", "kind": "named"} for k in vectors},
    }
    if source is not None:
        manifest["source_index"] = source
    (d / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture
def make_bundle(tmp_path):
    """A bundle file with a raw set and two hidden sets of width 2; ``release``: its steer_release (None: no key)."""
    _set(tmp_path, "who", "raw", {"gender": [0.0, 2.0, 0.0]})
    _set(tmp_path, "style", "attnres_source", {"whisper": [1.0, 0.0], "tempo": [0.0, 1.0]}, source=19)
    _set(tmp_path, "mood", "attnres_source", {"arousal": [2.0, 0.0]}, source=15)

    def make(release=RELEASE, name="bundle.json"):
        spec = {"name": "t", "sets": {"who": "who", "style": "style", "mood": "mood"}}
        if release is not None:
            spec["steer_release"] = release
        (tmp_path / name).write_text(json.dumps(spec))
        return load_bundle(tmp_path / name)

    return make


def test_release_weight_holds_full_dose_then_ramps_to_hold():
    assert [release_weight(f, 4, 2, 0.25) for f in (0, 3, 4, 5, 6, 100)] == [1.0, 1.0, 1.0, 0.625, 0.25, 0.25]
    assert [release_weight(f, 4, 0, 0.5) for f in (3, 4)] == [1.0, 0.5]  # ramp 0: a step at onset
    assert release_weight(0, 0, 0, 0.0) == 0.0
    base, rel = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 4.0])
    assert torch.equal(released_delta(base, rel, (4, 2, 0.25), 5), torch.tensor([1.0, 2.5]))
    assert released_delta(base, None, None, 5) is base  # nothing released: the constant delta itself
    assert torch.equal(released_delta(None, rel, (4, 2, 0.25), 0), rel)


def test_schedule_frames_round_at_the_codec_rate():
    assert ReleaseSchedule(4.0, 1.0, 0.25).frames(15.0) == (60, 15, 0.25)
    assert ReleaseSchedule(0.1, 0.03, 0.5).frames(15.0) == (2, 0, 0.5)


def test_bundle_key_loads_and_is_validated(make_bundle):
    assert make_bundle().steer_release == ReleaseSchedule(4.0, 1.0, 0.25, ("style.tempo",))
    assert make_bundle(None).steer_release is None
    assert make_bundle({"onset_seconds": 1, "ramp_seconds": 0, "hold": 1}).steer_release.exempt == ()
    bad = [
        ({**RELEASE, "hold": 1.5}, "steer_hold must be in"),
        ({**RELEASE, "onset_seconds": -1}, "steer_onset_seconds must be finite"),
        ({**RELEASE, "ramp_seconds": float("inf")}, "steer_ramp_seconds must be finite"),
        ({**RELEASE, "extra": 1}, "must be {onset_seconds"),
        ({"onset_seconds": 4, "hold": 0.5}, "must be {onset_seconds"),
        ({**RELEASE, "exempt": ["who.gender"]}, "not hidden directions"),
        ({**RELEASE, "exempt": ["style.shout"]}, "not hidden directions"),
    ]
    for release, message in bad:
        with pytest.raises(ValueError, match=message.replace("{", r"\{")):
            make_bundle(release, name="bad.json")


def test_request_release_fills_the_bundle_defaults(make_bundle):
    bundle = make_bundle()
    assert request_release(bundle, None, None, None, hint="h") == bundle.steer_release
    assert request_release(bundle, 0.5, 2.0, None, hint="h") == ReleaseSchedule(2.0, 1.0, 0.5, ("style.tempo",))
    assert request_release(bundle, 1.0, None, None, hint="h") is None  # hold 1: constant
    assert request_release(make_bundle({**RELEASE, "hold": 1.0}, "one.json"), None, 3.0, None, hint="h") is None
    # no key: constant unless the request asks for a hold below 1
    plain = make_bundle(None, "plain.json")
    assert request_release(plain, None, 2.0, 2.0, hint="h") is None
    assert request_release(plain, 1.0, None, None, hint="h") is None
    with pytest.raises(ValueError, match="this server's bundle has none"):
        request_release(plain, 0.5, None, None, hint="h")
    with pytest.raises(ValueError, match=r"no steering bundle \(start it\); steer_hold below 1"):
        request_release(None, 0.0, None, None, hint="start it")
    assert request_release(None, None, None, None, hint="h") is None


def test_split_keeps_exempt_controls_constant(make_bundle):
    bundle = make_bundle()
    controls = {"style.whisper": 2.0, "style.tempo": 1.0, "mood.arousal": 1.0, "who.gender": 1.0}
    kw = dict(text="x" * 10, speaker_embedding=torch.tensor([4.0, 0.0, 0.0]), sources=(15, 19), hidden_size=2)
    whole = resolve_controls(bundle, controls, None, **kw)
    split = resolve_controls(bundle, controls, None, release_exempt=("style.tempo",), **kw)
    assert whole.release_delta is None
    # sources (15, 19): [mood.arousal | style.*]
    assert torch.equal(split.hidden_delta, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.equal(split.release_delta, torch.tensor([2.0, 0.0, 2.0, 0.0]))
    assert torch.equal(split.hidden_delta + split.release_delta, whole.hidden_delta)
    assert torch.equal(split.speaker_embedding, whole.speaker_embedding)  # raw controls never release
    only_exempt = resolve_controls(bundle, {"style.tempo": 1.0}, None, release_exempt=("style.tempo",), **kw)
    assert only_exempt.release_delta is None and only_exempt.hidden_delta is not None
    none_exempt = resolve_controls(bundle, {"mood.arousal": 1.0}, None, release_exempt=("style.tempo",), **kw)
    assert none_exempt.hidden_delta is None and none_exempt.release_delta is not None


class SteerContext(Context):
    steering_sources = (15, 19)
    hidden_size = 2
    steer_hint = "start it with --steer-bundle"

    def __init__(self, bundle):
        super().__init__()
        self.steer_bundle = bundle


def _rules(ctx, request, embedding=None):
    req = plan.parse_request(request)
    head = plan.request_head(ctx, req)
    return plan.request_rules(ctx, req, head, plan.Speaker(head.mode, embedding, None))


def test_request_rules_release_the_request_controls(make_bundle):
    ctx = SteerContext(make_bundle())
    controls = {"style.whisper": 2.0, "style.tempo": 1.0}
    r = _rules(ctx, {"text": "hello", "controls": controls})
    assert r.release == (60, 15, 0.25)
    assert torch.equal(r.hidden_delta, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.equal(r.release_delta, torch.tensor([0.0, 0.0, 2.0, 0.0]))
    r = _rules(ctx, {"text": "hello", "controls": controls, "steer_hold": 0.5, "steer_ramp_seconds": 0.0})
    assert r.release == (60, 0, 0.5)
    # hold 1, or only exempt controls: the constant delta of before, nothing released
    constant = _rules(ctx, {"text": "hello", "controls": controls, "steer_hold": 1.0})
    assert constant.release is None and constant.release_delta is None
    assert torch.equal(constant.hidden_delta, torch.tensor([0.0, 0.0, 2.0, 1.0]))
    tempo = _rules(ctx, {"text": "hello", "controls": {"style.tempo": 1.0}})
    assert tempo.release is None and tempo.release_delta is None
    # control_spans stay in the schedule
    spans = _rules(ctx, {"text": "hello", "control_spans": [{"start": 0, "end": 2, "controls": {"mood.arousal": 1}}]})
    assert spans.release_delta is None and spans.schedule is not None


def test_bundle_without_the_key_is_the_constant_push(make_bundle):
    ctx = SteerContext(make_bundle(None))
    r = _rules(ctx, {"text": "hello", "controls": {"style.whisper": 2.0}, "steer_onset_seconds": 1.0})
    assert r.release is None and r.release_delta is None
    assert torch.equal(r.hidden_delta, torch.tensor([0.0, 0.0, 2.0, 0.0]))
    with pytest.raises(ValueError, match="this server's bundle has none"):
        _rules(ctx, {"text": "hello", "steer_hold": 0.5})
    with pytest.raises(ValueError, match=r"no steering bundle \(start it with --steer-bundle\); steer_hold below 1"):
        _rules(Context(), {"text": "hello", "steer_hold": 0.5})


def test_controls_cfg_twins_a_released_only_request(make_bundle):
    ctx = SteerContext(make_bundle())
    r = _rules(
        ctx,
        {"text": "hello", "controls": {"mood.arousal": 1.0}, "controls_cfg_scale": 2.0, "speaker_mode": "embedding"},
        embedding=torch.ones(3),
    )
    assert r.hidden_delta is None and r.release_delta is not None and r.cfg_kind == "controls"
    p = plan.plan_prompt(
        ctx,
        {"text": "hello", "controls": {"mood.arousal": 1.0}, "controls_cfg_scale": 2.0, "speaker_mode": "embedding"},
        lambda req, mode: plan.Speaker(mode, torch.ones(3), None),
    )
    assert p.release == (60, 15, 0.25) and p.twin is not None and not p.twin.steered


def test_edit_fills_split_their_region_controls(make_bundle):
    bundle = make_bundle()

    def job(**kw):
        return FillJob(
            text="x",
            language="en_us",
            text_normalization=False,
            sampling={},
            seed=0,
            max_tokens=10,
            min_frames=0,
            condition_buckets=None,
            condition_values=None,
            speaker_embedding=None,
            controls={"style.whisper": 1.0, "style.tempo": 1.0},
            controls_text="x",
            window_codes=np.zeros((1, 1)),
            spans=[[0, 1]],
            **kw,
        )

    kw = dict(hint="h", sources=(15, 19), hidden_size=2, frame_rate=15.0)
    s = fill_steering(bundle, job(), **kw)
    assert s.release == (60, 15, 0.25)
    assert torch.equal(s.release_delta, torch.tensor([0.0, 0.0, 1.0, 0.0]))
    assert torch.equal(s.hidden_delta, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    s = fill_steering(bundle, job(steer_hold=1.0), **kw)
    assert s.release is None and s.release_delta is None
    assert torch.equal(s.hidden_delta, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="bundle has none"):
        fill_steering(make_bundle(None, "plain.json"), job(steer_hold=0.5), **kw)


def test_documents_report_the_defaults(make_bundle):
    doc = {"onset_seconds": 4.0, "ramp_seconds": 1.0, "hold": 0.25, "exempt": ["style.tempo"]}
    assert release_document(make_bundle()) == doc
    assert controls_document(make_bundle(), speaker_embedding_dim=3)["steer_release"] == doc
    assert release_document(make_bundle(None, "plain.json")) is None and release_document(None) is None


@pytest.mark.parametrize(
    "fields",
    [{"steer_hold": 1.5}, {"steer_hold": -0.1}, {"steer_onset_seconds": -1.0}, {"steer_ramp_seconds": 601.0}],
)
def test_request_fields_are_range_checked(fields):
    with pytest.raises(ValidationError):
        GenerateRequest(text="hi", **fields)
    with pytest.raises(ValidationError):
        EditRequest(audio_base64="", target_text="hi", **fields)
    ok = GenerateRequest(text="hi", steer_hold=0.0, steer_onset_seconds=0.0, steer_ramp_seconds=600.0)
    assert plan.parse_request(ok.model_dump())["steer_hold"] == 0.0
