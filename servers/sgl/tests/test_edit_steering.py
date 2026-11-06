"""Steering doses of /v2/edit regions on the sgl server: `controls` + `region_controls[k]` (zonos3.tts.edit_runtime),
checked against the --steer-bundle by the sgl edit engine before any work."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

api_server = pytest.importorskip("zonos3_sgl.server.api_server")
from zonos3.server.schemas import EditRequest  # noqa: E402
from zonos3.tts.edit_runtime import EditModel, EditRuntime, region_doses  # noqa: E402


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


class _Reached(Exception):
    """The edit passed every request check and decodes the clip."""


class _Engine(api_server._SglEditEngine):
    """The sgl engine's checks, on an editing checkpoint with or without a speaker row."""

    def __init__(self, speaker_row):
        self.speaker_row = speaker_row

    @property
    def model(self):
        return EditModel(frame_rate=25.0, n_codebooks=9, audio_pad_id=4097, codebook_sizes=(1024,) * 9, max_spans=4,
                         speaker_row=self.speaker_row)

    async def decode_audio(self, audio):
        raise _Reached


class _Asr:
    def languages(self):
        return ["English"]


@pytest.fixture
def bundle(tmp_path):
    _set(tmp_path, "who", "raw", {"gender": [0.0, 2.0, 0.0]})
    _set(tmp_path, "style", "attnres_source", {"whisper": [1.0, 0.0], "angry": [0.0, 1.0]}, source=19)
    (tmp_path / "bundle.json").write_text(json.dumps({"name": "t", "sets": {"who": "who", "style": "style"}}))
    return str(tmp_path / "bundle.json")


@pytest.fixture
def serve(monkeypatch):
    """The server's EditRuntime over the sgl engine for a --steer-bundle (None: none)."""

    def serve(bundle, speaker_row=True):
        config = SimpleNamespace(steer_bundle=bundle, model_path="",
                                 model_config=SimpleNamespace(condition_tokens_enabled=False))
        monkeypatch.setattr(api_server, "get_global_state", lambda: SimpleNamespace(config=config))
        monkeypatch.setattr(api_server, "_text_tokenizer_spec", lambda config: None)  # byte text, no groups
        return EditRuntime(_Engine(speaker_row), _Asr(), None)

    return serve


def _doses(runtime, **fields):
    """The regions' doses of an edit request, after the engine checked them (returns the raw-space ids too)."""
    doses = region_doses(EditRequest(audio_base64="", target_text="x", **fields), runtime.model)
    return doses, asyncio.run(runtime.engine.check("x", doses, None, None))


def _edit(runtime, **fields):
    """An edit request's refusal message, or None when it passed every check."""
    try:
        asyncio.run(runtime.edit(EditRequest(audio_base64="", target_text="x", **fields)))
    except _Reached:
        return None
    except ValueError as exc:
        return str(exc)
    raise AssertionError("the edit ran without its clip")


SPANS = [[0.1, 0.4], [1.0, 1.5], [2.0, 2.5]]


def test_region_doses_add_to_the_shared_controls_per_id(serve, bundle):
    runtime = serve(bundle)
    doses, raw = _doses(runtime, spans=SPANS, controls={"style.whisper": 1, "who.gender": 1},
                        region_controls=[{"style.whisper": 1}, None, {"who.gender": -1, "style.angry": 4}])
    assert doses == [{"style.whisper": 2.0, "who.gender": 1.0}, {"style.whisper": 1.0, "who.gender": 1.0},
                     {"style.whisper": 1.0, "style.angry": 4.0}]
    assert raw == {"who.gender"}
    assert _doses(runtime, spans=SPANS) == (None, set())
    assert _doses(runtime, spans=SPANS, region_controls=[None, {"style.angry": 0}, None]) == (None, set())
    # Diffed regions (no explicit spans) all take `controls`.
    assert _doses(runtime, controls={"style.angry": 2}) == ([{"style.angry": 2.0}], set())


@pytest.mark.parametrize("fields, message", [
    ({"region_controls": [{"style.whisper": 1}]}, "needs explicit spans"),
    ({"spans": SPANS, "region_controls": [None, {"style.whisper": 1}]}, "2 entries for 3 spans"),
    ({"spans": SPANS, "region_controls": [None, None, {"style.shout": 1}]}, "Unknown control"),
    ({"controls": {"who.gender": 1}, "speaker_embedding": False}, "need the speaker row"),
    ({"spans": SPANS, "region_controls": [None, None, {"style.whisper": float("inf")}]}, "finite"),
])
def test_bad_region_controls_are_refused(serve, bundle, fields, message):
    assert message in _edit(serve(bundle), **fields)


def test_raw_controls_need_a_speaker_row_checkpoint_and_hidden_ones_do_not(serve, bundle):
    no_row = serve(bundle, speaker_row=False)
    assert "need the speaker row" in _edit(no_row, controls={"who.gender": 1})
    assert _edit(no_row, controls={"style.whisper": 1}) is None
    assert _edit(serve(bundle), controls={"style.whisper": 1}, speaker_embedding=False) is None
    assert _edit(serve(bundle), controls={"who.gender": 1}) is None


def test_controls_need_a_steering_bundle(serve):
    assert "no steering bundle (start it with --steer-bundle)" in _edit(serve(None), controls={"style.whisper": 1})
