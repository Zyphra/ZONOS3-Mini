"""The Qwen3-ASR worker (zonos3/workers/asr.py) on a stub model, and its client (zonos3.tts.edit_align.AsrWorker)."""

import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from zonos3.tts.edit_align import AsrError, AsrWorker
from zonos3.workers import asr as worker

LANGUAGES = ["Chinese", "English", "French"]


class StubModel:
    """qwen_asr's Qwen3ASRModel.transcribe: records the batches, answers with each part's length."""

    def __init__(self):
        self.batches = []

    def transcribe(self, audio, language):
        self.batches.append([(len(wav), sr, lang) for (wav, sr), lang in zip(audio, language, strict=True)])
        return [SimpleNamespace(text=f"  {len(wav)} samples ", language=lang or "English")
                for (wav, _), lang in zip(audio, language)]


def serve(requests, loads=None, model=None):
    model = model or StubModel()

    def load():
        if loads is not None:
            loads.append(1)
        return model

    out = io.StringIO()
    worker.serve([json.dumps(r) if isinstance(r, dict) else r for r in requests], out, "Qwen/stub", LANGUAGES, load)
    return [json.loads(line) for line in out.getvalue().splitlines()], model


def test_ping_answers_without_loading_and_transcribe_slices_and_batches(tmp_path):
    stereo = tmp_path / "a.wav"
    sf.write(stereo, np.zeros((16000 * 4, 2), dtype=np.float32), 16000)
    loads = []
    (ping, out), model = serve([
        {"id": 1, "op": "ping"},
        {"id": 2, "op": "transcribe", "items": [
            {"audio": str(stereo), "language": "French"},
            {"audio": str(stereo), "start": 1.0, "end": 2.5, "language": None},
            {"audio": str(stereo), "start": 3.5, "end": 99.0, "language": "English"},  # clipped to the file
            {"audio": str(stereo), "start": 5.0, "end": 6.0, "language": None},  # past the end: empty
        ]},
    ], loads)
    assert ping == {"id": 1, "status": "ok", "model": "Qwen/stub", "languages": LANGUAGES}
    assert out["status"] == "ok" and loads == [1]
    assert model.batches == [[(64000, 16000, "French"), (24000, 16000, None), (8000, 16000, "English")]]
    assert out["results"] == [{"text": "64000 samples", "language": "French"},
                              {"text": "24000 samples", "language": "English"},
                              {"text": "8000 samples", "language": "English"},
                              {"text": "", "language": ""}]


def test_errors_are_reported_and_the_worker_keeps_serving(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    (language, unknown, extra, start, missing, garbage, ping), _ = serve([
        {"id": 1, "op": "transcribe", "items": [{"audio": str(wav), "language": "Klingon"}]},
        {"id": 2, "op": "nope"},
        {"id": 3, "op": "transcribe", "items": [{"audio": str(wav), "language": None, "lang": "x"}]},
        {"id": 4, "op": "transcribe", "items": [{"audio": str(wav), "start": "1", "language": None}]},
        {"id": 5, "op": "transcribe", "items": [{"audio": str(tmp_path / "missing.wav"), "language": None}]},
        "{not json",
        {"id": 7, "op": "ping"},
    ])
    assert language["invalid"] is True and "not a Qwen3-ASR language" in language["message"]
    assert unknown["invalid"] is True and "unknown op" in unknown["message"]
    assert extra["invalid"] is True and "unknown fields ['lang']" in extra["message"]
    assert start["invalid"] is True and "start" in start["message"]
    assert missing["status"] == "error" and missing["invalid"] is False
    assert garbage["id"] is None and garbage["status"] == "error"
    assert ping["status"] == "ok"


def _scripted(code: str) -> AsrWorker:
    client = AsrWorker(python=sys.executable)
    client._command = [sys.executable, "-c", "import json, sys\n" + code]
    return client


def test_client_pings_once_restarts_and_maps_errors(tmp_path):
    echo = _scripted(
        "for line in sys.stdin:\n"
        "    r = json.loads(line)\n"
        "    a = {'model': 'm', 'languages': ['English']} if r['op'] == 'ping' else "
        "{'results': [{'text': it['audio'], 'language': 'English'} for it in r['items']]}\n"
        "    print(json.dumps({'id': r['id'], 'status': 'ok', **a}), flush=True)")
    try:
        assert echo.model() == "m" and echo.languages() == ["English"] and echo.installed
        assert echo.transcribe([{"audio": "x.wav", "language": None}]) == [{"text": "x.wav", "language": "English"}]
    finally:
        echo.close()
    dead = _scripted("sys.exit(3)")
    with pytest.raises(AsrError):
        dead.call("ping")
    for invalid, raised, message in ((True, ValueError, "ASR failed: bad"), (False, AsrError, "ASR is unavailable: E()")):
        failing = _scripted(
            "for line in sys.stdin:\n"
            f"    print(json.dumps({{'id': json.loads(line)['id'], 'status': 'error', 'error': 'E()', 'message': 'bad',"
            f" 'invalid': {invalid}}}), flush=True)")
        try:
            with pytest.raises(raised):
                failing.transcribe([{"audio": "x.wav", "language": None}])
        finally:
            failing.close()


def test_client_without_its_interpreter_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("ZONOS3_ASR_PYTHON", str(tmp_path / "missing" / "python"))
    client = AsrWorker()
    assert not client.installed
    with pytest.raises(AsrError):
        client.languages()


def test_client_defaults_to_the_repo_venv_and_the_device_env(monkeypatch):
    from zonos3.tts import edit_align

    monkeypatch.delenv("ZONOS3_ASR_PYTHON", raising=False)
    monkeypatch.setenv("ZONOS3_ASR_DEVICE", "cuda:3")
    client = AsrWorker()
    assert client.python == str(edit_align.MODEL_SRC.parents[1] / ".venv-asr" / "bin" / "python")
    assert client._command == [client.python, str(edit_align.ASR_WORKER), "--device", "cuda:3"]
    assert edit_align.ASR_WORKER == edit_align.MODEL_SRC / "zonos3" / "workers" / "asr.py"
    assert edit_align.ASR_WORKER.is_file()
