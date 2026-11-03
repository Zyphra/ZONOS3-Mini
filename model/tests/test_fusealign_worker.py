"""FuseAlign forced alignment: the worker protocol (zonos3/workers/fusealign.py, on fusealign_hub's tiny random model),
its client (zonos3.tts.edit_align.FuseAlignWorker), the alignment tokens (zonos3.tts.align_text) and the edit's words."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import fusealign_hub
from zonos3.tts.align_text import align_tokens
from zonos3.tts.edit_align import FuseAlignError, FuseAlignWorker, aligned_words
from zonos3.workers import fusealign as worker


def tiny_aligner():
    """fusealign_hub's tiny random FuseAlign with a stub text encoder (third_party/fusealign-multilingual/tests)."""
    path = Path(fusealign_hub.__file__).resolve().parents[1] / "tests" / "tiny.py"
    spec = importlib.util.spec_from_file_location("fusealign_tiny", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tiny_aligner()


def serve(requests, loads=None):
    def load():
        if loads is not None:
            loads.append(1)
        return tiny_aligner()

    out = io.StringIO()
    worker.serve([json.dumps(r) if isinstance(r, dict) else r for r in requests], out, "tiny", load)
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_align_tokens_split_spaced_text_and_segment_unspaced_text():
    assert align_tokens(" Hello,  world — again. ", "en") == ["Hello,", "world", "—", "again."]
    assert align_tokens("我们 去公园。OK go", "zh") == ["我", "们", "去", "公", "园。", "OK", "go"]
    assert align_tokens("「今日は」 いい 。", "ja") == ["「今", "日", "は」", "い", "い。"]  # whitespace inside a unit dropped
    assert align_tokens("佢哋 走咗", "yue") == ["佢", "哋", "走", "咗"]
    assert align_tokens("  ", "en") == [] and align_tokens("。", "zh") == []


def test_worker_places_one_word_per_token(tmp_path):
    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.random.default_rng(0).normal(0, 0.1, (32000, 2)).astype(np.float32), 16000)
    mono = tmp_path / "mono.flac"
    sf.write(mono, np.random.default_rng(1).normal(0, 0.1, 44100).astype(np.float32), 44100)
    loads = []
    ping, aligned = serve([
        {"id": 1, "op": "ping"},
        {"id": 2, "op": "align", "items": [
            {"audio": str(stereo), "tokens": ["Hello,", "world", "—"], "language": "en"},
            {"audio": str(mono), "tokens": align_tokens("我们去。", "zh"), "language": "zh"},
            {"audio": str(mono), "tokens": [], "language": None},
        ]},
    ], loads)
    assert ping == {"id": 1, "status": "ok", "model": "tiny"}
    assert aligned["id"] == 2 and aligned["status"] == "ok" and loads == [1]
    first, second, empty = (r["words"] for r in aligned["results"])
    assert [w["word"] for w in first] == ["Hello,", "world", "—"]
    assert [w["word"] for w in second] == ["我", "们", "去。"]
    assert empty == []
    for words, duration in ((first, 2.0), (second, 1.0)):
        for w in words:
            assert set(w) == {"word", "start", "end", "presence", "placed"}
            assert 0.0 <= w["start"] <= w["end"] <= duration and 0.0 <= w["presence"] <= 1.0
    dash = first[2]  # punctuation only: zero width at the previous word's end, unplaced
    assert (dash["start"], dash["end"], dash["presence"], dash["placed"]) == (first[1]["end"], first[1]["end"], 0.0, False)


def test_worker_reports_errors_and_keeps_serving(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    loads = []
    spaced, unknown, missing, garbage, ping = serve([
        {"id": 1, "op": "align", "items": [{"audio": str(wav), "tokens": ["big world"], "language": None}]},
        {"id": 2, "op": "nope"},
        {"id": 3, "op": "align", "items": [{"audio": str(tmp_path / "missing.wav"), "tokens": ["a"], "language": None}]},
        "{not json",
        {"id": 5, "op": "ping"},
    ], loads)
    assert spaced["id"] == 1 and spaced["status"] == "error" and spaced["invalid"] is True
    assert "whitespace" in spaced["message"] and spaced["error"].startswith("ValueError(")
    assert unknown["invalid"] is True and "unknown op" in unknown["message"]
    assert missing["id"] == 3 and missing["status"] == "error" and missing["invalid"] is False
    assert garbage["id"] is None and garbage["status"] == "error"
    assert ping["status"] == "ok" and loads == [1]  # the model loaded once, on the first align


def test_client_pings_the_worker_without_loading_the_model():
    client = FuseAlignWorker(device="cpu")
    try:
        assert client.model() == f"{fusealign_hub.aligner.HF_REPO}/{fusealign_hub.aligner.CHECKPOINT_FILE}"
        assert client.alive
    finally:
        client.close()
    assert not client.alive


def _scripted(code: str) -> FuseAlignWorker:
    client = FuseAlignWorker()
    client._command = [sys.executable, "-c", "import json, sys\n" + code]
    return client


def test_client_restarts_a_worker_that_exited_and_maps_errors():
    one_shot = _scripted("r = json.loads(sys.stdin.readline())\n"
                         "print(json.dumps({'id': r['id'], 'status': 'ok', 'model': 'one-shot'}), flush=True)")
    try:
        assert one_shot.call("ping")["model"] == "one-shot"
        assert one_shot.call("ping")["id"] == 2  # the first worker exited after one answer: a fresh one answered
    finally:
        one_shot.close()
    dead = _scripted("sys.exit(3)")
    with pytest.raises(FuseAlignError, match="exited with code 3"):
        dead.call("ping")
    for invalid, raised in ((True, ValueError), (False, FuseAlignError)):
        failing = _scripted(
            "for line in sys.stdin:\n"
            f"    print(json.dumps({{'id': json.loads(line)['id'], 'status': 'error', 'error': 'E()', 'message': 'bad',"
            f" 'invalid': {invalid}}}), flush=True)")
        try:
            with pytest.raises(raised):
                failing.align([{"audio": "x.wav", "tokens": ["a"], "language": None}])
        finally:
            failing.close()


def test_edit_words_take_presence_and_drop_punctuation_only_words():
    words = [{"word": "Hello,", "start": 0.1, "end": 0.4, "presence": 0.9, "placed": True},
             {"word": "—", "start": 0.4, "end": 0.4, "presence": 0.0, "placed": False},
             {"word": "e-mail", "start": 0.5, "end": 0.9, "presence": 0.05, "placed": False}]
    out = aligned_words(words, [["HELLO"], [], ["E", "MAIL"]])
    assert [(w.word, w.start, w.end, w.confidence, w.tokens) for w in out] == [
        ("Hello,", 0.1, 0.4, 0.9, ["HELLO"]), ("e-mail", 0.5, 0.9, 0.05, ["E", "MAIL"])]
    with pytest.raises(ValueError, match="One token list"):
        aligned_words(words, [["HELLO"]])


def test_edit_words_split_overlaps_so_they_round_trip_as_original_words():
    from zonos3.tts.edit_align import edit_words

    words = [{"word": "i'm", "start": 0.923, "end": 1.063, "presence": 1.0},
             {"word": "wen", "start": 1.062, "end": 1.234, "presence": 1.0},
             {"word": "di", "start": 1.235, "end": 1.368, "presence": 1.0}]
    out = aligned_words(words, [["I'M"], ["WEN"], ["DI"]])
    assert [(w.start, w.end) for w in out] == [(0.923, pytest.approx(1.0625)), (pytest.approx(1.0625), 1.234),
                                               (1.235, 1.368)]
    given = [{"word": w.word, "start": w.start, "end": w.end} for w in out]
    back = edit_words(given, [w.tokens for w in out], ["I'M", "WEN", "DI"], duration=2.0)
    assert [(w.start, w.end) for w in back] == [(w.start, w.end) for w in out]
