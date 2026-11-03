"""FuseAlign forced alignment: the worker protocol (zonos3/workers/fusealign.py, on fusealign_hub's tiny random model:
segments, align with Qwen3-ASR utterances, never Whisper), its client (zonos3.tts.edit_align.FuseAlignWorker), the
alignment tokens (zonos3.tts.align_text) and the edit's words."""

import importlib.util
import io
import json
import math
import re
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


def serve(requests, loads=None, load=None, speech_regions=None):
    from fusealign_hub.longform import guide_words

    def default_load():
        if loads is not None:
            loads.append(1)
        return tiny_aligner()

    def no_vad(_wav):
        raise AssertionError("unexpected VAD")

    out = io.StringIO()
    worker.serve([json.dumps(r) if isinstance(r, dict) else r for r in requests], out, "tiny", load or default_load,
                 speech_regions or no_vad, guide_words)
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
            {"audio": str(stereo), "tokens": ["Hello,", "world", "—"]},
            {"audio": str(mono), "tokens": align_tokens("我们去。", "zh")},
            {"audio": str(mono), "tokens": [], "utterances": []},
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
    spaced, unknown, missing, garbage, language, utterances, ping = serve([
        {"id": 1, "op": "align", "items": [{"audio": str(wav), "tokens": ["big world"]}]},
        {"id": 2, "op": "nope"},
        {"id": 3, "op": "align", "items": [{"audio": str(tmp_path / "missing.wav"), "tokens": ["a"]}]},
        "{not json",
        {"id": 5, "op": "align", "items": [{"audio": str(wav), "tokens": ["a"], "language": "en"}]},
        {"id": 6, "op": "align", "items": [{"audio": str(wav), "tokens": ["a"], "utterances": [{"text": "a"}]}]},
        {"id": 7, "op": "ping"},
    ], loads)
    assert spaced["id"] == 1 and spaced["status"] == "error" and spaced["invalid"] is True
    assert "whitespace" in spaced["message"] and spaced["error"].startswith("ValueError(")
    assert unknown["invalid"] is True and "unknown op" in unknown["message"]
    assert missing["id"] == 3 and missing["status"] == "error" and missing["invalid"] is False
    assert garbage["id"] is None and garbage["status"] == "error"
    assert language["invalid"] is True and "unknown fields ['language']" in language["message"]
    assert utterances["invalid"] is True and "utterances" in utterances["message"]
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
    for invalid, raised, message in ((True, ValueError, "FuseAlign alignment failed: bad"),
                                     (False, FuseAlignError, "FuseAlign alignment is unavailable: E()")):
        failing = _scripted(
            "for line in sys.stdin:\n"
            f"    print(json.dumps({{'id': json.loads(line)['id'], 'status': 'error', 'error': 'E()', 'message': 'bad',"
            f" 'invalid': {invalid}}}), flush=True)")
        try:
            with pytest.raises(raised, match=rf"^{re.escape(message)}$"):
                failing.align([{"audio": "x.wav", "tokens": ["a"]}])
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


def test_pack_regions_into_chunks_of_at_most_25_seconds():
    assert worker.pack_regions([]) == []
    assert worker.pack_regions([(0.5, 3.0), (4.0, 20.0), (21.0, 25.5), (26.0, 30.0)]) == [[0.5, 25.5], [26.0, 30.0]]
    # a region longer than 25 s: the fewest equal parts, each a chunk; its neighbours start chunks of their own
    assert worker.pack_regions([(0.0, 2.0), (3.0, 63.0), (64.0, 65.0)]) == [
        [0.0, 2.0], [3.0, 23.0], [23.0, 43.0], [43.0, 63.0], [64.0, 65.0]]
    assert worker.pack_regions([(1.0, 26.0), (26.5, 27.0)]) == [[1.0, 26.0], [26.5, 27.0]]  # exactly 25 s fits
    chunks = worker.pack_regions([(0.0, 51.0)])
    assert len(chunks) == 3 and all(e - s <= 25.0 for s, e in chunks) and chunks[-1][1] == 51.0


def test_segments_op_runs_vad_on_16k_mono_without_the_model(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros((44100 * 3, 2), dtype=np.float32), 44100)
    seen = []

    def vad(w):
        seen.append(w.numel())
        return [{"start": 0.2, "end": 1.0}, {"start": 1.5, "end": 2.5}]

    def load():
        raise AssertionError("segments must not load the aligner")

    (res,) = serve([{"id": 1, "op": "segments", "items": [{"audio": str(wav)}]}], load=load, speech_regions=vad)
    assert res == {"id": 1, "status": "ok", "results": [{"chunks": [[0.2, 2.5]]}]}
    assert seen == [16000 * 3]
    (bad,) = serve([{"id": 2, "op": "segments", "items": [{"audio": str(wav), "language": "en"}]}], speech_regions=vad)
    assert bad["invalid"] is True


def test_real_vad_finds_no_chunks_in_silence(tmp_path):
    from fusealign_hub.vad import speech_regions

    wav = tmp_path / "silence.wav"
    sf.write(wav, np.zeros(16000 * 2, dtype=np.float32), 16000)
    (res,) = serve([{"id": 1, "op": "segments", "items": [{"audio": str(wav)}]}], speech_regions=speech_regions)
    assert res["results"] == [{"chunks": []}]


class RecordingAligner:
    """An aligner that records what the worker hands to ``align_batch``."""

    def __init__(self):
        self.calls = []

    def align_batch(self, waveforms, sample_rate, transcripts, asr_words=None, language=None):
        self.calls.append({"asr_words": asr_words, "language": language})
        return [[] for _ in waveforms]


def test_align_passes_guide_words_of_the_utterances_and_never_none(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
    aligner = RecordingAligner()
    (res,) = serve([{"id": 1, "op": "align", "items": [
        {"audio": str(wav), "tokens": ["a", "b"], "utterances": [{"text": "a b", "start": 0.0, "end": 1.0}]},
        {"audio": str(wav), "tokens": ["c"]},
    ]}], load=lambda: aligner)
    assert res["status"] == "ok"
    (call,) = aligner.calls
    assert call["language"] is None
    assert call["asr_words"] == [[{"word": "a", "start": 0.0, "end": 0.5}, {"word": "b", "start": 0.5, "end": 1.0}], []]
    assert aligner.whisper_words is worker.no_whisper  # the guard the worker installs


def test_long_form_alignment_never_constructs_whisper(tmp_path, monkeypatch):
    """A 40 s recording and a transcript over MAX_TEXT_BYTES align long-form on Qwen3-ASR utterances (or FuseAlign's VAD
    fallback without them): transformers' Whisper pipeline is never built."""
    import transformers

    def no_pipeline(*args, **kwargs):
        raise AssertionError("Whisper pipeline constructed")

    monkeypatch.setattr(transformers, "pipeline", no_pipeline)
    long = tmp_path / "long.wav"
    sr = 16000
    t = np.arange(40 * sr) / sr
    sf.write(long, (0.1 * np.sin(2 * math.pi * 220 * t) * (np.sin(2 * math.pi * 0.25 * t) > 0)).astype(np.float32), sr)
    short = tmp_path / "short.wav"
    sf.write(short, np.random.default_rng(0).normal(0, 0.1, 5 * sr).astype(np.float32), sr)
    words = [f"word{k}" for k in range(40)]
    many = [f"token{k}" for k in range(200)]  # over 1022 bytes on a short clip
    utterances = [{"text": " ".join(words[:20]), "start": 0.0, "end": 20.0},
                  {"text": " ".join(words[20:]), "start": 20.0, "end": 40.0}]
    (res,) = serve([{"id": 1, "op": "align", "items": [
        {"audio": str(long), "tokens": words, "utterances": utterances},
        {"audio": str(long), "tokens": words},
        {"audio": str(short), "tokens": many, "utterances": []},
    ]}])
    assert res["status"] == "ok", res
    anchored, fallback, wide = (r["words"] for r in res["results"])
    assert len(anchored) == len(fallback) == 40 and len(wide) == 200
    for ws, duration in ((anchored, 40.0), (fallback, 40.0), (wide, 5.0)):
        assert all(0.0 <= w["start"] <= w["end"] <= duration + 1e-6 for w in ws)
