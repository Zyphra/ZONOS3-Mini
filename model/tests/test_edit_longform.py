"""Long-form orchestration of zonos3.tts.edit_align with stub workers: Qwen3-ASR transcripts chunk by chunk past 30 s
(``transcribe_audio``) and FuseAlign items routed by Qwen3-ASR utterances (``align_words``)."""

import numpy as np
import pytest
import soundfile as sf

from zonos3.tts import edit_align as E
from zonos3.workers import fusealign


class StubAsr:
    def __init__(self, texts=None, language="English"):
        self.calls: list[list[dict]] = []
        self.texts = texts or {}
        self.language = language

    def transcribe(self, items):
        self.calls.append(items)
        return [{"text": self.texts.get(it.get("start"), f"text {it.get('start')}"),
                 "language": it["language"] or self.language} for it in items]


class StubAligner:
    def __init__(self, chunks):
        self.chunks = chunks
        self.segment_calls: list[list[str]] = []
        self.align_calls: list[list[dict]] = []

    def segments(self, audios):
        self.segment_calls.append(audios)
        return [self.chunks[a] for a in audios]

    def align(self, items):
        self.align_calls.append(items)
        return [[{"word": t, "start": 0.0, "end": 0.1, "presence": 1.0, "placed": True} for t in it["tokens"]]
                for it in items]


def wav(path, seconds, sr=16000):
    sf.write(path, np.zeros(int(seconds * sr), dtype=np.float32), sr)
    return str(path)


def test_constants_match_fusealign_hub():
    from fusealign_hub.aligner import MAX_SECONDS
    from fusealign_hub.longform import MAX_TEXT_BYTES

    assert (E.LONG_FORM_SECONDS, E.MAX_TEXT_BYTES) == (MAX_SECONDS, MAX_TEXT_BYTES)
    assert fusealign.CHUNK_SECONDS < MAX_SECONDS


def test_short_audio_is_one_recognition_without_utterances(tmp_path):
    short = wav(tmp_path / "s.wav", 30.0)  # exactly 30 s is short
    asr, aligner = StubAsr({None: "hello world"}), StubAligner({})
    t = E.transcribe_audio(asr, aligner, short, "English")
    assert (t.text, t.language, t.utterances) == ("hello world", "English", None)
    assert asr.calls == [[{"audio": short, "language": "English"}]] and aligner.segment_calls == []


def test_long_audio_is_transcribed_chunk_by_chunk(tmp_path):
    long = wav(tmp_path / "l.wav", 45.0)
    chunks = {long: [[0.5, 20.0], [21.0, 30.0], [31.0, 44.0]]}
    asr, aligner = StubAsr({0.5: "first part", 21.0: "", 31.0: "last part"}), StubAligner(chunks)
    t = E.transcribe_audio(asr, aligner, long, None)
    assert t.text == "first part last part" and t.language == "English"
    assert t.utterances == [{"text": "first part", "start": 0.5, "end": 20.0},
                            {"text": "last part", "start": 31.0, "end": 44.0}]  # the empty chunk is dropped
    assert asr.calls == [[{"audio": long, "start": s, "end": e, "language": None} for s, e in chunks[long]]]


@pytest.mark.parametrize("forced, detected, joined", [
    ("Japanese", "Japanese", "今日はいい天気"),
    (None, "Chinese,English", "今日はいい天気"),  # detected: the first spoken chunk's first language
    (None, "English", "今日は いい天気"),
])
def test_unspaced_languages_join_chunks_without_spaces(tmp_path, forced, detected, joined):
    long = wav(tmp_path / "l.wav", 40.0)
    asr = StubAsr({0.0: "今日は", 20.0: "いい天気"}, language=detected)
    t = E.transcribe_audio(asr, StubAligner({long: [[0.0, 19.0], [20.0, 39.0]]}), long, forced)
    assert t.text == joined and t.language == (forced or detected)


def test_long_audio_without_speech_has_an_empty_transcript(tmp_path):
    long = wav(tmp_path / "l.wav", 40.0)
    asr = StubAsr()
    t = E.transcribe_audio(asr, StubAligner({long: []}), long, None)
    assert (t.text, t.language, t.utterances) == ("", "", []) and asr.calls == []


def test_align_words_anchors_only_long_form_items_in_one_batch(tmp_path):
    short, long_a, long_b = wav(tmp_path / "s.wav", 5.0), wav(tmp_path / "a.wav", 40.0), wav(tmp_path / "b.wav", 35.0)
    chunks = {long_a: [[0.0, 20.0], [21.0, 39.0]], long_b: [[1.0, 30.0]], short: [[0.0, 5.0]]}
    asr, aligner = StubAsr(), StubAligner(chunks)
    given = [{"text": "given", "start": 0.0, "end": 40.0, "language": "French"}]
    wide = [f"token{k}" for k in range(200)]  # over MAX_TEXT_BYTES: long-form on a short clip
    out = E.align_words(asr, aligner, [
        {"audio": short, "tokens": ["a", "b"], "language": "English"},
        {"audio": long_a, "tokens": ["c"], "language": "English"},
        {"audio": long_b, "tokens": ["d"], "language": None},
        {"audio": long_a, "tokens": ["e"], "language": "English", "utterances": given},
        {"audio": short, "tokens": wide, "language": "German"},
        {"audio": long_a, "tokens": [], "language": "English"},
    ])
    assert [[w["word"] for w in ws] for ws in out] == [["a", "b"], ["c"], ["d"], ["e"], wide, []]
    assert aligner.segment_calls == [[long_a, long_b, short]]  # one segments call for every item that needs anchors
    assert [[(it["start"], it["language"]) for it in batch] for batch in asr.calls] == [
        [(0.0, "English"), (21.0, "English"), (1.0, None), (0.0, "German")]]  # one transcribe call
    (items,) = aligner.align_calls
    assert "utterances" not in items[0] and "utterances" not in items[5]
    assert items[1]["utterances"] == [{"text": "text 0.0", "start": 0.0, "end": 20.0},
                                      {"text": "text 21.0", "start": 21.0, "end": 39.0}]
    assert items[2]["utterances"] == [{"text": "text 1.0", "start": 1.0, "end": 30.0}]
    assert items[3]["utterances"] == [{"text": "given", "start": 0.0, "end": 40.0}]  # reused, not recomputed
    assert items[4]["utterances"] == [{"text": "text 0.0", "start": 0.0, "end": 5.0}]
    assert all(set(it) <= {"audio", "tokens", "utterances"} for it in items)


def test_edit_aligned_words_carry_edit_tokens(tmp_path):
    short = wav(tmp_path / "s.wav", 3.0)
    aligner = StubAligner({})
    words = E.edit_aligned_words(StubAsr(), aligner, short, "Don’t — stop!", ("en", "English"))
    assert [(w.word, w.tokens) for w in words] == [("Don’t", ["DON'T"]), ("stop!", ["STOP"])]
    assert E.edit_aligned_words(StubAsr(), aligner, short, " 。", ("zh", "Chinese")) == []
    assert len(aligner.align_calls) == 1  # no tokens: no alignment call
