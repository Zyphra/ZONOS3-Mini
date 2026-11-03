"""zonos3.tts.edit_runtime.EditRuntime over a fake EditEngine and fake ASR / FuseAlign workers (no model, no
subprocess): what a /v2/edit caller sees."""

import asyncio
import base64

import numpy as np
import pytest
from zonos3.dub.runtime import AlignWorkerError, UnknownId
from zonos3.generation import SAMPLE_RATE
from zonos3.server.schemas import EditRequest
from zonos3.tts.edit_align import AsrError, FuseAlignError
from zonos3.tts.edit_runtime import EditModel, EditRuntime, status_of

K = 4
FRAME_RATE = 10.0
FRAMES = 30  # a 3 s clip
MODEL = EditModel(frame_rate=FRAME_RATE, n_codebooks=K, audio_pad_id=1024, codebook_sizes=(1024,) * K, max_spans=4,
                  speaker_row=True)
CLIP = base64.b64encode(b"an upload").decode("ascii")


class FakeEngine:
    """Codes 0.. of a 3 s clip; region k's fill is ``fill_frames`` frames of the code 900 + k."""

    def __init__(self, model: EditModel = MODEL, raw: set[str] = frozenset(), fill_frames: int = 7):
        self.model = model
        self.raw = set(raw)
        self.fill_frames = fill_frames
        self.codes = np.arange(FRAMES * K, dtype=np.int64).reshape(FRAMES, K)
        self.checked, self.jobs, self.decoded = [], [], []

    async def decode_audio(self, audio):
        assert audio == b"an upload"
        return np.zeros(int(FRAMES / FRAME_RATE * 16000), dtype=np.float32), 16000

    async def embed(self, audio):
        return [0.25] * 8

    async def encode(self, audio, max_seconds):
        return self.codes

    async def check(self, target_text, doses, condition_buckets, condition_values):
        self.checked.append(doses)
        return self.raw

    async def fill(self, job):
        self.jobs.append(job)
        # delay-stacked rows: the fill's frames plus the K-1 rows of its delay tail, ended on span-end at frame n
        n = self.fill_frames
        return np.full((n + K - 1, K), 900 + len(self.jobs) - 1, dtype=np.int64), n

    async def decode(self, codes):
        self.decoded.append(np.asarray(codes))
        return np.zeros(int(len(codes) * SAMPLE_RATE / FRAME_RATE), dtype=np.float32)


class FakeAsr:
    def __init__(self, heard: str | None = None):
        self.heard = heard  # None: transcribing is a test failure
        self.transcribed = []

    def languages(self):
        return ["English"]

    def transcribe(self, items):
        assert self.heard is not None, "the edit transcribed"
        self.transcribed.extend(items)
        return [{"text": self.heard, "language": "English"} for _ in items]


class FakeAligner:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.aligned = []

    def segments(self, audios):
        raise AssertionError("no clip here is long-form")

    def align(self, items):
        assert self.enabled, "the edit aligned words"
        self.aligned.extend(items)
        return [[{"word": t, "start": 0.2 * i, "end": 0.2 * i + 0.15, "presence": 0.9, "placed": True}
                 for i, t in enumerate(item["tokens"])] for item in items]


def edit(runtime: EditRuntime, **body) -> dict:
    return asyncio.run(runtime.edit(EditRequest(audio_base64=CLIP, target_text="hello brave new world", seed=5,
                                                **body)))


def test_unedited_frames_survive_the_splice_and_spans_skip_transcription():
    engine = FakeEngine(fill_frames=7)
    out = edit(EditRuntime(engine, FakeAsr(), FakeAligner()), spans=[[1.0, 1.5]], min_fill_ratio=0.0)

    (job,) = engine.jobs
    assert job.spans == [[10, 5]] and job.min_frames == 0 and job.seed == 5
    np.testing.assert_array_equal(job.window_codes, engine.codes)
    assert job.speaker_embedding == [0.25] * 8 and job.controls is None
    (spliced,) = engine.decoded
    assert spliced.shape == (FRAMES - 5 + 7, K)
    np.testing.assert_array_equal(spliced[:10], engine.codes[:10])
    np.testing.assert_array_equal(spliced[10:17], np.full((7, K), 900))
    np.testing.assert_array_equal(spliced[17:], engine.codes[15:])

    assert out["original_text"] is None and out["words"] is None and out["contexts"] is None
    assert out["spans"] == [[1.0, 1.5]] and out["expected_seconds"] == [0.5]
    (take,) = out["takes"]
    assert take["seed"] == 5 and take["stopped"] == [True] and take["verify"] is None
    assert take["edited_regions"] == [[1.0, 1.7]] and take["fill_seconds"] == [pytest.approx(0.7)]
    assert take["duration"] == pytest.approx((FRAMES + 2) / FRAME_RATE)


def test_raw_controls_need_the_speaker_row():
    engine = FakeEngine(raw={"voice.deeper"})
    runtime = EditRuntime(engine, FakeAsr(), FakeAligner())
    with pytest.raises(ValueError, match="voice.deeper move the speaker embedding"):
        edit(runtime, spans=[[1.0, 1.5]], controls={"voice.deeper": 1.0}, speaker_embedding=False)
    no_row = FakeEngine(model=EditModel(**{**MODEL.__dict__, "speaker_row": False}), raw={"voice.deeper"})
    with pytest.raises(ValueError, match="need the speaker row"):
        edit(EditRuntime(no_row, FakeAsr(), FakeAligner()), spans=[[1.0, 1.5]], controls={"voice.deeper": 1.0})
    assert not engine.jobs and not no_row.jobs


def test_region_controls():
    runtime = EditRuntime(FakeEngine(), FakeAsr(), FakeAligner())
    with pytest.raises(ValueError, match="region_controls needs explicit spans"):
        edit(runtime, region_controls=[{"a.b": 1.0}])
    with pytest.raises(ValueError, match="region_controls has 1 entries for 2 spans"):
        edit(runtime, spans=[[0.5, 1.0], [2.0, 2.5]], region_controls=[{"a.b": 1.0}])

    engine = FakeEngine()
    edit(EditRuntime(engine, FakeAsr(), FakeAligner()), spans=[[0.5, 1.0], [2.0, 2.5]], min_fill_ratio=0.0,
         controls={"a.b": 1.0}, region_controls=[{"a.b": -1.0}, {"c.d": 0.5}])
    assert engine.checked == [[{}, {"a.b": 1.0, "c.d": 0.5}]]
    assert [job.controls for job in engine.jobs] == [{}, {"a.b": 1.0, "c.d": 0.5}]
    assert [len(job.fills) for job in engine.jobs] == [0, 1]  # region 1 sees region 0's fill


def test_verify_reports_the_take_transcript_and_words():
    aligner = FakeAligner(enabled=True)
    out = edit(EditRuntime(FakeEngine(), FakeAsr(heard="hello brave new world"), aligner), spans=[[1.0, 1.5]],
               min_fill_ratio=0.0, verify=True)
    (take,) = out["takes"]
    assert take["verify"] == {"asr_text": "hello brave new world", "wer": 0.0, "cer": 0.0}
    assert [w["word"] for w in take["words"]] == ["hello", "brave", "new", "world"]
    assert set(take["words"][0]) == {"word", "start", "end", "confidence", "tokens"}
    assert len(aligner.aligned) == 1  # the take only: explicit spans never align the original


def test_status_of():
    assert status_of(UnknownId("x")) == 404
    assert status_of(AlignWorkerError("down")) == 503
    assert status_of(AsrError("down")) == 503 and status_of(FuseAlignError("down")) == 503
    assert status_of(ValueError("bad")) == 400
    assert status_of(RuntimeError("failed")) == 500
    assert status_of(KeyError("other")) is None
