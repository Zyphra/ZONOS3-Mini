"""zonos3.tts.edit_runtime.EditRuntime over a fake EditEngine and fake ASR / FuseAlign workers (no model, no
subprocess): what a /v2/edit caller sees, with and without speech recognition."""

import asyncio
import base64

import numpy as np
import pytest
from zonos3.generation import SAMPLE_RATE
from zonos3.server.errors import NotFound, status_of
from zonos3.server.schemas import EditAlignRequest, EditRequest, EditTranscribeRequest
from zonos3.tts.edit_align import AsrError, AsrUnavailable, AsrWorker, FuseAlignError, WorkerError
from zonos3.tts.edit_runtime import EditModel, EditRuntime, edit_capabilities

K = 4
FRAME_RATE = 10.0
FRAMES = 30  # a 3 s clip
MODEL = EditModel(
    frame_rate=FRAME_RATE, n_codebooks=K, audio_pad_id=1024, codebook_sizes=(1024,) * K, max_spans=4, speaker_row=True
)
CLIP = base64.b64encode(b"an upload").decode("ascii")


class FakeEngine:
    """Codes 0.. of a 3 s clip; region k's fill is ``fill_frames`` frames of the code 900 + k."""

    def __init__(self, model: EditModel = MODEL, raw: set[str] = frozenset(), fill_frames: int = 7):
        self.model = model
        self.raw = set(raw)
        self.fill_frames = fill_frames
        self.codes = np.arange(FRAMES * K, dtype=np.int64).reshape(FRAMES, K)
        self.checked, self.jobs, self.decoded, self.encoded = [], [], [], 0
        self.decoders = []  # the decoder of every decode

    async def decode_audio(self, audio):
        assert audio == b"an upload"
        return np.zeros(int(FRAMES / FRAME_RATE * 16000), dtype=np.float32), 16000

    async def embed(self, audio):
        return [0.25] * 8

    async def encode(self, audio, max_seconds):
        self.encoded += 1
        return self.codes

    async def check(self, target_text, doses, condition_buckets, condition_values, steer):
        self.checked.append(doses)
        self.steer = steer
        return self.raw

    async def fill(self, job):
        self.jobs.append(job)
        # delay-stacked rows: the fill's frames plus the K-1 rows of its delay tail, ended on span-end at frame n
        n = self.fill_frames
        return np.full((n + K - 1, K), 900 + len(self.jobs) - 1, dtype=np.int64), n

    async def decode(self, codes, decoder):
        self.decoded.append(np.asarray(codes))
        self.decoders.append(decoder)
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
        return [
            [
                {"word": t, "start": 0.2 * i, "end": 0.2 * i + 0.15, "presence": 0.9, "placed": True}
                for i, t in enumerate(item["tokens"])
            ]
            for item in items
        ]


def edit(runtime: EditRuntime, **body) -> dict:
    return asyncio.run(
        runtime.edit(EditRequest(audio_base64=CLIP, target_text="hello brave new world", seed=5, **body))
    )


def test_unedited_frames_survive_the_splice_and_spans_skip_transcription():
    engine = FakeEngine(fill_frames=7)
    out = edit(EditRuntime(engine, FakeAsr(), FakeAligner()), spans=[[1.0, 1.5]], min_fill_ratio=0.0)

    (job,) = engine.jobs
    assert job.spans == [[10, 5]] and job.min_frames == 0 and job.seed == 5
    np.testing.assert_array_equal(job.window_codes, engine.codes)
    assert job.speaker_embedding == [0.25] * 8 and job.controls is None
    (spliced,) = engine.decoded
    assert engine.decoders == [None]  # no decoder: the server default
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


def test_the_edit_decodes_through_its_decoder():
    engine = FakeEngine(fill_frames=7)
    edit(EditRuntime(engine, FakeAsr(), FakeAligner()), spans=[[1.0, 1.5]], min_fill_ratio=0.0, decoder="full_mr")
    assert engine.decoders == ["full_mr"]


def test_raw_controls_need_the_speaker_row():
    engine = FakeEngine(raw={"voice.deeper"})
    runtime = EditRuntime(engine, FakeAsr(), FakeAligner())
    with pytest.raises(ValueError):
        edit(runtime, spans=[[1.0, 1.5]], controls={"voice.deeper": 1.0}, speaker_embedding=False)
    no_row = FakeEngine(model=EditModel(**{**MODEL.__dict__, "speaker_row": False}), raw={"voice.deeper"})
    with pytest.raises(ValueError):
        edit(EditRuntime(no_row, FakeAsr(), FakeAligner()), spans=[[1.0, 1.5]], controls={"voice.deeper": 1.0})
    assert not engine.jobs and not no_row.jobs


def test_region_controls():
    runtime = EditRuntime(FakeEngine(), FakeAsr(), FakeAligner())
    with pytest.raises(ValueError):
        edit(runtime, region_controls=[{"a.b": 1.0}])
    with pytest.raises(ValueError):
        edit(runtime, spans=[[0.5, 1.0], [2.0, 2.5]], region_controls=[{"a.b": 1.0}])

    engine = FakeEngine()
    edit(
        EditRuntime(engine, FakeAsr(), FakeAligner()),
        spans=[[0.5, 1.0], [2.0, 2.5]],
        min_fill_ratio=0.0,
        controls={"a.b": 1.0},
        region_controls=[{"a.b": -1.0}, {"c.d": 0.5}],
    )
    assert engine.checked == [[{}, {"a.b": 1.0, "c.d": 0.5}]]
    assert [job.controls for job in engine.jobs] == [{}, {"a.b": 1.0, "c.d": 0.5}]
    assert [len(job.fills) for job in engine.jobs] == [0, 1]  # region 1 sees region 0's fill


def test_verify_reports_the_take_transcript_and_words():
    aligner = FakeAligner(enabled=True)
    out = edit(
        EditRuntime(FakeEngine(), FakeAsr(heard="hello brave new world"), aligner),
        spans=[[1.0, 1.5]],
        min_fill_ratio=0.0,
        verify=True,
    )
    (take,) = out["takes"]
    assert take["verify"] == {"asr_text": "hello brave new world", "wer": 0.0, "cer": 0.0}
    assert [w["word"] for w in take["words"]] == ["hello", "brave", "new", "world"]
    assert set(take["words"][0]) == {"word", "start", "end", "confidence", "tokens"}
    assert len(aligner.aligned) == 1  # the take only: explicit spans never align the original


def test_literal_angle_text_survives_editing_and_verification():
    original = "hello <laugh /> old world"
    target = "hello <laugh /> new world"
    runtime = EditRuntime(FakeEngine(), FakeAsr(heard="hello laugh new world"), FakeAligner(enabled=True))
    out = asyncio.run(
        runtime.edit(
            EditRequest(
                audio_base64=CLIP,
                original_text=original,
                target_text=target,
                verify=True,
                min_fill_ratio=0.0,
                seed=5,
            )
        )
    )
    assert out["original_text"] == original
    (take,) = out["takes"]
    assert take["verify"] == {"asr_text": "hello laugh new world", "wer": 0.0, "cer": 0.0}


ASR_DETAIL = "Speech recognition is not available on this server: "
NO_ASR = [None, AsrWorker(python="/nonexistent/.venv-asr/bin/python")]  # no worker; worker not installed


def test_edit_capabilities():
    on = edit_capabilities(MODEL, True, True)
    assert on == {
        "speech_editing": True,
        "edit_transcribe": True,
        "edit_max_spans": 4,
        "edit_frame_rate": FRAME_RATE,
        "edit_max_seconds": on["edit_max_seconds"],
        "edit_align_max_seconds": on["edit_align_max_seconds"],
    }
    assert edit_capabilities(MODEL, True, False)["edit_transcribe"] is False
    assert edit_capabilities(MODEL, True, False)["speech_editing"] is True
    off = edit_capabilities(MODEL, False, True)
    assert off["speech_editing"] is False and off["edit_transcribe"] is False
    plain = edit_capabilities(EditModel(**{**MODEL.__dict__, "max_spans": 0}), True, True)
    assert (plain["speech_editing"], plain["edit_transcribe"], plain["edit_max_spans"]) == (False, False, 0)


@pytest.mark.parametrize("asr", NO_ASR)
def test_without_asr_recognition_is_refused_before_any_model_work(asr):
    engine = FakeEngine()
    runtime = EditRuntime(engine, asr, FakeAligner())
    assert not runtime.transcribes
    with pytest.raises(AsrUnavailable) as exc:
        asyncio.run(runtime.transcribe(EditTranscribeRequest(audio_base64=CLIP)))
    assert str(exc.value) == ASR_DETAIL + "send the transcript to /v2/edit/align instead"
    assert status_of(exc.value) == 503
    with pytest.raises(AsrUnavailable) as exc:
        edit(runtime, spans=[[1.0, 1.5]], min_fill_ratio=0.0, verify=True)
    assert str(exc.value) == ASR_DETAIL + "set verify to false"
    with pytest.raises(AsrUnavailable) as exc:
        edit(runtime)  # no spans and no original_text: the clip's words would be transcribed
    assert str(exc.value) == ASR_DETAIL + "send the clip's transcript as original_text"
    assert engine.encoded == 0 and not engine.jobs and not engine.decoded and not engine.checked


@pytest.mark.parametrize("asr", NO_ASR)
def test_without_asr_align_and_edit_with_the_transcript(asr):
    aligner = FakeAligner(enabled=True)
    runtime = EditRuntime(FakeEngine(), asr, aligner)
    out = asyncio.run(runtime.align(EditAlignRequest(audio_base64=CLIP, text="hello brave old world")))
    assert [w["word"] for w in out["words"]] == ["hello", "brave", "old", "world"]
    assert "utterances" not in aligner.aligned[0]

    engine = FakeEngine()
    out = edit(EditRuntime(engine, asr, aligner), original_text="hello brave old world")
    assert out["original_text"] == "hello brave old world" and [w["word"] for w in out["words"]][2] == "old"
    assert len(engine.jobs) == 1 and len(out["spans"]) == 1
    assert (
        edit(EditRuntime(FakeEngine(), asr, FakeAligner()), spans=[[1.0, 1.5]], min_fill_ratio=0.0)["takes"][0][
            "verify"
        ]
        is None
    )  # explicit spans need neither worker's words


def test_language_is_checked_without_asr():
    class SilentAsr(FakeAsr):
        def languages(self):
            raise AssertionError("the edit pinged the ASR worker")

    runtime = EditRuntime(FakeEngine(), SilentAsr(), FakeAligner())
    assert runtime.language("en_us")[0] == "en"
    with pytest.raises(ValueError):
        runtime.language("xx_yy")
    edit(runtime, spans=[[1.0, 1.5]], min_fill_ratio=0.0)  # explicit spans: no recognition, no ping


def test_recognition_languages_come_from_the_asr_worker():
    engine = FakeEngine()  # FakeAsr transcribes English only
    with pytest.raises(ValueError):
        edit(EditRuntime(engine, FakeAsr(heard="x"), FakeAligner()), spans=[[1.0, 1.5]], verify=True, language="de")
    assert engine.encoded == 0  # refused before the encode
    edit(EditRuntime(engine, FakeAsr(), FakeAligner()), spans=[[1.0, 1.5]], min_fill_ratio=0.0, language="de")


def test_status_of():
    assert status_of(NotFound("x")) == 404
    assert status_of(WorkerError("down")) == 503
    assert status_of(AsrError("down")) == 503 and status_of(FuseAlignError("down")) == 503
    assert status_of(ValueError("bad")) == 400
    assert status_of(RuntimeError("failed")) == 500
    assert status_of(KeyError("other")) is None
