"""Server-hosted dub requests (zonos3.dub.runtime): voices, per-segment controls and their merging, regenerate's
inheritance, ref_words, pronunciation groups. Synthesis, the codec and the alignment worker are faked."""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from zonos3.dub import plan as P
from zonos3.dub import runtime as R

EDGES = (10.117, 12.123, 13.286, 14.25, 15.14, 16.079, 17.094, 18.489, 21.208)


class FakeAligner:
    """The server's FuseAlign worker: every token at 0.1-0.9 s, placed."""

    def __init__(self):
        self.items: list[dict] = []

    def model(self) -> str:
        return "Zyphra/fusealign/fake.safetensors"

    def align(self, items):
        self.items += items
        return [[{"word": t, "start": 0.1, "end": 0.9, "presence": 1.0, "placed": True} for t in it["tokens"]]
                for it in items]


class Fake:
    def __init__(self, source_voice=True, studio_voices=True, rate_scales=None):
        self.requests: list[list[dict]] = []
        self.sessions: list[str | None] = []
        self.encoded = 0
        model = R.DubModel(name="m", rate_edges=EDGES, n_buckets=10, laci_languages=frozenset({"fr_fr", "en_us"}),
                           studio_voices=studio_voices, source_voice=source_voice,
                           raw_controls=frozenset({"identity.gender"}), phoneme_groups=True,
                           rate_scales=rate_scales or {})
        self.aligner = FakeAligner()
        self.rt = R.DubRuntime(model, self.synthesize, self.encode_voice, self.aligner,
                               R.WorkerSettings(python="/nonexistent"))

        async def call(op, **payload):
            raise AssertionError(f"unexpected alignment-worker call {op}")

        self.rt._call = call
        sr = P.SAMPLE_RATE
        audio = (0.1 * np.sin(np.arange(4 * sr) / sr * 2 * math.pi * 220)).astype(np.float32)
        self.ref_id = self.rt.refs.put({"audio": audio, "sr": sr, "duration": 4.0})

    async def synthesize(self, jobs, session_id):
        self.requests.append(jobs)
        self.sessions.append(session_id)
        take = np.full(P.SAMPLE_RATE, 0.05, dtype=np.float32)
        record = {"attempts": 1, "failures": [], "capped": False, "end": "eos", "rate_bucket": 3}
        return [(take, {**record, "rate_bucket": j["bucket"]}) for j in jobs]

    async def encode_voice(self, audio, sr):
        self.encoded += 1
        return [[1] * 10] * 3

    def generate(self, **req):
        body = {"ref_id": self.ref_id, "target_lang": "fr", "seed": 0, "lipsync": False, **req}
        return asyncio.run(self.rt.generate(body, "session-1"))

    def regenerate(self, **req):
        body = {"seed": 5, "bucket": None, "lipsync": False, "voice": None, "controls": None, **req}
        return asyncio.run(self.rt.regenerate(body, None))


STUDIO = {"mode": "studio", "speaker_embedding_id": "voice-a", "controls": {"identity.gender": 1.0}}


def segments(*voices_controls):
    return [{"ref_start": 0.5 + 1.5 * k, "ref_end": 1.5 + 1.5 * k, "target_text": f"Bonjour numéro {k}.",
             "bucket": None, "voice": v, "controls": c} for k, (v, c) in enumerate(voices_controls)]


def test_generate_per_segment_voices_and_controls_reach_the_takes_and_the_rows():
    f = Fake()
    out = f.generate(voice=STUDIO, segments=segments((None, None), ({"mode": "source"}, {"emotions.angry": 2})))
    jobs = f.requests[0]
    by_segment = {}
    for j in jobs:
        by_segment.setdefault(j["text"], []).append(j)
    a, b = by_segment["Bonjour numéro 0."], by_segment["Bonjour numéro 1."]
    assert len(a) == len(b) == len(P.CANDIDATE_BUCKET_OFFSETS) * P.CANDIDATE_SEEDS
    assert {j["voice"]["mode"] for j in a} == {"studio"} and a[0]["voice"]["speaker_embedding_id"] == "voice-a"
    assert a[0]["controls"] == {"identity.gender": 1.0} and a[0]["ref_codes"] is None
    assert {j["voice"]["mode"] for j in b} == {"source"} and b[0]["controls"] == {"emotions.angry": 2.0}
    assert b[0]["ref_codes"] == [[1] * 10] * 3 and f.encoded == 1
    assert all(j["language"] == "fr_fr" and j["target_frames"] == pytest.approx(15.0) for j in jobs)
    assert all(j["adjust_rate"] for j in jobs) and f.sessions == ["session-1"]
    rows = out["segments"]
    assert [r["k"] for r in rows] == [0, 1]
    assert rows[0]["voice"]["mode"] == "studio" and rows[0]["controls"] is None
    assert rows[1]["voice"] == {"mode": "source", "controls": None} and rows[1]["controls"] == {"emotions.angry": 2.0}
    assert rows[0]["fit"] == pytest.approx(0.8) and rows[0]["start"] is not None and rows[0]["final_bucket"] is not None


def test_studio_only_dub_encodes_no_voice_prompt():
    f = Fake()
    f.generate(voice=STUDIO, segments=segments((None, None)))
    assert f.encoded == 0


def test_speaker_embedding_controls_need_a_studio_voice():
    f = Fake()
    with pytest.raises(ValueError, match="identity.gender .*studio voice"):
        f.generate(segments=segments((None, {"identity.gender": 1.0})))
    with pytest.raises(ValueError, match="studio voice"):
        f.generate(voice={"mode": "source", "controls": {"identity.gender": 1.0}}, segments=segments((None, None)))
    assert f.requests == []  # refused before any take was queued


def test_voice_modes_follow_the_checkpoint():
    with pytest.raises(ValueError, match="no speaker-embedding row"):
        Fake(studio_voices=False).generate(voice=STUDIO, segments=segments((None, None)))
    with pytest.raises(ValueError, match="no reference-clone path"):
        Fake(source_voice=False).generate(segments=segments((None, None)))


def test_regenerate_keeps_the_segment_voice_and_replaces_or_clears_its_controls():
    f = Fake()
    first = f.generate(voice=STUDIO, segments=segments((None, None), ({"mode": "source"}, {"emotions.angry": 2})))
    second = f.regenerate(dub_id=first["dub_id"], segment=1, target_text="Encore.", controls={"emotions.angry": 4})
    jobs = f.requests[-1]
    assert {j["text"] for j in jobs} == {"Encore."}
    assert jobs[0]["voice"]["mode"] == "source" and jobs[0]["controls"] == {"emotions.angry": 4.0}
    assert f.encoded == 1  # the dub's voice prompt is reused
    assert second["segments"][1]["controls"] == {"emotions.angry": 4.0}
    assert second["segments"][0] == {**first["segments"][0], **{k: second["segments"][0][k] for k in
                                     ("speech_start", "speech_end", "start", "end", "shift", "start_error",
                                      "end_error", "duration_error")}}

    third = f.regenerate(dub_id=second["dub_id"], segment=1, target_text="Encore.", controls={})
    assert f.requests[-1][0]["controls"] is None and third["segments"][1]["controls"] is None
    # a new voice for the segment: the voice's controls apply, the segment's own ones are kept
    fourth = f.regenerate(dub_id=second["dub_id"], segment=1, target_text="Encore.",
                          voice={"mode": "studio", "speaker_embedding_id": "voice-b", "controls": {"emotions.angry": 1}})
    job = f.requests[-1][0]
    assert job["voice"]["speaker_embedding_id"] == "voice-b" and job["controls"] == {"emotions.angry": 5.0}
    assert fourth["segments"][1]["voice"]["speaker_embedding_id"] == "voice-b"


def test_regenerate_switching_a_studio_dub_segment_to_source_encodes_the_prompt_then():
    f = Fake()
    first = f.generate(voice=STUDIO, segments=segments((None, None), (None, None)))
    assert f.encoded == 0
    f.regenerate(dub_id=first["dub_id"], segment=0, target_text="Oui.", voice={"mode": "source"})
    assert f.encoded == 1 and f.requests[-1][0]["ref_codes"] == [[1] * 10] * 3
    assert f.requests[-1][0]["controls"] is None  # the studio voice's raw control left with it


def test_merge_controls_sums_per_id_and_drops_zero():
    assert R.merge_controls({"a": 1, "b": 2}, {"b": -2, "c": 0.5}) == {"a": 1.0, "c": 0.5}
    assert R.merge_controls(None, {}) is None
    with pytest.raises(ValueError, match="finite"):
        R.merge_controls({"a": float("nan")})


def test_normalize_voice():
    assert R.normalize_voice(None, "voice") is None
    assert R.normalize_voice({"mode": "source", "speaker_embedding_id": None, "controls": None}, "v") == \
        {"mode": "source", "controls": None}
    with pytest.raises(ValueError, match="speaker_embedding_id"):
        R.normalize_voice({"mode": "studio"}, "v")
    with pytest.raises(ValueError, match="needs mode 'studio'"):
        R.normalize_voice({"mode": "source", "speaker_embedding_id": "x"}, "v")


def test_ref_words_are_checked_and_drive_the_segments():
    f = Fake()

    async def call(op, **payload):
        assert op == "word_align", op  # ref_words: no forced alignment of the reference
        return {"links": [[0, 0], [1, 1], [2, 2], [3, 3]]}

    f.rt._call = call
    words = [{"word": "hello", "start": 0.2, "end": 0.6}, {"word": "there", "start": 0.7, "end": 1.1},
             {"word": "good", "start": 2.0, "end": 2.4}, {"word": "night", "start": 2.5, "end": 3.0}]
    out = asyncio.run(f.rt.analyze({"ref_id": f.ref_id, "ref_words": words, "ref_lang": "en",
                                    "target_text": "salut toi bonne nuit", "target_lang": "fr", "min_pause_s": 0.3}))
    assert [(s["ref_text"], s["target_text"]) for s in out["segments"]] == [("hello there", "salut toi"),
                                                                           ("good night", "bonne nuit")]
    assert out["segments"][0]["bucket_auto"] == P.RateScale(EDGES).bucket(len(b"salut toi") / 0.9)
    with pytest.raises(ValueError, match="time order"):
        R.normalize_ref_words([words[1], words[0]], 4.0)
    with pytest.raises(ValueError, match="start <= end"):
        R.normalize_ref_words([{"word": "x", "start": 1.0, "end": 9.0}], 4.0)


FR_SCALE = P.RateScale.extended(EDGES, (9.2, 8.4), (23.3, 25.5), {-1: -0.99, 10: 1.44, 11: 1.93})
FR_FASTER = P.RateScale.extended(EDGES, (9.2, 8.4), (23.3, 25.5), {10: 1.44, 11: 1.93})  # source voices: faster side


def test_extended_scale_plans_retakes_and_lists_the_steered_buckets():
    f = Fake(rate_scales={"studio": {"fr_fr": FR_SCALE}, "source": {"fr_fr": FR_FASTER}})
    fast = "Bonjour, ceci est une phrase française bien trop longue."  # 57 bytes in 1 s: bucket 11
    f.generate(segments=[{"ref_start": 0.5, "ref_end": 1.5, "target_text": fast, "bucket": None}])
    jobs = f.requests[0]
    assert sorted({j["bucket"] for j in jobs}) == [9, 10, 11]  # offsets 0, -1, -2 on the scale; +1, +2 leave it
    assert all(j["rate_scale"] is FR_FASTER and j["adjust_rate"] for j in jobs)  # the source voice's scale
    f.generate(target_lang="en", segments=[{"ref_start": 0.5, "ref_end": 1.5, "target_text": fast, "bucket": None}])
    en = f.requests[1]
    assert sorted({j["bucket"] for j in en}) == [7, 8, 9] and all(j["rate_scale"] is None for j in en)
    slow = [{"ref_start": 0.5, "ref_end": 2.5, "target_text": "Oui.", "bucket": None}]  # 2 B/s: the lowest bucket
    f.generate(segments=slow)
    assert sorted({j["bucket"] for j in f.requests[2]}) == [0, 1, 2]
    f.generate(voice=STUDIO, segments=slow)
    assert sorted({j["bucket"] for j in f.requests[3]}) == [-1, 0, 1]  # a studio voice plans in the slower bucket
    assert all(j["rate_scale"] is FR_SCALE for j in f.requests[3])
    out = f.generate(voice=STUDIO, segments=[{"ref_start": 0.5, "ref_end": 1.5, "target_text": "Oui.", "bucket": -1}])
    assert {j["bucket"] for j in f.requests[4]} == {-1} and not f.requests[4][0]["adjust_rate"]
    assert out["segments"][0]["bucket"] == -1 and out["segments"][0]["final_bucket"] == -1
    with pytest.raises(ValueError, match=r"bucket must be in \[0, 11\]"):  # no slower steered bucket for a source voice
        f.generate(segments=[{"ref_start": 0.5, "ref_end": 1.5, "target_text": "Oui.", "bucket": -1}])
    with pytest.raises(ValueError, match=r"bucket must be in \[-1, 11\]"):
        f.generate(voice=STUDIO, segments=[{"ref_start": 0.5, "ref_end": 1.5, "target_text": "Oui.", "bucket": 12}])
    with pytest.raises(ValueError, match=r"bucket must be in \[0, 9\]"):
        f.generate(target_lang="en", segments=[{"ref_start": 0.5, "ref_end": 1.5, "target_text": "Oui.", "bucket": 10}])
    buckets = f.rt.capabilities()["buckets"]
    assert [b["index"] for b in buckets] == list(range(-1, 12))
    by = {b["index"]: b for b in buckets}
    assert by[5] == {"index": 5, "label": "5", "rate": pytest.approx((15.14 + 16.079) / 2), "low": 15.14,
                     "high": 16.079, "steered": False, "languages": ["en", "fr"], "voice_modes": ["source", "studio"]}
    assert by[-1]["steered"] and by[-1]["languages"] == ["fr"] and by[-1]["doses"] == {"fr": -0.99}
    assert by[-1]["voice_modes"] == ["studio"] and by[10]["voice_modes"] == ["source", "studio"]
    assert (by[-1]["low"], by[-1]["high"], by[0]["low"]) == (None, 9.2, 9.2)  # open below; bucket 0 bounded by fr's
    assert by[0]["rate"] == pytest.approx((9.2 + 10.117) / 2)
    assert (by[9]["high"], by[11]["high"]) == (23.3, None)  # en's 9 is open above, fr's is not
    assert by[11]["rate"] == pytest.approx(25.5 + (25.5 - 23.3) / 2)  # the open top end
    assert [b["index"] for b in Fake().rt.capabilities()["buckets"]] == list(range(10))


def test_pronunciation_groups_read_as_ipa_and_count_letters():
    text = "Hi \ue000ʃɪvɑn\ue001!"
    assert R.spoken_text(text) == "Hi ʃɪvɑn!"
    assert R.text_bytes(text) == round(len(b"Hi ") + 1.4 * 5 + 1)


def test_edit_worker_language_maps_dub_languages_and_refuses_the_rest():
    from zonos3.tts.edit_align import mms_language

    assert [mms_language(c) for c in ("en_us", "fr_fr", "pt_br", "cmn")] == ["en", "fr", "pt", "zh"]
    with pytest.raises(ValueError, match="does not serve language 'ar'"):
        mms_language("ar")


def test_takes_and_reference_are_aligned_by_fusealign_tokens():
    f = Fake()
    f.generate(segments=[{"ref_start": 0.5, "ref_end": 1.5, "target_text": "Hi \ue000ʃɪvɑn\ue001 <laugh/>!", "bucket": None}])
    assert f.aligner.items and all(it["tokens"] == ["Hi", "ʃɪvɑn", "!"] and it["language"] == "fr"
                                   for it in f.aligner.items)

    async def call(op, **payload):
        assert op == "word_align", op
        return {"links": [[0, 0]]}

    f.rt._call = call
    f.aligner.items.clear()
    out = asyncio.run(f.rt.analyze({"ref_id": f.ref_id, "ref_text": "我们 去公园。", "ref_lang": "zh",
                                    "target_text": "On y va.", "target_lang": "fr", "min_pause_s": 0.3}))
    assert f.aligner.items[0]["tokens"] == ["我", "们", "去", "公", "园。"] and f.aligner.items[0]["language"] == "zh"
    assert out["ref_words"][0] == {"word": "我", "start": 0.1, "end": 0.9}
    caps = f.rt.capabilities()
    assert caps["aligner_model"] == "Zyphra/fusealign/fake.safetensors"
    assert [lang["code"] for lang in caps["ref_languages"]] == list(P.REF_LANGUAGES)


@pytest.mark.parametrize("after,applied", [(0.15, True), (0.18, False)])
def test_lip_warp_is_kept_only_with_a_sync_gain(after, applied):
    import soundfile as sf

    f = Fake()
    f.rt.refs.get(f.ref_id)["lips"] = {"path": "lips.pt", "start": 0.0, "end": 4.0}

    def call_sync(op, **payload):
        assert op == "lips_sync" and payload["max_rate"] == P.LIP_GUARD_MAX_RATE
        x, sr = sf.read(payload["audio"], dtype="float32")
        sf.write(payload["out"], np.concatenate([np.zeros(sr // 10, np.float32), x]), sr)  # a 0.1 s delay
        return {"start": 0.0, "end": 4.0, "rate": 62.5, "src_frames": (np.arange(250) - 6.25).tolist(),
                "sync_distance_before": 0.16, "sync_distance_after": after}

    f.rt._call_sync = call_sync
    out = f.generate(voice=STUDIO, segments=segments((None, None)), lipsync=True)
    row = out["segments"][0]
    assert out["lipsync"]["applied"] is applied
    if applied:
        assert row["lip_start"] == pytest.approx(row["speech_start"] + 0.1, abs=1e-6)
    else:
        assert "no sync gain" in out["lipsync"]["reason"]
        assert row["lip_start"] == row["speech_start"] and row["lip_stretch"] == 1.0 and row["lip_scale"] == 1.0


def test_server_take_on_an_extended_scale_carries_its_bucket_and_deltas(monkeypatch):
    """A take in a steered bucket: rate row unknown, the bucket's tempo dose on top of its own steering delta, and
    the scale with its bucket on it (pinned takes too: LACI reports the bucket from it, not the unknown row id)."""
    from types import SimpleNamespace

    import torch

    from zonos3_sgl.message.tts import TTSSamplingParams, TTSTokenizeMsg
    from zonos3_sgl.server import api_server as A

    monkeypatch.setattr(A, "_steer_bundle", lambda config: SimpleNamespace(rate_extension=SimpleNamespace(direction="d")))
    monkeypatch.setattr(A, "_steering_sources", lambda config: (0,))
    monkeypatch.setattr(A, "resolve_controls", lambda bundle, controls, spans, **kw: SimpleNamespace(
        hidden_delta=torch.full((4,), controls["d"])))
    config = SimpleNamespace(model_config=SimpleNamespace(hidden_size=4))
    job = {"text": "Oui.", "language": "fr_fr", "seed": 1, "max_tokens": 90, "controls": None, "ref_codes": None,
           "voice": {"mode": "studio", "speaker_embedding_id": "v"}, "rate_scale": FR_SCALE}
    assert A._dub_take_request({**job, "bucket": -1}).condition_buckets == {"speaking_rate": None}
    assert A._dub_take_request({**job, "bucket": 5}).condition_buckets == {"speaking_rate": 5}
    assert A._dub_take_request({**job, "bucket": 5, "rate_scale": None}).condition_buckets == {"speaking_rate": 5}
    for adjust in (True, False):
        base = torch.ones(4)
        msg = TTSTokenizeMsg(uid=1, text="Oui.", sampling_params=TTSSamplingParams(), speaker_emotion_delta=base,
                             laci_adjust_rate=adjust)
        A._dub_take_rate_steering(config, msg, {**job, "bucket": -1})
        assert (msg.laci_rate_first, msg.laci_rate_bucket, msg.laci_rate_edges) == (-1, -1, list(FR_SCALE.edges))
        assert torch.allclose(msg.speaker_emotion_delta, torch.full((4,), 1 - 0.99))
        assert len(msg.laci_rate_deltas) == 13 and msg.laci_rate_deltas[1 + 5] is base  # row buckets: own delta only
        assert torch.allclose(msg.laci_rate_deltas[1 + 11], torch.full((4,), 1 + 1.93))
    row_only = TTSTokenizeMsg(uid=2, text="Oui.", sampling_params=TTSSamplingParams())
    A._dub_take_rate_steering(config, row_only, {**job, "bucket": 5, "rate_scale": None})
    assert row_only.laci_rate_edges is None and row_only.speaker_emotion_delta is None
