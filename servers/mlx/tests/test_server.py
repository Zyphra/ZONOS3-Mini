"""The MLX HTTP server on a tiny backbone (servers/ggml/scripts/make-dummy-model.py with edit rows, span-end and a
phoneme block) and the dummy decoder: what a client sees of streaming, speech editing, phonemize, session voices,
steering-bundle calibrations and the CFG rules. The reference encoder is replaced by the fixture codes (the edit test
needs codes, not audio)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import subprocess
import sys

import lmdb
import numpy as np
import pytest
import soundfile as sf
import torch
from conftest import FIXTURES, REPO
from fastapi.testclient import TestClient
from zonos3.server.schemas import GenerateRequest
from zonos3.tts.audio import pcm16_samples
from zonos3.tts.phonemes import normalize_headword
from zonos3.tts.plan import Speaker
from zonos3.tts.steering import controls_document, load_bundle
from zonos3_mlx.engine import Engine, plan_request
from zonos3_mlx.scheduler import Scheduler
from zonos3_mlx.server import speakers
from zonos3_mlx.server.app import create_app
from zonos3_mlx.server.state import ServerOptions, ServerState

HOP = 2940
PHONES = "aɪb"
CODES = np.load(FIXTURES / "dummy-zac-codes.npy").astype(np.int32)  # [40, 10]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("srv")
    subprocess.run([sys.executable, str(REPO / "servers/ggml/scripts/make-dummy-model.py"), "-o", str(root / "m.gguf"),
                    "--edit-markers", "3", "--span-end", "--phonemes", PHONES], check=True, capture_output=True)
    (root / "zac-keep2.gguf").write_bytes((FIXTURES / "dummy-zac.gguf").read_bytes())
    lexicon = root / "lexicon.lmdb"
    env = lmdb.open(str(lexicon), map_size=1 << 24, max_dbs=8, subdir=True)
    with env.begin(write=True) as txn:
        txn.put(b"meta", json.dumps({"version": 1, "languages": {"eng": {"mode": "words"}}}).encode())
    db = env.open_db(b"eng")
    with env.begin(write=True, db=db) as txn:
        for word, ipa in {"hello": "hɛloʊ", "abba": "aɪb"}.items():
            txn.put(normalize_headword(word).encode(), ipa.encode())
    env.close()
    return root


def make_client(files, batch: int) -> TestClient:
    from zonos3.tts.phonemes import PhonemeLexicon

    engine = Engine.load(files / "m.gguf")
    scheduler = Scheduler(engine, batch=batch)
    scheduler.start(warmup=False)
    state = ServerState(engine, scheduler, ServerOptions(model_path=str(files / "m.gguf"), batch=batch, max_frames=60))
    if batch > 1:
        state.lexicon = PhonemeLexicon(str(files / "lexicon.lmdb"))
    return TestClient(create_app(state))


@pytest.fixture(scope="module")
def client(files):
    with make_client(files, batch=2) as c:
        yield c


def wav_b64(seconds: float = 2.0) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(44100 * seconds), np.float32), 44100, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def test_studio_pages_are_no_store_and_assets_immutable(client, tmp_path, monkeypatch):
    (tmp_path / "assets").mkdir()
    for page in ("index", "studio", "lite"):
        (tmp_path / f"{page}.html").write_text(f'<div id="{page}"></div>')
    (tmp_path / "assets" / "index-abc.js").write_text("export {}")
    monkeypatch.setattr(client.app.state.zonos3.options, "studio_dir", tmp_path)
    # / is the chooser; /studio and /lite are their pages, with or without .html
    for path, page in (("/", "index"), ("/studio", "studio"), ("/studio.html", "studio"), ("/lite", "lite"), ("/lite.html", "lite")):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html"), path
        assert r.headers["cache-control"] == "no-store, max-age=0", path
        assert f'id="{page}"' in r.text, path
    r = client.get("/assets/index-abc.js")
    assert r.status_code == 200 and r.headers["cache-control"] == "public, max-age=31536000, immutable"
    for path in ("/nope.html", "/nope", "/studio/", "/assets/index-abc"):
        assert client.get(path).status_code == 404, path


def test_prefix_stream_is_the_whole_decode_without_the_silence_prefill(client):
    """A streamed request (prefix re-decode) carries the same audio as the buffered one; the buffered whole decode
    also emits the re-decoded silence-prefill frames (3 by default) in front."""
    body = {"text": "hello there", "seed": 4, "max_tokens": 40}
    with client.stream("POST", "/v2/generate", json=body) as r:
        assert r.status_code == 200 and r.headers["x-seed"] == "4"
        streamed = np.frombuffer(b"".join(r.iter_bytes()), np.float32)
    whole = np.frombuffer(client.post("/v2/generate", json={**body, "stream": False}).content, np.float32)
    assert len(streamed) > 0 and len(whole) == len(streamed) + 3 * HOP
    np.testing.assert_allclose(whole[3 * HOP:], streamed, atol=1e-5)


def test_request_limits_follow_the_shared_rules(client):
    """An explicit silence_prefill_frames outside 0..30 is refused (never clamped), inline reference codes keep their
    first 900 frames (the registry's reference_max_frames) and decode_context_frames 0 decodes the registry's 30."""
    r = client.post("/v2/generate", json={"text": "hi", "stream": False, "silence_prefill_frames": 31})
    assert r.status_code == 400 and r.json()["detail"] == "silence_prefill_frames must be 0..30"
    state = client.app.state.zonos3
    long = np.tile(CODES, (24, 1))  # 960 frames
    req = GenerateRequest(text="hi", speaker_mode="reference", speaker_reference_codes=long.tolist())
    assert torch.equal(asyncio.run(speakers.resolve_reference(state, None, req)), torch.from_numpy(long[:900]))
    speaker = Speaker("reference", reference_codes=torch.from_numpy(CODES))
    plan = plan_request(state.engine, {"text": "hi", "decode_context": True, "decode_context_frames": 0}, speaker)
    assert torch.equal(plan.context_codes, speaker.reference_codes[-30:])


def test_edit_keeps_the_unedited_audio_and_reports_the_fill(client, monkeypatch):
    async def fixture_codes(state, audio, max_seconds=60.0):
        return torch.from_numpy(CODES)

    monkeypatch.setattr(speakers, "encode_reference", fixture_codes)
    r = client.post("/v2/edit", json={"audio_base64": wav_b64(), "target_text": "hello world", "spans": [[1.0, 1.6]],
                                      "speaker_embedding": False, "seed": 7, "num_takes": 2, "min_fill_ratio": 0.0,
                                      "max_span_seconds": 2.0})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["spans"] == [[1.0, 1.6]] and [t["seed"] for t in d["takes"]] == [7, 8]
    client_app = client.app.state.zonos3
    original = client_app.engine.decoder.decode(CODES.astype(np.int64)) * np.float32(client_app.engine.output_gain(None))
    original = pcm16_samples(original)
    for take in d["takes"]:
        fill = round(take["fill_seconds"][0] * 15)
        assert take["edited_regions"] == [[1.0, (15 + fill) / 15]]
        assert take["duration"] == pytest.approx((len(CODES) - 9 + fill) / 15)
        pcm, sr = sf.read(io.BytesIO(base64.b64decode(take["audio_base64"])), dtype="int16")
        assert sr == 44100 and len(pcm) == round(take["duration"] * 15) * HOP
        # frames before the span are the original's (the frame next to it shares the iSTFT overlap with the fill)
        keep = 14 * HOP
        assert np.abs(pcm[:keep].astype(int) - original[:keep].astype(int)).max() <= 2


def test_edit_refuses_tag_cfg(client):
    r = client.post("/v2/edit", json={"audio_base64": wav_b64(), "target_text": "hi", "tag_cfg_scale": 2.0})
    assert r.status_code == 400 and "tag_cfg_scale" in r.json()["detail"]


class FakeAligner:
    """The server's FuseAlign worker: every token at 0.1-0.9 s with presence 0.5."""

    def __init__(self):
        self.items: list[dict] = []
        self.closed = False

    def model(self) -> str:
        return "Zyphra/fusealign/fake.safetensors"

    def align(self, items):
        self.items += items
        return [[{"word": t, "start": 0.1, "end": 0.9, "presence": 0.5, "placed": True} for t in it["tokens"]]
                for it in items]

    def close(self):
        self.closed = True


class FakeAsr:
    """The server's ASR worker: a fixed transcript for every item."""

    installed = True

    def __init__(self):
        self.items: list[dict] = []
        self.closed = False

    def model(self) -> str:
        return "Qwen/fake-asr"

    def languages(self) -> list[str]:
        return ["English", "French", "Chinese"]

    def transcribe(self, items):
        self.items += items
        return [{"text": "hello there world", "language": item["language"] or "English"} for item in items]

    def close(self):
        self.closed = True


def test_edit_transcribe_and_align_time_words_with_fusealign(client, monkeypatch):
    state = client.app.state.zonos3
    aligner, asr = FakeAligner(), FakeAsr()
    monkeypatch.setattr(state, "aligner", aligner)
    monkeypatch.setattr(state, "asr", asr)
    monkeypatch.setattr(state, "edit_runtime", None)
    r = client.post("/v2/edit/transcribe", json={"audio_base64": wav_b64(), "language": "fr_fr"})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "hello there world" and r.json()["duration"] == pytest.approx(2.0)
    assert [(w["word"], w["tokens"], w["confidence"]) for w in r.json()["words"]] == [
        ("hello", ["HELLO"], 0.5), ("there", ["THERE"], 0.5), ("world", ["WORLD"], 0.5)]
    assert [(it.get("start"), it["language"]) for it in asr.items] == [(None, "French")]  # 2 s: one whole-clip item
    assert aligner.items[0] == {"audio": aligner.items[0]["audio"], "tokens": ["hello", "there", "world"]}

    aligner.items.clear()
    r = client.post("/v2/edit/align", json={"audio_base64": wav_b64(), "text": "我们 去公园。", "language": "cmn"})
    assert r.status_code == 200, r.text
    assert aligner.items[0]["tokens"] == ["我", "们", "去", "公", "园。"] and "utterances" not in aligner.items[0]
    assert [w["word"] for w in r.json()["words"]] == ["我", "们", "去", "公", "园。"]
    assert len(asr.items) == 1  # aligning a short clip needs no recognition

    # transcription is filtered by the ASR worker's languages (FakeAsr has no Korean); alignment by the registry's
    r = client.post("/v2/edit/transcribe", json={"audio_base64": wav_b64(), "language": "ko"})
    assert r.status_code == 400 and r.json()["detail"].startswith("Speech editing does not serve language 'ko'")
    r = client.post("/v2/edit/align", json={"audio_base64": wav_b64(), "text": "hello", "language": "xx"})
    assert r.status_code == 400, r.text


class NoAsr(FakeAsr):
    """The server's ASR worker without .venv-asr."""

    installed = False

    def transcribe(self, items):
        raise AssertionError("no ASR installed")


def test_edit_without_asr_reports_no_transcription_and_refuses_transcribe(client, monkeypatch):
    state = client.app.state.zonos3
    aligner = FakeAligner()
    monkeypatch.setattr(state, "aligner", aligner)
    monkeypatch.setattr(state, "asr", NoAsr())
    monkeypatch.setattr(state, "edit_runtime", None)
    monkeypatch.setattr(state.options, "fusealign", True)
    monkeypatch.setattr(type(state.engine), "has_reference_encoder", property(lambda self: True))
    caps = client.get("/v2/capabilities").json()
    assert caps["speech_editing"] is True and caps["edit_transcribe"] is False
    r = client.post("/v2/edit/transcribe", json={"audio_base64": wav_b64(), "language": "en_us"})
    assert r.status_code == 503
    assert r.json()["detail"].startswith("Speech recognition is not available on this server: ")
    assert aligner.items == []
    # the transcript makes alignment work without recognition
    r = client.post("/v2/edit/align", json={"audio_base64": wav_b64(), "text": "hello there", "language": "en_us"})
    assert r.status_code == 200, r.text
    assert [w["word"] for w in r.json()["words"]] == ["hello", "there"]


def test_edit_refuses_bad_condition_buckets_before_recognition(client, monkeypatch):
    state = client.app.state.zonos3
    asr = FakeAsr()
    monkeypatch.setattr(state, "aligner", FakeAligner())
    monkeypatch.setattr(state, "asr", asr)
    monkeypatch.setattr(state, "edit_runtime", None)
    r = client.post("/v2/edit", json={"audio_base64": wav_b64(), "target_text": "hello world",
                                      "condition_buckets": {"nope": 1}})
    assert r.status_code == 400 and "unknown condition name" in r.json()["detail"]
    assert asr.items == []


def test_plugin_host_shares_the_server_asr_and_fusealign_workers(files, monkeypatch):
    import types

    from fastapi import APIRouter
    from zonos3.server import plugins

    seen, closed = {}, []

    class FakePlugin:
        name, host_api, keepalive_paths = "fake", plugins.HOST_API, ("/v2/fake/run",)

        def router(self, host):
            seen["host"] = host
            r = APIRouter()

            @r.get("/v2/fake/facts")
            def facts():
                return {"model_name": host.facts().model_name}

            @r.get("/fake")
            def page():
                return {"page": "fake"}

            return r

        def close(self):
            closed.append(self.name)

    monkeypatch.setitem(sys.modules, "zonos3_fake", types.SimpleNamespace(plugin=FakePlugin))
    aligner, asr = FakeAligner(), FakeAsr()
    engine = Engine.load(files / "m.gguf")
    scheduler = Scheduler(engine, batch=1)
    scheduler.start(warmup=False)
    state = ServerState(engine, scheduler, ServerOptions(model_path=str(files / "m.gguf"), batch=1, max_frames=60,
                                                         studio_dir=files, plugins=("fake",)),
                        asr=asr, aligner=aligner)
    state.plugins = plugins.load(state.options.plugins)
    with TestClient(create_app(state)) as c:
        assert c.get("/v2/capabilities").json()["plugins"] == ["fake"]
        assert c.get("/v2/fake/facts").json() == {"model_name": "m.gguf"}
        # the studio catch-all (studio_dir set) answers only after the plugin routes
        assert c.get("/fake").json() == {"page": "fake"}
    assert seen["host"].asr is asr and seen["host"].aligner is aligner
    assert closed == ["fake"]


def test_shutdown_closes_the_asr_and_fusealign_workers(files):
    aligner, asr = FakeAligner(), FakeAsr()
    with make_client(files, batch=1) as c:
        c.app.state.zonos3.aligner = aligner
        c.app.state.zonos3.asr = asr
    assert aligner.closed and asr.closed


def test_phonemize_marks_pronunciations_outside_the_phone_inventory(client):
    r = client.post("/v2/phonemize", json={"text": "hello abba zork", "language": "en_us", "guess": False})
    assert r.status_code == 200
    words = {s["text"]: (s["ipa"], s["status"]) for s in r.json()["segments"] if s["text"].strip()}
    assert words == {"hello": (None, "unsupported"), "abba": ("aɪb", "ok"), "zork": (None, "unknown")}


def test_session_voices_are_scoped_to_their_session(client):
    buf = io.BytesIO()
    np.save(buf, np.random.default_rng(0).standard_normal(192).astype(np.float32))
    created = client.post("/v2/speakers", headers={"X-TTS-Session-ID": "a"}, json={
        "speaker_embedding_base64": base64.b64encode(buf.getvalue()).decode(), "speaker_embedding_name": "v.npy"})
    assert created.status_code == 200
    voice = created.json()
    assert voice["source_type"] == "embedding" and voice["speaker_modes"] == ["embedding"]

    def ids(session):
        return [s["id"] for s in client.get("/v2/speakers", headers={"X-TTS-Session-ID": session}).json()["speakers"]]

    assert voice["id"] in ids("a") and voice["id"] not in ids("b")
    body = {"model": "zonos3", "input": "hi", "voice": voice["id"], "response_format": "wav", "max_tokens": 20,
            "speaker_mode": "embedding"}
    assert client.post("/v1/audio/speech", headers={"X-TTS-Session-ID": "a"}, json=body).status_code == 200
    other = client.post("/v1/audio/speech", headers={"X-TTS-Session-ID": "b"}, json=body)
    assert other.status_code == 400 and "unknown voice" in other.json()["detail"]
    # the default speaker mode is the raw reference prefix, which an embedding voice cannot supply
    reference = client.post("/v1/audio/speech", headers={"X-TTS-Session-ID": "a"},
                            json={**body, "speaker_mode": None})
    assert reference.status_code == 400 and "no reference audio" in reference.json()["detail"]


def test_cfg_needs_two_slots(files):
    buf = io.BytesIO()
    np.save(buf, np.ones(192, np.float32))
    body = {"text": "hi", "stream": False, "max_tokens": 10, "speaker_mode": "embedding", "speaker_cfg_scale": 1.5,
            "speaker_embedding_base64": base64.b64encode(buf.getvalue()).decode()}
    with make_client(files, batch=1) as one:
        assert one.get("/v2/capabilities").json()["speaker_cfg_supported"] is False
        r = one.post("/v2/generate", json=body)
        assert r.status_code == 400 and "--batch >= 2" in r.json()["detail"]


VOICE_ID = "default_" + hashlib.sha1(b"v.npy").hexdigest()[:16]  # the default voice v.npy
CALIBRATION = {"default": {"valence": 0.5}, "by_speaker": {VOICE_ID: {"valence-": 0.0}}, "global_default": 1.0}


@pytest.fixture(scope="module")
def steered(files, tmp_path_factory):
    """A server with a steer.gguf (tools/convert_gguf.py steer) of a raw set and a calibrated hidden set, and one
    default voice; yields (client, bundle JSON)."""
    import convert_gguf

    root = tmp_path_factory.mktemp("steer")
    rng = np.random.default_rng(5)
    for name, space, dim, direction, extra in (("who", "raw", 192, "gender", {}),
                                               ("mood", "attnres_source", 32, "valence", {"source_index": 1})):
        (root / name).mkdir()
        np.save(root / name / f"{direction}.npy", rng.standard_normal(dim).astype(np.float32))
        (root / name / "manifest.json").write_text(json.dumps(
            {"dim": dim, "space": space, "directions": {direction: {"file": f"{direction}.npy", "kind": "axis"}},
             **extra}))
    (root / "mood" / "calibration.json").write_text(json.dumps(CALIBRATION))
    bundle = root / "bundle.json"
    bundle.write_text(json.dumps({"name": "tiny", "sets": {"who": "who", "mood": "mood"}, "ui": {}}))
    convert_gguf.main(["steer", str(bundle), "-o", str(root / "steer.gguf")])
    (root / "voices").mkdir()
    np.save(root / "voices" / "v.npy", np.ones(192, np.float32))
    engine = Engine.load(files / "m.gguf", steer=str(root / "steer.gguf"))
    scheduler = Scheduler(engine, batch=2)
    scheduler.start(warmup=False)
    state = ServerState(engine, scheduler, ServerOptions(model_path=str(files / "m.gguf"), batch=2, max_frames=60,
                                                         default_voices_dir=root / "voices"))
    with TestClient(create_app(state)) as c:
        yield c, bundle


def test_steer_gguf_serves_the_set_calibrations_in_controls_and_speakers(steered):
    client, bundle = steered
    doc = client.get("/v2/controls").json()
    assert doc["sets"]["mood"]["calibration"] == CALIBRATION and "calibration" not in doc["sets"]["who"]
    want = controls_document(load_bundle(bundle), speaker_embedding_dim=doc["embedding_dim"])
    assert doc == json.loads(json.dumps(want))  # the GGUF bundle is the directory bundle
    voices = {s["id"]: s for s in client.get("/v2/speakers").json()["speakers"]}
    assert voices[VOICE_ID]["disabled_controls"] == ["mood.valence-"]
    buf = io.BytesIO()
    np.save(buf, np.ones(192, np.float32))
    created = client.post("/v2/speakers", headers={"X-TTS-Session-ID": "s"}, json={
        "speaker_embedding_base64": base64.b64encode(buf.getvalue()).decode(), "speaker_embedding_name": "u.npy"})
    assert created.status_code == 200 and created.json()["disabled_controls"] == []


def test_controls_cfg_twins_the_prompt_without_the_hidden_delta(steered):
    engine = steered[0].app.state.zonos3.engine
    speaker = Speaker("embedding", torch.ones(192))
    body = {"text": "hi", "speaker_mode": "embedding", "controls": {"mood.valence": 1.0}, "controls_cfg_scale": 2.0}
    plan = plan_request(engine, body, speaker)
    assert (plan.cfg_kind, plan.cfg_scale) == ("controls", 2.0) and plan.hidden_delta is not None
    assert plan.cond.steered and not plan.twin.steered
    assert torch.equal(plan.twin.input_ids, plan.cond.input_ids)
    assert plan_request(engine, {**body, "controls": {"who.gender": 1.0}}, speaker).twin is None  # no hidden delta
    with pytest.raises(ValueError, match="^Speaker and controls CFG cannot be combined$"):
        plan_request(engine, {**body, "speaker_cfg_scale": 1.5}, speaker)
