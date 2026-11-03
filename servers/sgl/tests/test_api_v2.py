"""The v2 HTTP surface of the sgl server against api/openapi.json, with a stub frontend (no model): request models
transcribe the spec, routes live under /v1 and /v2 only, and the generate / speech / speaker responses take the
v2 shapes (format, X-Seed, voice ids)."""
from __future__ import annotations

import io
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

api_server = pytest.importorskip("zonos3_sgl.server.api_server")
TestClient = pytest.importorskip("fastapi.testclient").TestClient
from zonos3.server import schemas  # noqa: E402

SPEC = json.loads((Path(__file__).resolve().parents[3] / "api" / "openapi.json").read_text())
CODEC = "opus_dual_3n_600k_sched600k_r4096_step600000"


@pytest.mark.parametrize("name, model", [
    ("GenerateRequest", schemas.GenerateRequest),
    ("SpeechRequest", schemas.SpeechRequest),
    ("ControlSpan", schemas.ControlSpan),
    ("SpeakerCreateRequest", schemas.SpeakerCreateRequest),
    ("PhonemizeRequest", schemas.PhonemizeRequest),
    ("EditTranscribeRequest", schemas.EditTranscribeRequest),
    ("EditAlignRequest", schemas.EditAlignRequest),
    ("EditWordIn", schemas.EditWordIn),
    ("EditRequest", schemas.EditRequest),
    ("DubTranscribeRequest", schemas.DubTranscribeRequest),
    ("DubWord", schemas.DubWord),
    ("DubAnalyzeRequest", schemas.DubAnalyzeRequest),
    ("DubVoice", schemas.DubVoice),
    ("DubSegmentIn", schemas.DubSegmentIn),
    ("DubGenerateRequest", schemas.DubGenerateRequest),
    ("DubRegenerateRequest", schemas.DubRegenerateRequest),
    ("ProjectPutRequest", schemas.ProjectPutRequest),
])
def test_request_models_transcribe_the_spec(name, model):
    schema = SPEC["components"]["schemas"][name]
    assert schema["additionalProperties"] is False
    fields = model.model_fields
    assert set(fields) == set(schema["properties"])
    assert {n for n, f in fields.items() if f.is_required()} == set(schema.get("required", ()))
    for prop, spec in schema["properties"].items():
        if "default" in spec:
            assert fields[prop].get_default(call_default_factory=True) == spec["default"], prop


def _config(tmp_path, **extra):
    fields = dict(
        model_path=str(tmp_path / "lm"),
        model_config=SimpleNamespace(audio_codec=CODEC, speaker_enabled=True, speaker_embedding_dim=3,
                                     speaker_reference_prefix_enabled=False, condition_tokens_enabled=False,
                                     span_end_id=None, text_extra_tokens=0, hidden_size=4),
        max_seq_len=4096, tts_default_voices_dir=None, studio_projects_dir=None, studio_dir=str(tmp_path / "studio"),
        accent_adapters=None, accent_directions=None, steer_bundle=None, laci_heads={},
    )
    return SimpleNamespace(**{**fields, **extra})


class _Frontend:
    def __init__(self, config):
        self.config = config
        self.sent = []

    def new_user(self):
        return len(self.sent) + 1

    async def send_one(self, message):
        self.sent.append(message)

    async def stream_tts_audio(self, uid, **kwargs):
        yield np.full(8, 0.25, dtype=np.float32).tobytes()

    async def wait_for_tts_audio(self, uid, **kwargs):
        return np.array([0.5, -2.0, 2.0, 0.0], dtype=np.float32).tobytes()

    async def abort_user(self, uid):
        pass


@pytest.fixture
def frontend(tmp_path, monkeypatch):
    voices = tmp_path / "voices"
    voices.mkdir()
    np.save(voices / "Alice.npy", np.array([0.25, 0.5, 0.75], dtype=np.float32))
    state = _Frontend(_config(tmp_path, tts_default_voices_dir=str(voices)))
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    monkeypatch.setattr(api_server, "_text_tokenizer_spec", lambda model_path: None)
    monkeypatch.delenv("ZONOS3_ACCESS_TOKEN", raising=False)
    api_server._DEFAULT_SPEAKER_CACHE.clear()
    api_server._SESSION_SPEAKER_CACHE.clear()
    return state


@pytest.fixture
def client(frontend):
    return TestClient(api_server.app)


def test_meta_routes(client, frontend):
    assert client.get("/v2/health").json() == {"status": "ok"}
    assert client.get("/v1/models").json() == {"object": "list", "data": [
        {"id": "zonos3", "object": "model", "created": 0, "owned_by": "zyphra", "root": frontend.config.model_path}]}
    exposed = client.get("/v2/health", headers={"Origin": "https://x.example"}).headers["access-control-expose-headers"]
    assert "X-Seed" in exposed


@pytest.mark.parametrize("method, path", [
    ("GET", "/v1"), ("POST", "/v1"), ("POST", "/tts/generate"), ("GET", "/tts/capabilities"), ("GET", "/dub/capabilities"),
    ("GET", "/projects"), ("GET", "/studio/"), ("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json"),
    ("GET", "/v2/nope"),
])
def test_removed_paths_are_404(client, method, path):
    r = client.request(method, path)
    assert r.status_code == 404 and isinstance(r.json()["detail"], str)


def test_studio_pages_are_no_store_and_assets_immutable(client, tmp_path):
    studio = tmp_path / "studio"
    (studio / "assets").mkdir(parents=True)
    (studio / "index.html").write_text('<div id="app"></div>')
    (studio / "lite.html").write_text('<div id="app"></div>')
    (studio / "assets" / "index-abc.js").write_text("export {}")
    for path in ("/", "/lite.html"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html"), path
        assert r.headers["cache-control"] == "no-store, max-age=0", path
    r = client.get("/assets/index-abc.js")
    assert r.status_code == 200 and r.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert client.get("/nope.html").status_code == 404


def test_generate_streams_pcm_with_the_drawn_seed(client, frontend):
    r = client.post("/v2/generate", json={"text": "hi"})
    assert r.status_code == 200 and r.headers["x-audio-format"] == "float32"
    (msg,) = frontend.sent
    assert r.headers["x-seed"] == str(msg.sampling_params.seed) and 0 <= msg.sampling_params.seed < 2**31
    assert np.frombuffer(r.content, dtype="<f4").tolist() == [0.25] * 8


def test_generate_wav_is_buffered_pcm16_and_clipped(client, frontend):
    r = client.post("/v2/generate", json={"text": "hi", "stream": False, "format": "wav", "seed": 4294967295})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/wav")
    assert r.headers["x-audio-format"] == "pcm16" and r.headers["x-seed"] == "4294967295"
    assert frontend.sent[0].sampling_params.seed == 4294967295
    with wave.open(io.BytesIO(r.content)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (44100, 1, 2)
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert samples.tolist() == [16383, -32767, 32767, 0]


def test_stream_with_wav_is_400(client, frontend):
    r = client.post("/v2/generate", json={"text": "hi", "format": "wav"})
    assert r.status_code == 400 and r.json() == {"detail": "format=wav needs stream=false"}
    assert not frontend.sent


@pytest.mark.parametrize("fields", [
    {"no_such_field": 1}, {"speed": 1.0}, {"prepend_silence": True}, {"quality_buckets": {"a": 1}},
    {"speaking_rate_bucket": 2}, {"clean_speaker_background": True}, {"accurate_mode": True},
    {"speaker_wav_base64": "AAAA"}, {"seed": -1}, {"seed": 2**32}, {"headroom_db": 25}, {"tag_cfg_scale": 5},
    {"speaker_mode": "x"}, {"stream_decode": "x"}, {"format": "mp3"}, {"speaker_blend_t": 2},
    {"speaker_reference_codes": []}, {"control_spans": [{"start": -1, "end": 2, "controls": {}}]},
])
def test_schema_violations_are_422(client, frontend, fields):
    r = client.post("/v2/generate", json={"text": "hi", **fields})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][0] == "body"
    assert not frontend.sent


def _default_voice(client):
    (voice,) = client.get("/v2/speakers").json()["speakers"]
    return voice


def test_speakers_take_the_v2_shape(client):
    voice = _default_voice(client)
    assert voice == {
        "id": voice["id"], "label": "Alice", "source_type": "embedding", "scope": "default", "is_default": True,
        "original_name": "Alice.npy", "created_at": voice["created_at"], "has_preview": False,
        "speaker_modes": ["embedding"], "embedding_dim": 3, "transcript": None, "emotion_disabled": False,
    }
    assert client.post("/v2/speakers", json={"label": "x"}).status_code == 400            # no session header
    assert client.post("/v2/speakers", json={"speaker_wav_base64": "AAAA"}).status_code == 422


def test_speech_voice_is_a_speaker_id(client, frontend):
    voice = _default_voice(client)
    r = client.post("/v1/audio/speech", json={"model": "anything", "input": "hi", "voice": voice["id"]})
    assert r.status_code == 200 and r.headers["x-audio-format"] == "float32"
    (msg,) = frontend.sent
    assert msg.text == "hi" and msg.speaker_embedding.tolist() == [0.25, 0.5, 0.75]
    r = client.post("/v1/audio/speech", json={"model": "m", "input": "hi", "response_format": "wav", "seed": 3})
    assert r.status_code == 200 and r.content[:4] == b"RIFF" and r.headers["x-seed"] == "3"


def test_speech_voice_from_the_session_cache(client, frontend):
    entry = api_server.CachedSpeakerReference(speaker_id="spk_1", label="s", source_type="embedding",
                                              embedding=api_server.torch.tensor([1.0, 0.0, 0.0]), created_at=0.0)
    api_server._SESSION_SPEAKER_CACHE["sess"] = {"spk_1": entry}
    body = {"model": "m", "input": "hi", "voice": "spk_1"}
    assert client.post("/v1/audio/speech", json=body).status_code == 400              # another session
    assert client.post("/v1/audio/speech", json=body, headers={"X-TTS-Session-ID": "sess"}).status_code == 200


@pytest.mark.parametrize("fields, status, detail", [
    ({"voice": "alloy"}, 400, "unknown voice 'alloy'"),
    ({"voice": "alloy", "speaker_audio_base64": "AAAA"}, 400, "voice cannot be combined with speaker_* fields"),
    ({"speed": 1.0}, 422, None),
])
def test_speech_refusals(client, frontend, fields, status, detail):
    r = client.post("/v1/audio/speech", json={"model": "m", "input": "hi", **fields})
    assert r.status_code == status
    if detail is not None:
        assert r.json() == {"detail": detail}
    assert not frontend.sent


def test_capabilities_drop_the_legacy_keys(frontend, monkeypatch):
    import asyncio

    monkeypatch.setattr(api_server, "_decoder_capability", lambda config: None)
    caps = asyncio.run(api_server.tts_capabilities())
    assert set(caps) == set(SPEC["components"]["schemas"]["Capabilities"]["required"])
    assert "prepend_silence" not in json.dumps(caps["silence_prefill"])
