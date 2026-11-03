"""The MLX HTTP server on a tiny backbone (servers/ggml/scripts/make-dummy-model.py with edit rows, span-end and a
phoneme block) and the dummy decoder: what a client sees of streaming, speech editing, phonemize, session voices and
the CFG slot rule. The reference encoder is replaced by the fixture codes (the edit test needs codes, not audio)."""

from __future__ import annotations

import base64
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
from zonos3.tts.phonemes import normalize_headword
from zonos3_mlx.engine import Engine
from zonos3_mlx.scheduler import Scheduler
from zonos3_mlx.server import speakers
from zonos3_mlx.server.app import create_app
from zonos3_mlx.server.state import ServerOptions, ServerState

HOP = 2940
PHONES = "aɪb"
CODES = np.load(FIXTURES / "dummy-dac-codes.npy").astype(np.int32)  # [40, 10]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("srv")
    subprocess.run([sys.executable, str(REPO / "servers/ggml/scripts/make-dummy-model.py"), "-o", str(root / "m.gguf"),
                    "--edit-markers", "3", "--span-end", "--phonemes", PHONES], check=True, capture_output=True)
    (root / "dac.gguf").write_bytes((FIXTURES / "dummy-dac.gguf").read_bytes())
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
    original = (np.clip(original, -1, 1) * 32767).astype(np.int16)
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


def fake_mms(calls: list[dict]):
    """The MMS worker: a fixed transcript, and each normalized text upper-cased per word."""

    async def mms_one(state, request, language):
        calls.append({**request, "language": language})
        if request["op"] == "transcribe":
            return {"id": request["id"], "text": "hello there world", "tokens": ["HELLO", "THERE", "WORLD"]}
        assert request["op"] == "normalize", request
        return {"id": request["id"], "normalized": [t.upper().split() for t in request["normalize"]]}

    return mms_one


def test_edit_transcribe_and_align_time_words_with_fusealign(client, monkeypatch):
    from zonos3_mlx.server import edit

    aligner, calls = FakeAligner(), []
    monkeypatch.setattr(client.app.state.zonos3, "aligner", aligner)
    monkeypatch.setattr(edit, "mms_one", fake_mms(calls))
    r = client.post("/v2/edit/transcribe", json={"audio_base64": wav_b64(), "language": "fr_fr"})
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "hello there world" and r.json()["duration"] == pytest.approx(2.0)
    assert [(w["word"], w["tokens"], w["confidence"]) for w in r.json()["words"]] == [
        ("hello", ["HELLO"], 0.5), ("there", ["THERE"], 0.5), ("world", ["WORLD"], 0.5)]
    assert aligner.items[0]["tokens"] == ["hello", "there", "world"] and aligner.items[0]["language"] == "fr"
    assert [c["op"] for c in calls] == ["transcribe", "normalize"]

    aligner.items.clear()
    r = client.post("/v2/edit/align", json={"audio_base64": wav_b64(), "text": "我们 去公园。", "language": "cmn"})
    assert r.status_code == 200, r.text
    assert aligner.items[0]["tokens"] == ["我", "们", "去", "公", "园。"] and aligner.items[0]["language"] == "zh"
    assert [w["word"] for w in r.json()["words"]] == ["我", "们", "去", "公", "园。"]
    assert all(c["op"] != "align" for c in calls)  # MMS only transcribes and normalizes


def test_dub_shares_the_server_fusealign_worker(client, monkeypatch):
    state = client.app.state.zonos3
    aligner = FakeAligner()
    monkeypatch.setattr(state, "aligner", aligner)
    monkeypatch.setattr(state, "dub_runtime", None)
    caps = client.get("/v2/dub/capabilities").json()
    assert caps["aligner_model"] == "Zyphra/fusealign/fake.safetensors"
    assert state.dub_runtime.aligner is aligner


def test_shutdown_closes_the_fusealign_worker(files):
    aligner = FakeAligner()
    with make_client(files, batch=1) as c:
        c.app.state.zonos3.aligner = aligner
    assert aligner.closed


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
