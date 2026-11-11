import asyncio
import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
api_server = pytest.importorskip("zonos3_sgl.server.api_server")
from zonos3.reference.synth import slerp  # noqa: E402
from zonos3.server import voices  # noqa: E402
from zonos3.server.schemas import SpeakerCreateRequest  # noqa: E402
from zonos3.tts import plan  # noqa: E402


class _DummyModelConfig:
    speaker_enabled = True
    speaker_embedding_dim = 3


class _DummyConfig:
    model_config = _DummyModelConfig()
    tts_default_voices_dir = None


class _DefaultVoiceConfig(_DummyConfig):
    def __init__(self, default_voices_dir):
        self.tts_default_voices_dir = str(default_voices_dir)


def test_slerp_embeddings_respects_endpoints_and_midpoint():
    v0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    v1 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    start = slerp(v0, v1, 0.0)
    end = slerp(v0, v1, 1.0)
    mid = slerp(v0, v1, 0.5)

    assert torch.allclose(start, v0)
    assert torch.allclose(end, v1)
    expected_mid = torch.tensor([2**-0.5, 2**-0.5, 0.0], dtype=torch.float32)
    assert torch.allclose(mid, expected_mid, atol=1e-5)


def test_resolve_speaker_embedding_uses_cached_ids_and_blends():
    async def _run():
        api_server._SESSION_SPEAKER_CACHE.clear()
        speaker_a = await api_server._cache_speaker_reference(
            session_id="test-session",
            label="Speaker A",
            source_type="audio",
            embedding=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        )
        speaker_b = await api_server._cache_speaker_reference(
            session_id="test-session",
            label="Speaker B",
            source_type="audio",
            embedding=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
        )

        single = await api_server._resolve_speaker_embedding(
            config=_DummyConfig(),
            session_id="test-session",
            speaker_embedding_id=speaker_a.speaker_id,
        )
        blended = await api_server._resolve_speaker_embedding(
            config=_DummyConfig(),
            session_id="test-session",
            speaker_blend_embedding_id_a=speaker_a.speaker_id,
            speaker_blend_embedding_id_b=speaker_b.speaker_id,
            speaker_blend_t=0.5,
        )
        return single, blended

    single, blended = asyncio.run(_run())

    assert torch.allclose(single, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))
    expected_mid = torch.tensor([2**-0.5, 2**-0.5, 0.0], dtype=torch.float32)
    assert torch.allclose(blended, expected_mid, atol=1e-5)


def test_default_speaker_directory_lists_and_resolves_embedding(tmp_path):
    async def _run():
        api_server._DEFAULT_SPEAKER_CACHE.clear()
        np.save(tmp_path / "Alice Voice.npy", np.array([0.25, 0.5, 0.75], dtype=np.float32))

        config = _DefaultVoiceConfig(tmp_path)
        speakers = await api_server._list_default_speakers(config)
        resolved = await api_server._resolve_speaker_embedding(
            config=config,
            session_id=None,
            speaker_embedding_id=speakers[0].speaker_id,
        )
        serialized = api_server._speaker_document(config, speakers[0])
        return speakers, resolved, serialized

    speakers, resolved, serialized = asyncio.run(_run())

    assert len(speakers) == 1
    assert speakers[0].label == "Alice Voice"
    assert speakers[0].source_type == "embedding"
    assert torch.allclose(resolved, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32))
    assert serialized["is_default"] is True and serialized["scope"] == "default"
    assert serialized["has_preview"] is False
    assert serialized["speaker_modes"] == ["embedding"] and serialized["embedding_dim"] == 3
    assert serialized["disabled_controls"] == []


def test_default_speaker_transcript_sidecar(tmp_path):
    (tmp_path / "Example clip.wav").write_bytes(b"RIFF")
    (tmp_path / "Example clip.txt").write_text("Hello there.\n", encoding="utf-8")
    (tmp_path / "Other.mp3").write_bytes(b"ID3")

    async def _run():
        api_server._DEFAULT_SPEAKER_CACHE.clear()
        config = _DefaultVoiceConfig(tmp_path)
        speakers = await api_server._list_default_speakers(config)
        return [api_server._speaker_document(config, entry) for entry in speakers]

    serialized = asyncio.run(_run())

    assert {item["label"]: item["transcript"] for item in serialized} == {
        "Example clip": "Hello there.",
        "Other": None,
    }


class _RawReferenceModelConfig:
    speaker_enabled = False
    speaker_reference_prefix_enabled = True
    speaker_embedding_dim = 0
    residual_type = "full_attnres"
    hidden_size = 4
    n_codebooks = 10


class _RawReferenceConfig:
    model_config = _RawReferenceModelConfig()
    model_path = ""
    accent_adapters = None
    max_seq_len = 4096

    def __init__(self, default_voices_dir, steer_bundle):
        self.tts_default_voices_dir = str(default_voices_dir)
        self.steer_bundle = str(steer_bundle)


def _write_bank_bundle(root, hidden=4):
    """A steering bundle with one hidden-space set ``mood`` (bank source 4) and its calibration."""
    directory = root / "moods"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "sad.npy", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "dim": hidden,
                "space": "attnres_source",
                "source_index": 4,
                "directions": {"sad": {"file": "sad.npy", "kind": "named", "norm": 1.0}},
            }
        )
    )
    (directory / "calibration.json").write_text(
        json.dumps(
            {
                "global_default": 2.0,
                "default": {"sad": 3.0},
                "by_speaker": {"voice_a": {"sad": 0.0}},
            }
        )
    )
    (root / "bundle.json").write_text(json.dumps({"name": "t", "sets": {"mood": "moods"}, "ui": {"controls": []}}))
    return root / "bundle.json"


def test_raw_reference_model_steers_without_speaker_embedding(tmp_path, monkeypatch):
    """A hidden-space (attnres_source) bundle set steers raw-reference checkpoints: controls give a hidden delta
    with no embedding at their raw dose (the set's calibration is exposed, not applied), and a voice the
    calibration gives dose 0 lists the control in its disabled_controls."""
    from types import SimpleNamespace

    api_server._STEER_BUNDLE_CACHE.clear()
    monkeypatch.setattr(api_server, "_text_tokenizer_spec", lambda config: None)  # byte text
    config = _RawReferenceConfig(tmp_path / "voices", _write_bank_bundle(tmp_path))

    assert api_server._steering_sources(config) == (4,)
    ctx = api_server._RulesContext(config)
    request = plan.parse_request({"text": "Hello.", "controls": {"mood.sad": -1.5}})
    rules = plan.request_rules(ctx, request, plan.request_head(ctx, request), plan.Speaker("reference"))
    assert rules.embedding is None and rules.schedule is None and rules.sources == (4,)
    assert torch.allclose(rules.hidden_delta, torch.tensor([0.0, -1.5, 0.0, 0.0]))

    def speaker(speaker_id):
        entry = SimpleNamespace(
            speaker_id=speaker_id,
            label=speaker_id,
            source_type="audio",
            embedding=torch.empty(0),
            created_at=0.0,
            original_name=None,
            audio_bytes=b"clip",
        )
        return api_server._speaker_document(config, entry)

    assert speaker("voice_a")["disabled_controls"] == ["mood.sad"]
    assert speaker("voice_b")["disabled_controls"] == []
    api_server._STEER_BUNDLE_CACHE.clear()


def test_raw_reference_checkpoints_resolve_voice_ids_to_reference_audio(tmp_path, monkeypatch):
    # The web UI selects voices by id; raw-reference checkpoints need the audio behind it.
    monkeypatch.setattr(voices, "_transcode_audio_bytes_to_wav", lambda data: b"WAV:" + data)
    (tmp_path / "Bob.mp3").write_bytes(b"mp3")
    config = _DefaultVoiceConfig(tmp_path)

    def resolve(**overrides):
        request = {
            "config": config,
            "session_id": "session",
            "speaker_audio_base64": None,
            "speaker_embedding_base64": None,
            "speaker_embedding_id": None,
            "speaker_blend_embedding_id_a": None,
            "speaker_blend_embedding_id_b": None,
        }
        request.update(overrides)
        return asyncio.run(api_server._resolve_speaker_reference_audio(**request))

    async def _setup():
        api_server._DEFAULT_SPEAKER_CACHE.clear()
        api_server._SESSION_SPEAKER_CACHE.clear()
        default = (await api_server._list_default_speakers(config))[0]
        clip = await api_server._cache_speaker_reference(
            session_id="session", label="Clip", source_type="audio", embedding=torch.empty(0), audio_bytes=b"clip"
        )
        embedding_only = await api_server._cache_speaker_reference(
            session_id="session",
            label="Embedding",
            source_type="embedding",
            embedding=torch.zeros(3),
        )
        return default.speaker_id, clip.speaker_id, embedding_only.speaker_id

    default_id, clip_id, embedding_id = asyncio.run(_setup())

    assert resolve() is None
    assert resolve(speaker_embedding_id=default_id) == b"WAV:mp3"
    assert resolve(speaker_embedding_id=clip_id) == b"clip"
    with pytest.raises(ValueError):
        resolve(speaker_embedding_id=embedding_id)
    with pytest.raises(ValueError):
        resolve(speaker_embedding_id=clip_id, speaker_audio_base64="AAAA")
    with pytest.raises(ValueError):
        resolve(speaker_blend_embedding_id_a=clip_id, speaker_blend_embedding_id_b=default_id)


class _DualPathModelConfig:
    speaker_enabled = True
    speaker_reference_prefix_enabled = True
    speaker_embedding_dim = 3


def test_dual_path_checkpoints_cache_clips_and_embedding_files(monkeypatch):
    """A checkpoint with both speaker paths (the release LM) caches an uploaded clip with its
    embedding, and an embedding file (a studio-designed voice) on its own, usable in
    speaker_mode=embedding only."""
    import base64
    import io
    from types import SimpleNamespace

    config = SimpleNamespace(model_config=_DualPathModelConfig(), tts_default_voices_dir=None)
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", SimpleNamespace(config=config))
    monkeypatch.setattr(api_server, "_transcode_audio_bytes_to_wav", lambda data: data)
    monkeypatch.setattr(api_server, "_decode_wav_bytes", lambda data: (torch.zeros(16000), 16000))

    async def embed(wav, rate, **_):
        return torch.tensor([0.0, 1.0, 0.0])

    monkeypatch.setattr(api_server, "_compute_speaker_embedding_from_waveform", embed)
    api_server._SESSION_SPEAKER_CACHE.clear()
    buf = io.BytesIO()
    np.save(buf, np.array([1.0, 2.0, 2.0], dtype=np.float32))
    npy = "data:application/octet-stream;base64," + base64.b64encode(buf.getvalue()).decode()

    clip = asyncio.run(
        api_server.tts_cache_speaker(
            SpeakerCreateRequest(speaker_audio_base64="data:audio/wav;base64,AAAA", label="Clip"), "s"
        )
    )
    designed = asyncio.run(
        api_server.tts_cache_speaker(
            SpeakerCreateRequest(speaker_embedding_base64=npy, speaker_embedding_name="designed.npy"), "s"
        )
    )

    resolve = lambda sid: asyncio.run(
        api_server._resolve_cached_speaker_embedding(config=config, session_id="s", speaker_id=sid)
    )
    assert torch.equal(resolve(clip["id"]), torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(resolve(designed["id"]), torch.tensor([1.0, 2.0, 2.0]))
    # v2 Speaker shape: a clip supports both modes, an embedding file speaker_mode=embedding only
    assert clip["scope"] == "session" and clip["is_default"] is False and clip["transcript"] is None
    assert clip["source_type"] == "audio" and clip["speaker_modes"] == ["reference", "embedding"]
    assert designed["source_type"] == "embedding" and designed["speaker_modes"] == ["embedding"]
    assert clip["embedding_dim"] == designed["embedding_dim"] == 3


def test_accents_are_adapters_only(monkeypatch):
    """`accent` picks an accent adapter; anything else is refused with the loaded adapter names."""
    from types import SimpleNamespace

    ctx = api_server._RulesContext(SimpleNamespace(accent_adapters=None))
    assert plan.resolve_accent(ctx.accent_names, None, 2.0) is None
    with pytest.raises(ValueError):
        plan.resolve_accent(ctx.accent_names, "cmn", 1.0)
    monkeypatch.setattr(api_server, "_adapter_accents", lambda config: ["cmn", "spa"])
    assert plan.resolve_accent(ctx.accent_names, "cmn", 0.5) == ("cmn", 0.5)
    with pytest.raises(ValueError):
        plan.resolve_accent(ctx.accent_names, "kor", 1.0)


def test_reference_clips_encode_once_and_failed_encodes_are_retried(monkeypatch):
    """The frontend encodes raw reference clips (one codec encoder per server, not one per tokenizer process):
    concurrent requests with one clip share a single encode, and a clip the encoder refused is encoded again."""
    import threading

    calls = []
    release = threading.Event()  # holds the first encode of each clip until every request is waiting on it

    def encode(audio_codec, audio, *, max_seconds):
        calls.append((audio, max_seconds))
        release.wait(5)
        if audio == b"short":
            raise ValueError("Reference shorter than 0.5 s")
        return torch.full((3, 10), len(calls), dtype=torch.int32)

    monkeypatch.setattr(api_server, "encode_reference_audio_bytes", encode)
    monkeypatch.setattr(api_server, "_audio_codec", lambda config: "codec")
    monkeypatch.setattr(api_server, "_REFERENCE_CODES", type(api_server._REFERENCE_CODES)())

    async def requests(audio, n):
        pending = [asyncio.ensure_future(api_server._encode_reference_audio(None, audio, 60.0)) for _ in range(n)]
        await asyncio.sleep(0.05)
        release.set()
        results = await asyncio.gather(*pending, return_exceptions=True)
        release.clear()
        return results

    async def _run():
        return await requests(b"clip", 8), await requests(b"short", 2), await requests(b"short", 1)

    same, refused, retried = asyncio.run(_run())

    assert calls == [(b"clip", 60.0), (b"short", 60.0), (b"short", 60.0)]
    assert all(torch.equal(codes, torch.full((3, 10), 1, dtype=torch.int32)) for codes in same)
    same[0][0, 0] = -1  # every caller gets its own copy of the memoized codes
    assert same[1][0, 0] == 1
    assert all(isinstance(exc, ValueError) for exc in refused + retried)
