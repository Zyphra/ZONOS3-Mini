import asyncio
import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
api_server = pytest.importorskip("zonos3_sgl.server.api_server")
from zonos3.server.schemas import SpeakerCreateRequest  # noqa: E402


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

    start = api_server._slerp_embeddings(v0, v1, 0.0)
    end = api_server._slerp_embeddings(v0, v1, 1.0)
    mid = api_server._slerp_embeddings(v0, v1, 0.5)

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
        serialized = api_server._serialize_default_speaker(config, speakers[0], expected_dim=3)
        return speakers, resolved, serialized

    speakers, resolved, serialized = asyncio.run(_run())

    assert len(speakers) == 1
    assert speakers[0].label == "Alice Voice"
    assert speakers[0].source_type == "embedding"
    assert torch.allclose(resolved, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32))
    assert serialized["is_default"] is True and serialized["scope"] == "default"
    assert serialized["has_preview"] is False
    assert serialized["speaker_modes"] == ["embedding"] and serialized["embedding_dim"] == 3
    assert serialized["emotion_disabled"] is False


def test_default_speaker_transcript_sidecar(tmp_path):
    (tmp_path / "Example clip.wav").write_bytes(b"RIFF")
    (tmp_path / "Example clip.txt").write_text("Hello there.\n", encoding="utf-8")
    (tmp_path / "Other.mp3").write_bytes(b"ID3")

    async def _run():
        api_server._DEFAULT_SPEAKER_CACHE.clear()
        config = _DefaultVoiceConfig(tmp_path)
        speakers = await api_server._list_default_speakers(config)
        return [api_server._serialize_default_speaker(config, entry, 3) for entry in speakers]

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


class _RawReferenceConfig:
    model_config = _RawReferenceModelConfig()

    def __init__(self, default_voices_dir, directions_dir):
        self.tts_default_voices_dir = str(default_voices_dir)
        self.tts_emotion_directions_dir = str(directions_dir)


def _write_bank_directions(directory, hidden=4):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "sad.npy", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    (directory / "manifest.json").write_text(json.dumps({
        "dim": hidden, "space": "attnres_source", "source_index": 4,
        "directions": {"sad": {"file": "sad.npy", "kind": "named", "norm": 1.0}},
    }))
    (directory / "calibration.json").write_text(json.dumps({
        "global_default": 2.0, "default": {"sad": 3.0}, "by_speaker": {"voice_a": {"sad": 0.0}},
    }))


def test_raw_reference_model_steers_without_speaker_embedding(tmp_path):
    """attnres_source directions apply to raw-reference checkpoints: the hidden
    delta is produced with no embedding, calibrated per voice, and a dose-0 voice
    reports the control as disabled."""
    api_server._EMOTION_DIRECTIONS_CACHE.clear()
    api_server._EMOTION_CALIBRATION_CACHE.clear()
    directions_dir = tmp_path / "dirs"
    _write_bank_directions(directions_dir)
    config = _RawReferenceConfig(tmp_path / "voices", directions_dir)

    caps = api_server._emotion_capabilities(config)
    assert caps["emotion_enabled"] is True
    assert caps["emotion_space"] == "attnres_source"
    assert caps["emotion_names"] == ["sad"]

    emb, delta = api_server._apply_request_emotion(
        config, None, speaker_id="voice_b", emotion_enabled=True,
        sliders={"sad": -1.0}, valence=0.0, arousal=0.0, strength=1.0,
    )
    assert emb is None
    assert torch.allclose(delta, torch.tensor([0.0, -3.0, 0.0, 0.0]))  # calibrated default dose 3

    _, raw = api_server._apply_request_emotion(
        config, None, speaker_id="voice_b", emotion_enabled=True,
        sliders={"sad": 1.0}, valence=0.0, arousal=0.0, strength=5.0, use_calibration=False,
    )
    assert torch.allclose(raw, torch.tensor([0.0, 5.0, 0.0, 0.0]))

    _, disabled = api_server._apply_request_emotion(
        config, None, speaker_id="voice_a", emotion_enabled=True,
        sliders={"sad": 1.0}, valence=0.0, arousal=0.0, strength=1.0,
    )
    assert disabled is None  # dose 0 folds the slider to nothing
    assert api_server._speaker_emotion_disabled(config, "voice_a") == ["sad"]


def test_embedding_space_directions_stay_off_for_raw_reference_models(tmp_path):
    api_server._EMOTION_DIRECTIONS_CACHE.clear()
    api_server._EMOTION_CALIBRATION_CACHE.clear()
    directions_dir = tmp_path / "proj"
    directions_dir.mkdir()
    np.save(directions_dir / "happy.npy", np.array([1.0, 0.0], dtype=np.float32))
    (directions_dir / "manifest.json").write_text(json.dumps({
        "dim": 2, "space": "proj",
        "directions": {"happy": {"file": "happy.npy", "kind": "named", "norm": 1.0}},
    }))
    config = _RawReferenceConfig(tmp_path / "voices", directions_dir)
    assert api_server._emotion_capabilities(config)["emotion_enabled"] is False
    with pytest.raises(ValueError):
        api_server._apply_request_emotion(
            config, None, emotion_enabled=True, sliders={"happy": 1.0},
            valence=0.0, arousal=0.0, strength=1.0,
        )


def test_raw_reference_checkpoints_resolve_voice_ids_to_reference_audio(tmp_path, monkeypatch):
    # The web UI selects voices by id; raw-reference checkpoints need the audio behind it.
    monkeypatch.setattr(api_server, "_transcode_audio_bytes_to_wav", lambda data: b"WAV:" + data)
    (tmp_path / "Bob.mp3").write_bytes(b"mp3")
    config = _DefaultVoiceConfig(tmp_path)

    def resolve(**overrides):
        request = dict(
            config=config,
            session_id="session",
            speaker_audio_base64=None,
            speaker_embedding_base64=None,
            speaker_embedding_id=None,
            speaker_blend_embedding_id_a=None,
            speaker_blend_embedding_id_b=None,
        )
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
    with pytest.raises(ValueError, match="has no reference audio"):
        resolve(speaker_embedding_id=embedding_id)
    with pytest.raises(ValueError, match="either a reference audio file or a cached speaker id"):
        resolve(speaker_embedding_id=clip_id, speaker_audio_base64="AAAA")
    with pytest.raises(ValueError, match="embedding blends"):
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

    clip = asyncio.run(api_server.tts_cache_speaker(
        SpeakerCreateRequest(speaker_audio_base64="data:audio/wav;base64,AAAA", label="Clip"), "s"))
    designed = asyncio.run(api_server.tts_cache_speaker(
        SpeakerCreateRequest(speaker_embedding_base64=npy, speaker_embedding_name="designed.npy"), "s"))

    resolve = lambda sid: asyncio.run(api_server._resolve_cached_speaker_embedding(config=config, session_id="s", speaker_id=sid))
    assert torch.equal(resolve(clip["id"]), torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(resolve(designed["id"]), torch.tensor([1.0, 2.0, 2.0]))
    # v2 Speaker shape: a clip supports both modes, an embedding file speaker_mode=embedding only
    assert clip["scope"] == "session" and clip["is_default"] is False and clip["transcript"] is None
    assert clip["source_type"] == "audio" and clip["speaker_modes"] == ["reference", "embedding"]
    assert designed["source_type"] == "embedding" and designed["speaker_modes"] == ["embedding"]
    assert clip["embedding_dim"] == designed["embedding_dim"] == 3



class _AccentSteeringConfig:
    model_config = _RawReferenceModelConfig()
    accent_adapters = None

    def __init__(self, directions_dir):
        self.accent_directions = str(directions_dir)


def test_steering_accent_folds_into_the_hidden_delta(tmp_path):
    """--accent-directions accents resolve like adapters but become a hidden delta:
    strength x the voice's calibrated dose x direction, added to the emotion delta;
    a dose-0 voice gets none, the raw dose applies without calibration, and adapter
    accents pass through."""
    api_server._ACCENT_DIRECTIONS_CACHE.clear()
    api_server._ACCENT_CALIBRATION_CACHE.clear()
    directions_dir = tmp_path / "accents"
    directions_dir.mkdir()
    np.save(directions_dir / "cmn.npy", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    (directions_dir / "manifest.json").write_text(json.dumps({
        "dim": 4, "space": "attnres_source", "source_index": 19,
        "directions": {"cmn": {"file": "cmn.npy", "kind": "named", "norm": 1.0}},
    }))
    (directions_dir / "calibration.json").write_text(json.dumps({
        "global_default": 1.0, "default": {"cmn": 8.0}, "by_speaker": {"voice_a": {"cmn": 0.0}},
    }))
    config = _AccentSteeringConfig(directions_dir)

    assert api_server._server_accents(config) == ["cmn"]
    assert api_server._resolve_tts_request_accent(config, "cmn", 0.5) == ("cmn", 0.5)
    with pytest.raises(ValueError, match="Unknown accent"):
        api_server._resolve_tts_request_accent(config, "kor", 1.0)

    emotion = torch.tensor([1.0, 0.0, 0.0, 0.0])
    adapter, _, delta = api_server._apply_steering_accent(config, "cmn", 0.5, "voice_b", None, emotion)
    assert adapter is None
    assert torch.allclose(delta, torch.tensor([1.0, 0.0, 4.0, 0.0]))       # 0.5 x calibrated dose 8
    _, _, alone = api_server._apply_steering_accent(config, "cmn", 0.5, "voice_b", None, None)
    assert torch.allclose(alone, torch.tensor([0.0, 0.0, 4.0, 0.0]))
    _, _, raw = api_server._apply_steering_accent(config, "cmn", 3.0, "voice_b", None, None, use_calibration=False)
    assert torch.allclose(raw, torch.tensor([0.0, 0.0, 3.0, 0.0]))
    assert api_server._apply_steering_accent(config, "cmn", 1.0, "voice_a", None, None) == (None, None, None)
    assert api_server._apply_steering_accent(config, "spa", 1.0, "voice_b", None, emotion)[0] == "spa"


class _SpeakerEmbeddingModelConfig:
    speaker_enabled = True


def test_raw_accent_moves_the_speaker_embedding(tmp_path):
    """A raw (speaker-embedding) accent set shifts the request's embedding by strength x
    direction, rescaled to the embedding's norm, and leaves the hidden delta alone; a
    request without an embedding (speaker_mode=reference) is refused, not silently
    unaccented."""
    api_server._ACCENT_DIRECTIONS_CACHE.clear()
    api_server._ACCENT_CALIBRATION_CACHE.clear()
    root = tmp_path / "accents"
    root.mkdir()
    np.save(root / "hin.npy", np.array([0.0, 2.0, 0.0], dtype=np.float32))
    (root / "manifest.json").write_text(json.dumps({
        "dim": 3, "space": "raw", "directions": {"hin": {"file": "hin.npy", "kind": "named", "norm": 2.0}},
    }))
    config = _AccentSteeringConfig(root)
    config.model_config = _SpeakerEmbeddingModelConfig()
    emotion = torch.tensor([0.5, 0.0])

    adapter, emb, delta = api_server._apply_steering_accent(
        config, "hin", 1.5, "voice", torch.tensor([4.0, 0.0, 0.0]), emotion)
    assert adapter is None
    assert torch.allclose(emb, torch.tensor([4.0, 3.0, 0.0]) * (4.0 / 5.0))  # [4, 0, 0] + 1.5 x [0, 2, 0], norm 4
    assert torch.equal(delta, emotion)
    with pytest.raises(ValueError, match="speaker_mode=embedding"):
        api_server._apply_steering_accent(config, "hin", 1.0, "voice", None, None)


def test_axis_directions_are_accents_with_signed_strength(tmp_path):
    """An accent set's axes (e.g. a tempo or conversational style) are request accents
    like its named directions, listed and steerable both ways."""
    api_server._ACCENT_DIRECTIONS_CACHE.clear()
    api_server._ACCENT_CALIBRATION_CACHE.clear()
    root = tmp_path / "styles"
    root.mkdir()
    np.save(root / "tempo.npy", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    (root / "manifest.json").write_text(json.dumps({
        "dim": 4, "space": "attnres_source", "source_index": 19,
        "directions": {"tempo": {"file": "tempo.npy", "kind": "axis"}},
    }))
    config = _AccentSteeringConfig(root)

    assert api_server._server_accents(config) == ["tempo"]
    _, _, delta = api_server._apply_steering_accent(config, "tempo", -2.0, "voice", None, None)
    assert torch.equal(delta, torch.tensor([0.0, -2.0, 0.0, 0.0]))



def test_emotion_and_accent_on_different_sources_get_their_own_slots(tmp_path):
    """Emotion directions on bank source 19 and accent directions on source 15: the
    request delta is one slot per steered source in ascending source order (15, 19),
    flattened, so the engine adds each set at its own source -- also when only one
    of the two is requested."""
    api_server._ACCENT_DIRECTIONS_CACHE.clear()
    api_server._ACCENT_CALIBRATION_CACHE.clear()
    api_server._EMOTION_DIRECTIONS_CACHE.clear()
    for name, source, vector in (("accents", 15, [0.0, 0.0, 1.0, 0.0]), ("emotions", 19, [1.0, 0.0, 0.0, 0.0])):
        root = tmp_path / name
        root.mkdir()
        np.save(root / "d.npy", np.array(vector, dtype=np.float32))
        label = "cmn" if name == "accents" else "happy"
        (root / "manifest.json").write_text(json.dumps({
            "dim": 4, "space": "attnres_source", "source_index": source,
            "directions": {label: {"file": "d.npy", "kind": "named", "norm": 1.0}},
        }))
    config = _AccentSteeringConfig(tmp_path / "accents")
    config.tts_emotion_directions_dir = str(tmp_path / "emotions")

    assert api_server._steering_sources(config) == (15, 19)
    emotion = torch.tensor([0.0, 2.0, 0.0, 0.0])
    _, _, both = api_server._apply_steering_accent(config, "cmn", 3.0, "voice", None, emotion)
    assert torch.equal(both, torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0]))
    _, _, emotion_only = api_server._apply_steering_accent(config, None, 1.0, "voice", None, emotion)
    assert torch.equal(emotion_only, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]))
    _, _, accent_only = api_server._apply_steering_accent(config, "cmn", 3.0, "voice", None, None)
    assert torch.equal(accent_only, torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert api_server._apply_steering_accent(config, None, 1.0, "voice", None, None) == (None, None, None)
    api_server._EMOTION_DIRECTIONS_CACHE.clear()
