"""Request validation and plumbing for tagged-vs-plain TTS guidance."""

import asyncio
from types import SimpleNamespace

import pytest

api_server = pytest.importorskip("zonos3_sgl.server.api_server")
HTTPException = api_server.HTTPException
from pydantic import ValidationError  # noqa: E402
from zonos3.server.capabilities import tag_cfg_capabilities  # noqa: E402
from zonos3.server.errors import call as edit_call  # noqa: E402
from zonos3.server.schemas import EditRequest, GenerateRequest, SpeechRequest  # noqa: E402
from zonos3.tts.markup import NONVERBAL_MARKUP_TAGS  # noqa: E402


class _Frontend:
    def __init__(self):
        self.config = SimpleNamespace(
            model_path="",
            model_config=SimpleNamespace(
                audio_codec="opus_dual_3n_600k_sched600k_r4096_step600000",
                speaker_enabled=False,
                speaker_embedding_enabled=False,
                speaker_reference_prefix_enabled=False,
                speaking_rate_num_buckets=0,
                quality_features=(),
                quality_num_buckets=0,
                condition_tokens_enabled=False,
                condition_count=0,
                span_end_id=1026,
                text_extra_tokens=4,
                max_seq_len=4096,
                n_codebooks=10,
                audio_pad_id=4097,
                codebook_sizes=(4096,) * 10,
            ),
            max_seq_len=4096,
            tts_n_codebooks=10,
            tts_audio_pad_id=4097,
            tts_audio_codec="opus_dual_3n_600k_sched600k_r4096_step600000",
            accent_adapters=None,
            laci_heads={},
        )
        self.sent = []

    def new_user(self):
        return len(self.sent) + 1

    async def send_one(self, message):
        self.sent.append(message)

    async def stream_tts_audio(self, uid, **kwargs):
        yield b""

    async def abort_user(self, uid):
        pass


@pytest.fixture
def frontend(monkeypatch):
    state = _Frontend()
    monkeypatch.setattr(api_server, "get_global_state", lambda: state)
    monkeypatch.setattr(api_server, "_text_tokenizer_spec", lambda config: None)  # byte text, no phoneme groups
    return state


def _request(endpoint, text, **kwargs):
    if endpoint == "generate":
        return GenerateRequest(text=text, **kwargs)
    return SpeechRequest(model="zonos3", input=text, **kwargs)


def _send(endpoint, request):
    handler = api_server.tts_generate if endpoint == "generate" else api_server.create_speech
    return asyncio.run(handler(request, x_tts_session_id=None))


def _refused(frontend, endpoint, text, **kwargs):
    with pytest.raises(HTTPException) as exc:
        _send(endpoint, _request(endpoint, text, **kwargs))
    assert exc.value.status_code == 400
    assert not frontend.sent
    return exc.value.detail


ENDPOINTS = pytest.mark.parametrize("endpoint", ["generate", "speech"])


@ENDPOINTS
@pytest.mark.parametrize("scale", [0.9, 0.0, 4.1, float("nan"), float("inf"), -float("inf")])
def test_out_of_range_and_nonfinite_scales_fail_the_schema(frontend, endpoint, scale):
    with pytest.raises(ValidationError):
        _request(endpoint, "Say <laugh/> hello", tag_cfg_scale=scale)


@ENDPOINTS
@pytest.mark.parametrize(
    "text", ["Say hello", "Say <unknown/> hello", "Say <giggle/> hello", "Say  hello", "<Laugh/> hi"]
)
def test_guidance_needs_recognized_markup(frontend, endpoint, text):
    detail = _refused(frontend, endpoint, text, tag_cfg_scale=1.5)
    assert "requires recognized nonverbal markup" in detail and "<laugh/>" in detail


@ENDPOINTS
def test_guidance_does_not_stack_with_speaker_or_controls_cfg(frontend, endpoint):
    detail = _refused(frontend, endpoint, "Say <laugh/> hello", tag_cfg_scale=2.0, speaker_cfg_scale=1.5)
    assert "speaker_cfg_scale" in detail
    detail = _refused(frontend, endpoint, "Say <laugh/> hello", tag_cfg_scale=2.0, controls_cfg_scale=1.25)
    assert (
        detail == "tag_cfg_scale above 1.0 cannot be combined with controls_cfg_scale; set controls_cfg_scale to 1.0."
    )


@ENDPOINTS
@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0])
def test_valid_requests_reach_the_tokenizer_with_their_scale(frontend, endpoint, scale):
    text = "So I said <laugh/> no way. <shouting>Get out!</shouting>"
    _send(endpoint, _request(endpoint, text, tag_cfg_scale=scale))
    (message,) = frontend.sent
    assert message.text == text and message.sampling_params.tag_cfg_scale == scale


@ENDPOINTS
def test_default_is_the_unguided_path(frontend, endpoint):
    _send(endpoint, _request(endpoint, "plain text, no markup"))
    (message,) = frontend.sent
    assert message.sampling_params.tag_cfg_scale == 1.0
    # markup at the default scale is plain text to the guidance and serves normally
    frontend.sent.clear()
    _send(endpoint, _request(endpoint, "Say <laugh/> hello", speaker_cfg_scale=1.0))
    assert frontend.sent[0].sampling_params.tag_cfg_scale == 1.0


def test_speech_edit_refuses_tag_guidance(frontend):
    request = EditRequest(audio_base64="", target_text="Say <laugh/> hello", tag_cfg_scale=2.0)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(edit_call(api_server._EDIT_RUNTIME.edit(request)))
    assert exc.value.status_code == 400 and "not supported by /v2/edit" in exc.value.detail
    assert not frontend.sent


def test_trained_flag_reads_the_nv_training_config(tmp_path):
    api_server._model_trained_on_tags.cache_clear()
    assert api_server._model_trained_on_tags(str(tmp_path)) is False  # no config.yaml
    (tmp_path / "config.yaml").write_text(
        "data:\n  sources:\n    full: {weight: 0.5, nonverbal: false}\n    zv3subsample: {weight: 0.45, nonverbal: true}\n"
    )
    api_server._model_trained_on_tags.cache_clear()  # memoized per checkpoint path
    assert api_server._model_trained_on_tags(str(tmp_path)) is True
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.yaml").write_text("data:\n  sources:\n    full: {weight: 1.0}\n")
    assert api_server._model_trained_on_tags(str(plain)) is False
    api_server._model_trained_on_tags.cache_clear()


def test_capabilities_document(frontend, tmp_path):
    frontend.config.model_path = str(tmp_path)
    api_server._model_trained_on_tags.cache_clear()
    caps = tag_cfg_capabilities(True, api_server._model_trained_on_tags(str(frontend.config.model_path)))
    assert caps == {
        "available": True,
        "trained": False,
        "tags": list(NONVERBAL_MARKUP_TAGS),
        "min_scale": 1.0,
        "max_scale": 4.0,
        "default_scale": 1.0,
    }
    api_server._model_trained_on_tags.cache_clear()


def test_laci_refuses_tag_guidance(frontend):
    frontend.config.laci_heads = {"en_us": "heads.pt"}
    detail = _refused(frontend, "generate", "Say <laugh/> hello", tag_cfg_scale=2.0, laci=True)
    assert detail == "LACI needs tag_cfg_scale = 1"
    _send("generate", _request("generate", "Say <laugh/> hello", tag_cfg_scale=1.0, laci=True))  # markup at 1.0 is fine
    (message,) = frontend.sent
    assert message.laci is True and message.sampling_params.tag_cfg_scale == 1.0
