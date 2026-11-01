from zonos3_sgl.core import TTSSamplingParams as CoreTTSSamplingParams
from zonos3_sgl.message.tts import TTSSamplingParams as MessageTTSSamplingParams
from zonos3_sgl.server.schemas import EditRequest, GenerateRequest, SpeechRequest

EXPECTED_DEFAULTS = {
    "temperature": 0.75,
    "topk": 80,
    "top_p": 0.69,
    "min_p": 0.0,
    "repetition_window": 96,
    "repetition_penalty": 2.4,
    "repetition_codebooks": 6,
}


def _sampling_values(params):
    return {name: getattr(params, name) for name in EXPECTED_DEFAULTS}


def test_internal_sampling_defaults_match_tuned_configuration():
    assert _sampling_values(MessageTTSSamplingParams()) == EXPECTED_DEFAULTS
    assert _sampling_values(CoreTTSSamplingParams()) == EXPECTED_DEFAULTS


def test_api_defaults_match_tuned_configuration():
    assert _sampling_values(GenerateRequest(text="test")) == EXPECTED_DEFAULTS
    assert _sampling_values(SpeechRequest(model="zonos3", input="test")) == EXPECTED_DEFAULTS
    assert _sampling_values(EditRequest(audio_base64="", target_text="test")) == EXPECTED_DEFAULTS
