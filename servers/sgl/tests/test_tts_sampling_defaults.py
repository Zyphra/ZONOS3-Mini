from zonos3.generation import SAMPLING
from zonos3.server.schemas import EditRequest, GenerateRequest, SpeechRequest
from zonos3_sgl.core import TTSSamplingParams as CoreTTSSamplingParams
from zonos3_sgl.message.tts import TTSSamplingParams as MessageTTSSamplingParams

# the generation registry's sampling defaults (model/tests/test_generation.py pins them to servers/api/openapi.json)
EXPECTED_DEFAULTS = dict(SAMPLING)


def _sampling_values(params):
    return {name: getattr(params, name) for name in EXPECTED_DEFAULTS}


def test_internal_sampling_defaults_match_tuned_configuration():
    assert _sampling_values(MessageTTSSamplingParams()) == EXPECTED_DEFAULTS
    assert _sampling_values(CoreTTSSamplingParams()) == EXPECTED_DEFAULTS


def test_api_defaults_match_tuned_configuration():
    assert _sampling_values(GenerateRequest(text="test")) == EXPECTED_DEFAULTS
    assert _sampling_values(SpeechRequest(model="zonos3", input="test")) == EXPECTED_DEFAULTS
    assert _sampling_values(EditRequest(audio_base64="", target_text="test")) == EXPECTED_DEFAULTS
