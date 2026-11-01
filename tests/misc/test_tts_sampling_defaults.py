from zonos3.core import TTSSamplingParams as CoreTTSSamplingParams
from zonos3.message.tts import TTSSamplingParams as MessageTTSSamplingParams
from zonos3.server.api_server import OpenAISpeechRequest, TTSGenerateRequest

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


def test_generate_api_defaults_match_tuned_configuration():
    request = TTSGenerateRequest(text="test")
    assert _sampling_values(request) == EXPECTED_DEFAULTS


def test_openai_speech_repetition_defaults_match_tuned_configuration():
    request = OpenAISpeechRequest(model="zonos3", input="test")
    assert {
        "repetition_window": request.repetition_window,
        "repetition_penalty": request.repetition_penalty,
        "repetition_codebooks": request.repetition_codebooks,
    } == {
        key: EXPECTED_DEFAULTS[key]
        for key in (
            "repetition_window",
            "repetition_penalty",
            "repetition_codebooks",
        )
    }
