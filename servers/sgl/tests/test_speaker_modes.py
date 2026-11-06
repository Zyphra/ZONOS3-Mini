"""speaker_mode: reference | embedding per request for checkpoints that carry both conditioning paths."""

from types import SimpleNamespace

import pytest

api_server = pytest.importorskip("zonos3_sgl.server.api_server")
from pydantic import ValidationError  # noqa: E402
from zonos3.server.schemas import GenerateRequest  # noqa: E402
from zonos3.tts import plan  # noqa: E402


def cfg(reference: bool, embedding: bool):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            speaker_reference_prefix_enabled=reference,
            speaker_enabled=embedding,
            speaker_embedding_dim=256,
            audio_codec="opus_dual_3n_600k_sched600k_r4096_step600000",
        )
    )


def mode(config, value):
    """zonos3.tts.plan.resolve_speaker_mode over the server's rules context."""
    return plan.resolve_speaker_mode(api_server._RulesContext(config).speaker_modes, value)


def test_default_prefers_the_reference_prompt_then_the_embedding():
    assert api_server._speaker_modes(cfg(True, True)) == ["reference", "embedding"]
    assert mode(cfg(True, True), None) == "reference"
    assert mode(cfg(True, True), "embedding") == "embedding"
    assert mode(cfg(True, False), None) == "reference"
    assert mode(cfg(False, True), None) == "embedding"
    assert mode(cfg(False, False), None) is None


def test_requests_for_a_missing_path_are_refused():
    with pytest.raises(ValueError):
        mode(cfg(True, False), "embedding")
    with pytest.raises(ValueError):
        mode(cfg(False, True), "reference")


def test_unknown_modes_fail_the_schema():
    with pytest.raises(ValidationError):
        GenerateRequest(text="hi", speaker_mode="sideways")


def test_output_gain_from_headroom():
    c = cfg(True, False)
    assert api_server._output_gain(c, 0.0) == 1.0
    assert abs(api_server._output_gain(c, 6.0) - 0.5012) < 1e-3
    assert abs(api_server._output_gain(c, None) - 10 ** (-3.0 / 20)) < 1e-9  # the registry decoder's headroom_db 3.0
    assert api_server._apply_gain_pcm(b"", 0.5) == b""
