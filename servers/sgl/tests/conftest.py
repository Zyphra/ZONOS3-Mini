from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def reference_parity_tts():
    """The offline TTSLLM of the reference parity tests: one per session (a process sets up one engine)."""
    from zonos3_sgl.tts.llm import TTSLLM

    return TTSLLM(
        model_path=os.environ["ZONOS3_TEST_MODEL"], attention_backend="fa", cache_type="naive", decode_audio=False
    )
