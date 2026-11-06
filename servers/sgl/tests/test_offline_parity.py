"""Offline TTSLLM <-> server front-end parity (CPU, no model load).

Covers the three pieces that bring the offline path to parity with the server:
  * per-prompt argument broadcasting (a single tensor is a scalar, not a list),
  * text-normalization integration in _tokenize_one,
  * speaker-embedding / conditioning passthrough in offline_receive_msg.

The model is never loaded: a bare TTSLLM is built via object.__new__ and only the
few attributes each method touches are populated.
"""

from __future__ import annotations

import pytest
import torch
from zonos3.tts.prompt import BYTE_TEXT_VOCAB_SIZE, TTSPromptBuilder, TTSPromptConfig
from zonos3_sgl.message.tts import TTSSamplingParams, TTSUserMsg
from zonos3_sgl.tts.llm import TTSLLM, PendingRequest


class _StubNormalizer:
    """Records calls and rewrites a digit run so the effect is observable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def normalize(self, text: str, language: str) -> str:
        self.calls.append((text, language))
        return text.replace("123", "one two three")


def _bare_llm(text_normalizer=None) -> TTSLLM:
    llm = object.__new__(TTSLLM)
    llm._prompt_builder = TTSPromptBuilder(TTSPromptConfig(text_vocab=BYTE_TEXT_VOCAB_SIZE))
    llm._text_normalizer = text_normalizer
    # No quality conditioning on this dummy config -> no quality tokens.
    llm.quality_bucket_counts = ()
    llm.quality_features = ()
    return llm


# --------------------------------------------------------------------------- #
# _broadcast_per_prompt
# --------------------------------------------------------------------------- #
def test_broadcast_scalar_expands():
    assert TTSLLM._broadcast_per_prompt(3, 3) == [3, 3, 3]
    assert TTSLLM._broadcast_per_prompt(True, 2) == [True, True]


def test_broadcast_tensor_is_a_scalar_not_a_list():
    emb = torch.zeros(2048)
    out = TTSLLM._broadcast_per_prompt(emb, 2)
    assert len(out) == 2 and out[0] is emb and out[1] is emb


def test_broadcast_list_passes_through_and_validates_length():
    assert TTSLLM._broadcast_per_prompt([1, 2], 2) == [1, 2]
    with pytest.raises(ValueError):
        TTSLLM._broadcast_per_prompt([1], 2)


# --------------------------------------------------------------------------- #
# _tokenize_one text normalization
# --------------------------------------------------------------------------- #
def test_tokenize_applies_normalization_when_enabled():
    stub = _StubNormalizer()
    llm = _bare_llm(stub)

    out = llm._tokenize_one("num 123", language="en_us", text_normalization=True)

    assert stub.calls == [("num 123", "en_us")]
    # Matches manually normalizing then building with the same builder.
    expected = llm._prompt_builder.build("num one two three")
    assert torch.equal(out, expected)


def test_tokenize_skips_normalization_when_disabled():
    stub = _StubNormalizer()
    llm = _bare_llm(stub)

    out = llm._tokenize_one("num 123", language="en_us", text_normalization=False)

    assert stub.calls == []  # normalizer untouched
    assert torch.equal(out, llm._prompt_builder.build("num 123"))


def test_tokenize_normalization_changes_tokens():
    llm = _bare_llm(_StubNormalizer())
    norm = llm._tokenize_one("num 123", text_normalization=True)
    raw = llm._tokenize_one("num 123", text_normalization=False)
    assert norm.shape != raw.shape  # "123" -> "one two three" lengthens the prompt


def test_tokenize_without_normalizer_is_noop():
    # normalization requested but no normalizer available -> raw bytes (server's
    # graceful-degradation behavior).
    llm = _bare_llm(text_normalizer=None)
    out = llm._tokenize_one("num 123", text_normalization=True)
    assert torch.equal(out, llm._prompt_builder.build("num 123"))


# --------------------------------------------------------------------------- #
# offline_receive_msg speaker passthrough
# --------------------------------------------------------------------------- #
def _drain(llm: TTSLLM):
    llm.status_map = {}
    llm.counter = 0
    llm.prefill_budget = 10_000
    return llm.offline_receive_msg(blocking=False)


def test_offline_receive_threads_speaker_fields():
    llm = _bare_llm()
    ids = llm._prompt_builder.build("hello")
    emb = torch.zeros(2048)
    llm.pending_requests = [
        PendingRequest(
            input_ids=ids,
            sampling_params=TTSSamplingParams(),
            speaker_embedding=emb,
        )
    ]

    msgs = _drain(llm)

    assert len(msgs) == 1
    msg = msgs[0]
    assert isinstance(msg, TTSUserMsg)
    assert msg.speaker_embedding is emb


def test_offline_receive_no_voice_defaults():
    llm = _bare_llm()
    ids = llm._prompt_builder.build("hello")
    llm.pending_requests = [
        PendingRequest(input_ids=ids, sampling_params=TTSSamplingParams())
    ]

    msg = _drain(llm)[0]

    assert msg.speaker_embedding is None


# --------------------------------------------------------------------------- #
# Decoupling guard: the offline path must not drag in the HTTP server stack.
# --------------------------------------------------------------------------- #
def test_offline_path_does_not_import_server():
    import subprocess
    import sys

    code = (
        "import sys; "
        "import zonos3_sgl.tts.llm, zonos3.tts.conditioning, zonos3.tts.audio; "
        "leaked = sorted(m for m in sys.modules "
        "if m in ('fastapi', 'uvicorn', 'starlette') or m.startswith('zonos3_sgl.server')); "
        "assert not leaked, leaked; "
        "print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
