"""The shared request rules (zonos3.tts.plan) on a release-like context without weights: every refusal of the
cross-server fixture model/tests/fixtures/request_errors.json, and the rules where an off-by-one hides (flag rows,
the max_tokens budget at the context edge, the silence bounds, the reference cut, the decode context)."""

import json
from pathlib import Path

import pytest
import torch
from zonos3.generation import (
    DECODE_CONTEXT_FRAMES,
    REFERENCE_MAX_FRAMES,
    SILENCE_PREFILL_MAX_FRAMES,
)
from zonos3.tts import plan
from zonos3.tts.prompt import BYTE_TEXT_VOCAB_SIZE, TTSPromptBuilder, TTSPromptConfig

ERRORS = json.loads((Path(__file__).parent / "fixtures" / "request_errors.json").read_text())
N_CB, CODEBOOK = 10, 4096
# the Mini release's condition rows (hf_release_zonos3mini_20261020/lm/mect_60k/params.json)
EDGES = {
    "loudness": [-29.002, -26.861, -25.366, -23.607, -21.916, -20.505, -19.38, -18.151, -16.297],
    "fmax": [7578.125, 8000.0, 8000.01, 11062.5, 12445.94, 15328.125, 15773.438, 16623.633, 18758.496],
    "speaking_rate": [10.117, 12.123, 13.286, 14.25, 15.14, 16.079, 17.094, 18.489, 21.208],
}


class Context:
    """A release-like plan context: reference + embedding modes, three 10-bucket condition rows, byte text (no
    phoneme tokens), no accents, no LACI heads, no steering bundle."""

    n_codebooks, codebook_size, hidden_size = N_CB, CODEBOOK, 1024
    audio_pad_id, text_vocab, span_end_id = CODEBOOK + 1, BYTE_TEXT_VOCAB_SIZE, None
    condition_names = ("loudness", "fmax", "speaking_rate")
    condition_n_buckets = 10
    condition_edges = EDGES
    speaker_modes = ("reference", "embedding")
    text_tokenizer_spec = None
    accent_names = ()
    steer_bundle, steer_hint, steering_sources = None, "start it with --steer-bundle", ()
    laci_languages = ()
    reference_max_frames = REFERENCE_MAX_FRAMES
    silence_available = True

    def __init__(self, max_seqlen: int = 32768, flags: dict | None = None):
        self.max_seqlen = self.max_frames = max_seqlen
        self.condition_flags = flags or {}
        self._builder = TTSPromptBuilder(TTSPromptConfig(n_codebooks=N_CB, codebook_sizes=(CODEBOOK,) * N_CB,
                                                         audio_pad_id=self.audio_pad_id, text_vocab=self.text_vocab))

    def prompt_builder(self):
        return self._builder

    def normalize_text(self, text, language, enabled):
        return text

    def silence_frame(self):
        return torch.arange(N_CB, dtype=torch.int32)


def run(request: dict, ctx: Context | None = None, embedding=None, codes=None) -> plan.PromptPlan:
    ctx = ctx or Context()
    return plan.plan_prompt(ctx, request, lambda req, mode: plan.Speaker(mode, embedding, codes))


@pytest.mark.parametrize("entry", ERRORS, ids=[e["name"] for e in ERRORS])
def test_every_fixture_refusal(entry):
    assert set(entry["requires"]) <= {"conditions", "no_phoneme_tokens"}
    with pytest.raises(ValueError) as exc:
        run(entry["request"])
    assert entry["error"] in str(exc.value)


def test_flag_rows_take_their_values_or_unknown():
    ctx = Context(flags={"speaking_rate": {"values": [0, 1]}})
    assert plan.resolve_condition_buckets(ctx, {"speaking_rate": 1}, None) == [10, 10, 1]
    assert plan.resolve_condition_buckets(ctx, {"speaking_rate": 10}, None) == [10, 10, 10]
    assert plan.resolve_condition_buckets(ctx, None, {"speaking_rate": 0.5, "fmax": 8000.0}) == [10, 2, 1]
    assert plan.resolve_condition_buckets(ctx, None, {"speaking_rate": 0.49}) == [10, 10, 0]
    with pytest.raises(ValueError, match=r"^condition_buckets\[speaking_rate\] is a flag: one of \[0, 1\] or 10"):
        plan.resolve_condition_buckets(ctx, {"speaking_rate": 2}, None)


def test_max_tokens_shrinks_to_the_context_left():
    rows = len(run({"text": "Hi.", "silence_prefill_frames": 0}).cond.input_ids)
    assert run({"text": "Hi.", "silence_prefill_frames": 0}, Context(rows + 1)).max_tokens == 1
    assert run({"text": "Hi.", "silence_prefill_frames": 0, "max_tokens": 7}, Context(rows + 100)).max_tokens == 7
    with pytest.raises(ValueError, match=f"^TTS input len {rows} exceeds {rows}$"):
        run({"text": "Hi.", "silence_prefill_frames": 0}, Context(rows))


def test_silence_prefill_bounds(monkeypatch):
    monkeypatch.delenv("ZONOS3_SILENCE_PREFILL_FRAMES", raising=False)
    monkeypatch.delenv("ZONOS3_SILENCE_PREFILL_DEFAULT", raising=False)
    assert run({"text": "Hi.", "silence_prefill_frames": 0}).prefill_rows is None
    assert len(run({"text": "Hi.", "silence_prefill_frames": SILENCE_PREFILL_MAX_FRAMES}).prefill_rows) == 30
    assert len(run({"text": "Hi."}).prefill_rows) == 3
    for bad in (31, -1, 2.5):
        with pytest.raises(ValueError, match=r"^silence_prefill_frames must be 0\.\.30$"):
            plan.silence_prefill_frames(bad)
    with pytest.raises(ValueError, match=r"^silence_prefill_frames must be 0\.\.30$"):
        plan.silence_prefill_frames(31, available=False)
    assert plan.silence_prefill_frames(5, available=False) == 0
    monkeypatch.setenv("ZONOS3_SILENCE_PREFILL_DEFAULT", "off")
    assert plan.silence_prefill_frames(None) == 0 and plan.silence_prefill_frames(4) == 4


def test_long_reference_keeps_its_first_frames_and_decodes_the_default_context():
    codes = torch.randint(0, CODEBOOK, (950, N_CB), dtype=torch.int32)
    kept = plan.reference_codes(Context(), codes.tolist())
    assert torch.equal(kept, codes[:REFERENCE_MAX_FRAMES])
    p = run({"text": "Hi.", "speaker_mode": "reference", "decode_context": True, "decode_context_frames": 0},
            codes=kept)
    assert torch.equal(p.context_codes, codes[REFERENCE_MAX_FRAMES - DECODE_CONTEXT_FRAMES:REFERENCE_MAX_FRAMES])
    assert p.reference_prefix_length == REFERENCE_MAX_FRAMES + N_CB - 1
