"""zonos3.reference (plain PyTorch, float32) against the sgl offline TTSLLM (bfloat16 kernels): prefill logits.

Skipped unless ZONOS3_TEST_MODEL points at the release LM, e.g.

    ZONOS3_TEST_MODEL=hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/rate30_20k_r2grpo uv run pytest -m gpu \
        servers/sgl/tests/test_reference_parity_gpu.py

The reference's prompt plan of three fixture requests (plain, speaker embedding, reference codes) is fed to both: every
prompt row's logits must agree to cosine >= 0.999. The bfloat16 activations of sgl can flip a near-tied second MoE
expert against the float32 reference, and on some prompts (e.g. the AmericanFemale default voice's reference codes
30:75) such a flip in a late layer moves a reference-prefix row far off (cosine 0.53); the fixture prompts hit none.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

MODEL = os.environ.get("ZONOS3_TEST_MODEL")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not MODEL, reason="set ZONOS3_TEST_MODEL to run (loads weights)"),
]

FIXTURES = Path(__file__).resolve().parents[3] / "model" / "tests" / "fixtures" / "prompt_requests.json"
NAMES = ("plain", "embedding", "reference_codes")


@pytest.fixture(scope="module")
def tts(reference_parity_tts):
    return reference_parity_tts


@pytest.fixture(scope="module")
def reference():
    from zonos3.reference.lm import Zonos3LM
    from zonos3.reference.synth import ReferenceAssets

    return Zonos3LM.load(MODEL, device="cuda", dtype=torch.float32), ReferenceAssets()


def _sgl_prefill_logits(tts, plan) -> torch.Tensor:
    """Every prompt row's logits from one sgl prefill of the plan's prompt (TTSLLM run with max_tokens=1)."""
    from zonos3_sgl.message.tts import TTSSamplingParams
    from zonos3_sgl.tts.llm import PendingRequest, RequestAllFinished

    model = tts.engine.model
    captured = []
    original = model.compute_logits

    def capture(hidden):
        logits = original(hidden)
        if not captured:
            captured.append(logits.float().clone())
        return logits

    # the offline path's prompts carry no speaker slot: the scheduler prepends it and shifts the condition rows
    cond = plan.cond
    lead = 1 if cond.speaker_embedding is not None else 0
    tts.pending_requests.append(
        PendingRequest(
            input_ids=cond.input_ids[lead:].clone(),
            sampling_params=TTSSamplingParams(max_tokens=1, seed=plan.seed),
            speaker_embedding=cond.speaker_embedding,
            speaker_reference_prefix_length=plan.reference_prefix_length,
            condition_values=cond.condition_values,
            condition_token_start=cond.condition_token_start - lead,
            prefill_rows=plan.prefill_rows,
        )
    )
    model.compute_logits = capture
    try:
        tts.run_forever()
    except RequestAllFinished:
        pass
    finally:
        del model.compute_logits
    return captured[0]


@pytest.mark.parametrize("name", NAMES)
def test_prefill_logits_match_sgl(tts, reference, name):
    from zonos3.reference.lm import LMCache
    from zonos3.reference.synth import plan_prompt

    lm, assets = reference
    request = next(e["request"] for e in json.loads(FIXTURES.read_text()) if e["name"] == name)
    plan = plan_prompt(lm, request, assets=assets)
    cond = plan.cond
    ours = lm(
        cond.input_ids, LMCache(), speaker_rows=cond.speaker_rows(), condition_rows=cond.condition_rows()
    ).logits.float()
    theirs = _sgl_prefill_logits(tts, plan).to(ours.device)
    assert theirs.shape == ours.shape
    cosine = torch.nn.functional.cosine_similarity(ours.reshape(len(ours), -1), theirs.reshape(len(theirs), -1), dim=-1)
    worst = int(cosine.argmin())
    assert float(cosine.min()) >= 0.999, f"{name}: row {worst} of {len(cosine)} has cosine {float(cosine[worst]):.6f}"
