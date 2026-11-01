"""The codebook0 stop rule ignores end-of-audio tokens sampled in other codebooks."""

import pytest
import torch
from zonos3_sgl.core import TTSReq
from zonos3_sgl.message.tts import TTSSamplingParams

EOA, N = 4096, 10


def _req(policy):
    req = TTSReq(
        input_ids=torch.zeros((4, N + 1), dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=64,
        uid=0,
        sampling_params=TTSSamplingParams(),
        cache_handle=None,
        n_codebooks=N,
        audio_pad_id=EOA + 1,
        stop_id=EOA,
        eoa_policy=policy,
    )
    req.total_generated = 5
    return req


def _frame(*eoa_columns):
    return [EOA if column in eoa_columns else 7 for column in range(N)]


def _stops(req, first_frame):
    """Whether the request finishes once the stop countdown runs out."""
    if req.check_eos(first_frame):
        return True
    return any(req.check_eos(_frame()) for _ in range(N + 2))


@pytest.mark.parametrize("policy", ["any", "codebook0"])
def test_codebook0_eoa_stops_under_both_rules(policy):
    assert _stops(_req(policy), _frame(0))


def test_only_the_any_rule_stops_on_an_acoustic_codebook_eoa():
    assert _stops(_req("any"), _frame(3))
    req = _req("codebook0")
    assert not _stops(req, _frame(3))
    assert req.eos_frame == -1
