"""Accent adapters on the GPU.

The expert side path inside the SonicMoE kernels runs on any CUDA GPU with the
pinned sonicmoe / quack packages. The three parity checks of
servers/sgl/scripts/accent_parity.py run on the real model and are skipped unless these
point at real artifacts:

    ZONOS3_TEST_MODEL          converted base (checkpoints/gdn375k)
    ZONOS3_TEST_ACCENT_DIR     directory of <accent>.pt adapters (two or more)
    ZONOS3_TEST_MERGED_MODEL   converted base + adapter checkpoint (lingua accent_adapter
                               --reconstruct, then tools/convert_lingua_checkpoint.py)
    ZONOS3_TEST_MERGED_ACCENT  the adapter that checkpoint merges
    ZONOS3_TEST_MIXED_ACCENTS  optional: comma-separated accents of the mixed batch
    ZONOS3_TEST_REFERENCE_AUDIO / ZONOS3_TEST_SPEAKER_EMBEDDING
                               optional: a wav / .npy speaker vector, so requests also
                               cycle over the raw reference prefix / speaker-embedding row
    ZONOS3_TEST_PARITY_OUT     optional: keep report.json there (default: a pytest tmp dir)

Run in the exact-mode environment of servers/sgl/scripts/env.sh (the five ZONOS3_USE_TRAINING_*
variables). The parity checks start three engines in
turn, a few minutes in all.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ENV = ("ZONOS3_TEST_MODEL", "ZONOS3_TEST_ACCENT_DIR", "ZONOS3_TEST_MERGED_MODEL", "ZONOS3_TEST_MERGED_ACCENT")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "accent_parity.py"


def _swiglu_moe(x, scores, ids, w13, w2):
    """fp32 dense reference: sum_k s_k * w2_e @ (silu(gate) * up), w13 rows interleaving gate/up."""
    out = torch.zeros(x.shape[0], w2.shape[2], dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            pre = w13[t, k] @ x[t].float()
            h = torch.nn.functional.silu(pre[0::2]) * pre[1::2]
            out[t] += scores[t, k] * (w2[t, k] @ h)
    return out


def test_expert_adapters_ride_inside_the_sonic_kernels():
    pytest.importorskip("quack")
    pytest.importorskip("sonicmoe")
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional import moe_general_routing_inputs
    from zonos3_sgl.layers.accent import sonic_experts_with_accents
    from zonos3_sgl.models.accent_adapters import ExpertAccents, ExpertRankGroup

    torch.manual_seed(0)
    dev, dtype = torch.device("cuda"), torch.bfloat16
    E, H, I, K, T = 16, 256, 1536, 2, 48
    w13 = (torch.randn(E, 2 * I, H, device=dev) * 0.06).to(dtype)
    w2 = (torch.randn(E, H, I, device=dev) * 0.03).to(dtype)
    # Slots: 0 base, 1-2 expert rank 16, 3 expert rank 64, 4 a /1 adapter (no expert factors).
    factors = {}
    groups = []
    for rank, slots in ((16, (1, 2)), (64, (3,))):
        first = torch.full((5,), -1, dtype=torch.int64)
        stacks = {}
        for weight, (out_f, in_f) in (("w13", (2 * I, H)), ("w2", (H, I))):
            up = (torch.randn(len(slots) * E, out_f, rank, device=dev) * 0.05).to(dtype)
            down = (torch.randn(len(slots) * E, rank, in_f, device=dev) * 0.05).to(dtype)
            stacks[weight] = (down.transpose(1, 2).contiguous(), up.transpose(1, 2).contiguous())
            for local, slot in enumerate(slots):
                first[slot] = local * E
                rows = slice(local * E, (local + 1) * E)
                factors[slot, weight] = (up[rows].float(), down[rows].float())
        groups.append(ExpertRankGroup(rank, first.to(dev), stacks))
    op = SimpleNamespace(_sonic_w13=w13, _sonic_w2=w2, _accent_experts=ExpertAccents(E, groups))

    x = torch.randn(T, H, device=dev).to(dtype)
    scores, ids = torch.randn(T, E, device=dev).softmax(-1).topk(K, -1)
    scores = scores / scores.sum(-1, keepdim=True)
    slots = torch.arange(T, device=dev, dtype=torch.int32) % 5
    scales = torch.tensor([0.7, 1.0, 1.3], device=dev).repeat(T)[:T] * slots.ne(0)

    def plain(rows):
        out, _ = moe_general_routing_inputs(
            x=x[rows].contiguous(),
            router_scores=scores[rows].reshape(-1).contiguous(),
            token_indices=torch.arange(len(rows), dtype=torch.int32, device=dev).repeat_interleave(K),
            expert_indices=ids[rows].reshape(-1).to(torch.int32).contiguous(),
            w1=w13.permute(1, 2, 0),
            b1=None,
            w2=w2.permute(1, 2, 0),
            b2=None,
            E=E,
            stream_id=torch.cuda.current_stream().cuda_stream,
            activation_type=ActivationType.SWIGLU,
            is_inference_mode_enabled=True,
        )
        return out

    served = sonic_experts_with_accents(op, x, scores, ids, slots, scales)
    base = plain(torch.arange(T, device=dev))

    # Rows without expert factors (no accent, or the /1 adapter): the plain kernels' bits.
    untouched = (slots == 0) | (slots == 4)
    assert torch.equal(served[untouched], base[untouched])
    assert not torch.equal(served[~untouched], base[~untouched])

    # Every routed pair runs W_e + alpha * up_e @ down_e (fp32 dense reference).
    merged13 = w13.float()[ids].clone()
    merged2 = w2.float()[ids].clone()
    for t in range(T):
        slot, alpha = int(slots[t]), float(scales[t])
        if (slot, "w13") not in factors:
            continue
        for k in range(K):
            e = int(ids[t, k])
            for weight, merged in (("w13", merged13), ("w2", merged2)):
                up, down = factors[slot, weight]
                merged[t, k] += alpha * (up[e] @ down[e])
    dense = _swiglu_moe(x, scores, ids, merged13, merged2)
    dense_base = _swiglu_moe(x, scores, ids, w13.float()[ids], w2.float()[ids])
    rows = ~untouched
    error = (served[rows].float() - dense[rows]).norm() / dense[rows].norm()
    base_error = (base.float() - dense_base).norm() / dense_base.norm()
    effect = (dense[rows] - dense_base[rows]).norm() / dense[rows].norm()
    assert error < 2 * base_error + 1e-3, (error, base_error)  # bf16 rounding, like the base kernels
    assert effect > 20 * error, (effect, error)

    # A token's bits do not depend on its batch.
    for t in (0, 1, 3, 7, 30):
        one = torch.tensor([t], device=dev)
        solo = sonic_experts_with_accents(op, x[one], scores[one], ids[one], slots[one], scales[one])
        assert torch.equal(solo[0], served[t]), t


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    if not all(os.environ.get(k) for k in ENV):
        pytest.skip(f"set {', '.join(ENV)}")
    out = Path(os.environ.get("ZONOS3_TEST_PARITY_OUT") or tmp_path_factory.mktemp("accent_parity"))
    mixed = os.environ.get("ZONOS3_TEST_MIXED_ACCENTS")
    prompts = []
    for flag, key in (
        ("--reference-audio", "ZONOS3_TEST_REFERENCE_AUDIO"),
        ("--speaker-embedding", "ZONOS3_TEST_SPEAKER_EMBEDDING"),
    ):
        if os.environ.get(key):
            prompts += [flag, os.environ[key]]
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base-model",
            os.environ["ZONOS3_TEST_MODEL"],
            "--accent-dir",
            os.environ["ZONOS3_TEST_ACCENT_DIR"],
            "--merged-model",
            os.environ["ZONOS3_TEST_MERGED_MODEL"],
            "--merged-accent",
            os.environ["ZONOS3_TEST_MERGED_ACCENT"],
            "--out-dir",
            str(out),
        ]
        + (["--mixed-accents", mixed] if mixed else [])
        + prompts,
        check=True,
    )
    return json.loads((out / "report.json").read_text())


def test_requests_without_an_accent_are_bit_equal_to_the_server_without_adapters(report):
    rows = report["a_base_rows_vs_no_adapter_server"]
    assert rows and all(row["bit_equal"] and row["tokens_equal"] for row in rows.values()), rows


def test_a_mixed_batch_equals_each_request_run_alone(report):
    rows = report["b_mixed_batch_vs_solo"]
    assert len(rows) == len(report["requests"])
    assert all(row["bit_equal"] and row["tokens_equal"] for row in rows.values()), rows


def test_strength_one_matches_the_merged_checkpoint_within_bf16_rounding(report):
    rows = report["c_adapter_strength1_vs_merged_checkpoint"]
    reference = report["c_reference_base_vs_merged_checkpoint"]
    assert rows
    for key, row in rows.items():
        # The served sum rounds differently from the merged bf16 weights, and the
        # MoE top-k turns some of those ulps into different experts; the base model
        # (the adapter's whole effect) is several times further from the merge.
        assert row["argmax_agreement"] >= 0.9, row
        assert row["mean_abs_dlogit"] * 4 < reference[key]["mean_abs_dlogit"], (row, reference[key])
