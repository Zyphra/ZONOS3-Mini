"""zonos3.reference prompt plans against the sgl request path (tokenizer worker + scheduler), per fixture request.

Skipped unless ZONOS3_TEST_MODEL points at a checkpoint, e.g.

    ZONOS3_TEST_MODEL=hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/rate30_20k_r2grpo uv run pytest -m gpu \
        servers/sgl/tests/test_reference_prompt_parity_gpu.py

The sgl side resolves each request with the request rules (zonos3.tts.plan) on api_server's rules context, builds the
rows as the tokenizer worker does (zonos3_sgl.tokenizer.server: its silence prefill, plan.text_rows / prompt_rows) and
lets the scheduler's ``_schedule_prefill`` add the speaker rows and the CFG twin. The conditional and twin input_ids,
speaker vectors (1e-6), condition rows and steering deltas must equal the reference's ``plan_prompt``. A fixture whose
``requires`` the loaded model and assets do not meet is skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODEL = os.environ.get("ZONOS3_TEST_MODEL")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not MODEL, reason="set ZONOS3_TEST_MODEL to run (loads weights)"),
]

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "model" / "tests" / "fixtures" / "prompt_requests.json"
STEER_BUNDLE = REPO / "steering" / "studio_bundle_mect60k.json"
ENTRIES = json.loads(FIXTURES.read_text())


@pytest.fixture(scope="module")
def tts(reference_parity_tts):
    return reference_parity_tts


@pytest.fixture(scope="module")
def reference():
    """The reference LM's configuration and assets (prompt planning reads no weights)."""
    from zonos3.reference.lm import Zonos3LM, Zonos3LMConfig
    from zonos3.reference.synth import ReferenceAssets
    from zonos3.utils.release import resolve_dir_or_local

    root = Path(resolve_dir_or_local(MODEL))
    with torch.device("meta"):
        lm = Zonos3LM(Zonos3LMConfig.from_params(json.loads((root / "params.json").read_text())))
    lm.root = root
    return lm, ReferenceAssets.load(steer_bundle=STEER_BUNDLE if STEER_BUNDLE.is_file() else None)


def _capabilities(lm, assets) -> set[str]:
    from zonos3.tts.superbpe import load_spec

    spec = load_spec(lm.root)
    caps = {"tag_cfg"}  # every checkpoint serves tag CFG (the sgl facts' tag_cfg_available)
    if spec is not None and spec.get("phonemes") is not None:
        caps.add("phoneme_tokens")
    if assets.steer_bundle is not None:
        caps.add("controls")
    return caps


def _sgl_reqs(tts, request: dict):
    """The scheduler's TTSReq (and CFG twin) for a v2 request on the sgl path."""
    from zonos3.tokenizer.textnorm import collapse_line_breaks
    from zonos3.tts import plan
    from zonos3.utils.release import resolve_dir_or_local
    from zonos3_sgl.message.tts import TTSSamplingParams, TTSUserMsg
    from zonos3_sgl.server import api_server as api
    from zonos3_sgl.tokenizer.server import _silence_rows, _tag_cfg_plain_text
    from zonos3_sgl.tokenizer.vocoder import TTSVocoderManager

    model_config = tts.engine.model_config
    config = SimpleNamespace(model_config=model_config, model_path=str(resolve_dir_or_local(MODEL)),
                             max_seq_len=tts.engine.max_seq_len,
                             steer_bundle=str(STEER_BUNDLE) if STEER_BUNDLE.is_file() else None,
                             accent_adapters=None, laci_heads={}, tts_audio_codec=model_config.audio_codec,
                             tts_speaker_reference_max_seconds=float(model_config.speaker_reference_max_seconds))
    # api_server: the request rules on the server's context, with the speaker embedding of the request's file
    ctx = api._RulesContext(config)
    r = plan.parse_request(request)
    head = plan.request_head(ctx, r)
    embedding = None
    if head.mode == "embedding" and r.speaker_embedding_base64:
        embedding = api._compute_speaker_embedding_from_file(
            r.speaker_embedding_base64, int(model_config.speaker_embedding_dim), r.speaker_embedding_name)
    codes = plan.reference_codes(ctx, r.speaker_reference_codes)
    rules = plan.request_rules(ctx, r, head, plan.Speaker(head.mode, embedding, codes))
    # tokenizer worker
    sampling = TTSSamplingParams(seed=rules.seed, speaker_cfg_scale=r.speaker_cfg_scale,
                                 controls_cfg_scale=r.controls_cfg_scale, tag_cfg_scale=r.tag_cfg_scale)
    msg = SimpleNamespace(sampling_params=sampling, edit_codes=None, silence_prefill_frames=r.silence_prefill_frames)
    vocoder = TTSVocoderManager(n_codebooks=tts.n_codebooks, audio_pad_id=tts.audio_pad_id,
                                audio_codec=model_config.audio_codec)
    builder = tts._prompt_builder
    text = tts._text_normalizer.normalize(collapse_line_breaks(r.text), rules.language)
    prefill = _silence_rows(msg, builder=builder, vocoder_manager=vocoder)
    text_rows = plan.text_rows(builder, text, silence=prefill)
    rows = plan.prompt_rows(builder, text_rows, condition_count=len(rules.condition_ids or ()), reference_codes=codes,
                            speaker_row=rules.embedding is not None)
    plain_text = _tag_cfg_plain_text(msg, text)
    prefill = None if prefill is None else prefill.to(torch.int32)
    user = TTSUserMsg(
        uid=10_000 + len(tts.waiting_reqs), input_ids=rows.input_ids,
        plain_input_ids=None if plain_text is None else rows.with_text(
            text_rows, plan.text_rows(builder, plain_text, silence=prefill)),
        speaker_reference_prefix_length=rows.reference_prefix_length, sampling_params=sampling,
        speaker_embedding=rules.embedding, speaker_token_position=rows.speaker_row, steering_delta=rules.hidden_delta,
        steer_schedule=rules.schedule, condition_values=rules.condition_ids,
        condition_token_start=rows.condition_token_start, prefill_rows=prefill,
    )
    tts.waiting_reqs.append(user)
    batch = tts._schedule_prefill()
    try:
        return [req for req in batch.reqs if not req.is_cfg_uncond][0], prefill
    finally:
        for req in batch.reqs:
            tts.cache_manager.free_handle(req.cache_handle)
            tts.table_manager.free(req.table_idx)


def _same_vector(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        torch.testing.assert_close(a.float().cpu().reshape(-1), b.float().cpu().reshape(-1), rtol=0, atol=1e-6)


@pytest.mark.parametrize("entry", ENTRIES, ids=[e["name"] for e in ENTRIES])
def test_prompt_plan_matches_sgl(tts, reference, entry):
    from zonos3.reference.synth import plan_prompt

    lm, assets = reference
    missing = set(entry["requires"]) - _capabilities(lm, assets)
    if missing:
        pytest.skip(f"{entry['name']} needs {sorted(missing)}")
    plan = plan_prompt(lm, entry["request"], assets=assets, device="cuda")
    req, prefill = _sgl_reqs(tts, entry["request"])
    cond = plan.cond
    assert torch.equal(req.input_ids, cond.input_ids)
    _same_vector(req.speaker_embedding, cond.speaker_embedding)
    if cond.speaker_embedding is not None:
        assert req.speaker_token_position == cond.speaker_row
    assert req.condition_values == cond.condition_values
    assert req.condition_token_start == cond.condition_token_start
    _same_vector(req.steering_delta, plan.hidden_delta)
    _same_vector(prefill, plan.prefill_rows)
    twin = req.cfg_twin
    assert (twin is None) == (plan.twin is None)
    if twin is not None:
        assert req.cfg_scale == plan.cfg_scale
        assert torch.equal(twin.input_ids, plan.twin.input_ids)
        _same_vector(twin.speaker_embedding, plan.twin.speaker_embedding)
        assert twin.condition_values == plan.twin.condition_values
        assert twin.condition_token_start == plan.twin.condition_token_start
        steered = twin.steering_delta is not None or twin.steer_schedule is not None
        assert steered == (plan.twin.steered and (plan.hidden_delta is not None or plan.schedule is not None))
