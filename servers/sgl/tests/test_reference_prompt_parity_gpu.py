"""zonos3.reference prompt plans against the sgl request path (tokenizer worker + scheduler), per fixture request.

Skipped unless ZONOS3_TEST_MODEL points at a checkpoint, e.g.

    ZONOS3_TEST_MODEL=hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/mect_60k uv run pytest -m gpu \
        servers/sgl/tests/test_reference_prompt_parity_gpu.py

The sgl side resolves each request with the api_server helpers, builds the text / silence rows with the tokenizer
worker's functions (zonos3_sgl.tokenizer.server), assembles the TTSUserMsg as the worker does and lets the scheduler's
``_schedule_prefill`` add the speaker rows and the CFG twin. The conditional and twin input_ids, speaker vectors (1e-6),
condition rows and steering deltas must equal the reference's ``plan_prompt``. A fixture whose ``requires`` the loaded
model and assets do not meet is skipped.
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
EMOTION_DIR = REPO / "steering" / "emotion_directions_mect60k_axes"
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
    return lm, ReferenceAssets.load(emotion_dir=EMOTION_DIR if (EMOTION_DIR / "manifest.json").is_file() else None)


def _capabilities(lm, assets) -> set[str]:
    from zonos3.tts.superbpe import load_spec

    spec = load_spec(lm.root)
    caps = {"tag_cfg"}  # every checkpoint serves tag CFG (api_server _tag_cfg_capabilities)
    if spec is not None and spec.get("phonemes") is not None:
        caps.add("phoneme_tokens")
    if assets.emotion_directions is not None:
        caps.add("emotion")
    return caps


def _sgl_reqs(tts, request: dict):
    """The scheduler's TTSReq (and CFG twin) for a v2 request on the sgl path."""
    from zonos3.tokenizer.textnorm import collapse_line_breaks
    from zonos3_sgl.message.tts import TTSSamplingParams, TTSUserMsg
    from zonos3_sgl.server import api_server as api
    from zonos3_sgl.tokenizer.server import (
        _plain_twin_prompt,
        _tag_cfg_plain_text,
        _text_prompt_rows,
    )
    from zonos3_sgl.tokenizer.vocoder import TTSVocoderManager

    model_config = tts.engine.model_config
    config = SimpleNamespace(model_config=model_config, model_path=MODEL, max_seq_len=tts.engine.max_seq_len,
                             tts_emotion_directions_dir=str(EMOTION_DIR), accent_directions=None, steer_bundle=None,
                             accent_adapters=None, laci_heads={})
    r = {"language": "en_us", "speaker_mode": None, "emotion_enabled": False, "emotion_sliders": None,
         "emotion_valence": 0.0, "emotion_arousal": 0.0, "emotion_strength": 1.0, "emotion_calibrated": True,
         "speaker_cfg_scale": 1.0, "emotion_cfg_scale": 1.0, "tag_cfg_scale": 1.0, **request}
    # api_server: speaker mode, condition rows, speaker embedding, emotion
    mode = api._resolve_speaker_mode(config, r["speaker_mode"])
    condition_ids = api._resolve_condition_buckets(config, r.get("condition_buckets"), r.get("condition_values"))
    embedding = None
    if mode == "embedding" and r.get("speaker_embedding_base64"):
        embedding = api._compute_speaker_embedding_from_file(
            r["speaker_embedding_base64"], int(model_config.speaker_embedding_dim), r.get("speaker_embedding_name"))
    embedding, delta = api._apply_request_emotion(
        config, embedding, speaker_id=None, emotion_enabled=r["emotion_enabled"], sliders=r["emotion_sliders"],
        valence=r["emotion_valence"], arousal=r["emotion_arousal"], strength=r["emotion_strength"],
        use_calibration=r["emotion_calibrated"])
    delta = api._layout_steering_delta(config, delta, None)
    # tokenizer worker
    sampling = TTSSamplingParams(seed=r.get("seed"), speaker_cfg_scale=r["speaker_cfg_scale"],
                                 emotion_cfg_scale=r["emotion_cfg_scale"], tag_cfg_scale=r["tag_cfg_scale"])
    msg = SimpleNamespace(sampling_params=sampling, edit_codes=None,
                          silence_prefill_frames=r.get("silence_prefill_frames"))
    vocoder = TTSVocoderManager(n_codebooks=tts.n_codebooks, audio_pad_id=tts.audio_pad_id,
                                audio_codec=model_config.audio_codec)
    builder = tts._prompt_builder
    text = tts._text_normalizer.normalize(collapse_line_breaks(r["text"]), r["language"])
    t = text_rows = _text_prompt_rows(msg, text, builder=builder, vocoder_manager=vocoder, span_end_id=None)
    plain_text = _tag_cfg_plain_text(msg, text)
    plain = None if plain_text is None else _text_prompt_rows(msg, plain_text, builder=builder, vocoder_manager=vocoder,
                                                              span_end_id=None)
    t = torch.cat((builder.condition_rows(tts.condition_count, dtype=t.dtype), t), dim=0)
    prefix_length = 0
    if r.get("speaker_reference_codes") is not None:
        codes = torch.as_tensor(r["speaker_reference_codes"], dtype=torch.int32)
        codes = codes[: int(model_config.speaker_reference_max_seconds * 44100 // vocoder.hop_length)]
        prefix = builder.speaker_reference_prefix(codes)
        prefix_length = len(prefix)
        t = torch.cat((prefix, t), dim=0)
    frames = len(text_rows) - len(builder.build(text))
    prefill = None if frames <= 0 else text_rows[-frames:].to(torch.int32)
    position = -1
    if embedding is not None:
        t = torch.cat([builder.speaker_slot(dtype=t.dtype, device=t.device), t], dim=0)
        position = 0
    user = TTSUserMsg(
        uid=10_000 + len(tts.waiting_reqs), input_ids=t,
        plain_input_ids=None if plain is None else _plain_twin_prompt(t, text_rows, plain),
        speaker_reference_prefix_length=prefix_length, sampling_params=sampling, speaker_embedding=embedding,
        speaker_token_position=position, speaker_emotion_delta=delta, condition_values=condition_ids,
        condition_token_start=prefix_length + (1 if embedding is not None else 0), prefill_rows=prefill,
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
    _same_vector(req.speaker_emotion_delta, plan.hidden_delta if plan.sources else cond.proj_delta)
    _same_vector(prefill, plan.prefill_rows)
    twin = req.cfg_twin
    assert (twin is None) == (plan.twin is None)
    if twin is not None:
        assert req.cfg_scale == plan.cfg_scale
        assert torch.equal(twin.input_ids, plan.twin.input_ids)
        _same_vector(twin.speaker_embedding, plan.twin.speaker_embedding)
        assert twin.condition_values == plan.twin.condition_values
        assert twin.condition_token_start == plan.twin.condition_token_start
        steered = twin.speaker_emotion_delta is not None or twin.steer_schedule is not None
        assert steered == (plan.twin.steered and (plan.hidden_delta is not None or plan.schedule is not None))
