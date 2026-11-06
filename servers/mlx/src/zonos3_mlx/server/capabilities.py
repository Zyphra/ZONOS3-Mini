"""GET /v2/capabilities, /v2/controls, /v2/health, /v1/models and POST /v2/phonemize (the sgl / ggml routes over the
GGUF metadata: servers/ggml/src/server/capabilities.cpp)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from zonos3 import languages
from zonos3.generation import (
    DECODE_CONTEXT_FRAMES,
    SAMPLE_RATE,
    TAG_CFG_MAX_SCALE,
    TAG_CFG_MIN_SCALE,
)
from zonos3.server.schemas import PhonemizeRequest
from zonos3.tts.edit_runtime import EditModel, edit_capabilities
from zonos3.tts.markup import NONVERBAL_MARKUP_TAGS
from zonos3.tts.plan import silence_prefill_default
from zonos3.tts.steering import controls_document

from zonos3_mlx.server.state import ServerState


def condition_tokens(state: ServerState) -> dict | None:
    c = state.config
    if c.condition_count == 0:
        return None
    names = list(c.condition_names)
    counts = dict(zip(names, (int(n) for n in c.condition_bucket_counts)))
    return {"enabled": True, "names": names, "flags": {}, "count": c.condition_count, "bucket_counts": counts,
                "unknown_ids": dict(counts),
                "edges": {n: [float(v) for v in c.condition_edges.get(n, ())] for n in names},
                "units": {n: c.condition_units[n] for n in names if n in c.condition_units},
                "layout": "[reference prefix | speaker row]" + "".join(f"[{n}]" for n in names) + " BOS text EOS"}


def silence_prefill(state: ServerState) -> dict:
    """sgl _silence_prefill_spec: the server default (``zonos3.tts.plan.silence_prefill_default``)."""
    hop = state.engine.decoder.hop
    default = silence_prefill_default()
    return {"available": True, "default_frames": default.frames, "default_on": default.on,
                "default_seconds": round(default.frames * hop / SAMPLE_RATE, 3), "frame_seconds": hop / SAMPLE_RATE,
                "codec": state.config.audio_codec,
                "note": "codec-matched silence frames open the audio stream before the model starts (their audio is not "
                     "emitted); a request without silence_prefill_frames gets the server default, "
                     "silence_prefill_frames=0 turns it off and silence_prefill_frames=N sets the length"}


def decoder(state: ServerState) -> dict:
    name = Path(state.engine.paths["zac"]).name
    return {"kind": "istft_mr", "label": f"Multi-resolution iSTFT head ({name}); causal", "checkpoint": name,
                "expect_step": None, "headroom_db": float(state.engine.decoder.headroom_db), "causal": True,
                "codec": state.config.audio_codec}


def phonemes(state: ServerState) -> dict:
    spec = state.config.text_tokenizer_spec
    ph = None if not spec else spec.get("phonemes")
    lex = state.lexicon
    return {
        "phoneme_tokens": None if ph is None else {"open_marker": ph["open_marker"], "close_marker": ph["close_marker"],
                                                   "symbols": list(ph["symbols"])},
        "phonemize_enabled": lex is not None,
        "phonemize_languages": [] if lex is None else lex.server_languages,
        "phonemize_guess_languages": [] if lex is None else lex.guess_languages,
    }


def edit_model(state: ServerState) -> EditModel:
    from zonos3.tts.edit import edit_marker_rows

    c = state.config
    max_spans = edit_marker_rows(c.text_extra_tokens, c.text_tokenizer_spec or None) if c.span_end_id is not None else 0
    return EditModel(frame_rate=SAMPLE_RATE / state.engine.decoder.hop, n_codebooks=c.n_codebooks,
                     audio_pad_id=c.audio_pad_id, codebook_sizes=tuple(int(s) for s in c.codebook_sizes),
                     max_spans=max_spans, speaker_row=c.speaker_embedding_dim > 0)


def edit(state: ServerState) -> dict:
    """Speech editing needs the checkpoint's span-end/marker rows, Qwen3-ASR (.venv-asr), FuseAlign (fusealign_hub in
    its worker's interpreter) and the reference encoder."""
    return edit_capabilities(edit_model(state), state.asr.installed and state.options.fusealign
                             and state.engine.has_reference_encoder)


def speaking_rate_edges(state: ServerState) -> list[float]:
    return [float(v) for v in state.config.condition_edges.get("speaking_rate", ())]


def capabilities(state: ServerState) -> dict:
    c, engine = state.config, state.engine
    modes = engine.speaker_modes
    embedding = c.speaker_embedding_dim > 0
    cfg = bool(modes) and state.options.batch >= 2
    caps = {
        "text_tokenizer_spec": c.text_tokenizer_spec or None,
        "text_normalization_enabled": engine.normalizer is not None,
        "text_norm_languages": list(languages.CODES),
        "accents": engine.accent_names,
        "speaker_enabled": bool(modes),
        "speaker_conditioning_type": "raw_reference_prefix",
        "speaker_embedding_dim": c.speaker_embedding_dim if embedding else None,
        "speaker_embedding_codec": c.speaker_embedding_codec if embedding else None,
        "n_codebooks": c.n_codebooks,
        "codebook_sizes": list(c.codebook_sizes),
        "audio_codec": c.audio_codec,
        "max_tokens": engine.max_frames,
        "stream_decode_modes": ["prefix", "whole"],
        "stream_decode_default": state.options.stream_decode_default,
        "silence_prefill": silence_prefill(state),
        "tag_cfg": {"available": state.options.batch >= 2, "trained": bool(c.tag_cfg_trained),
                    "tags": list(NONVERBAL_MARKUP_TAGS), "min_scale": TAG_CFG_MIN_SCALE, "max_scale": TAG_CFG_MAX_SCALE,
                    "default_scale": 1.0},
        "decode_context": {"available": True, "default_frames": DECODE_CONTEXT_FRAMES, "max_frames": 450,
                           "note": "decode the generated frames behind the last decode_context_frames of the "
                                   "raw-reference prompt (their audio dropped), so the first generated frame is not the "
                                   "decoder's position 0; no-op without a reference prompt"},
        "speaker_modes": modes,
        "speaker_mode_default": modes[0] if modes else None,
        "decoder": decoder(state),
        "condition_tokens": condition_tokens(state),
        "speaker_audio_upload": bool(modes) and (engine.speaker_encoder is not None or engine.has_reference_encoder),
        "speaker_cfg_supported": cfg,
        "speaker_cfg_modes": modes if cfg else [],
        "speaker_cfg_unconditional": "remove_reference_prefix_or_speaker_row_condition_rows_unknown_keep_text_and_"
                                     "generated_history",
        "speaker_cfg_formula": "uncond + scale * (cond - uncond)",
        "laci_languages": sorted(engine.laci_heads),
        "plugins": [p.name for p in state.plugins],
        "speaker_embedding_upload": embedding,
        "speaker_embedding_cache": bool(modes),
        "speaker_embedding_blend": embedding,
        "default_voices_enabled": state.options.default_voices_dir is not None,
        "projects": state.projects.capabilities() if state.projects is not None else {"enabled": False},
    }
    caps.update(edit(state))
    caps.update(phonemes(state))
    return caps


def router(state: ServerState) -> APIRouter:
    r = APIRouter()

    @r.get("/v2/health")
    async def health():
        return {"status": "ok"}

    @r.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "zonos3", "object": "model", "created": 0, "owned_by": "zyphra",
                                            "root": Path(state.options.model_path).name}]}

    @r.get("/v2/capabilities")
    async def get_capabilities():
        return capabilities(state)

    @r.get("/v2/controls")
    async def controls():
        bundle = state.engine.steer_bundle
        if bundle is None:
            raise HTTPException(status_code=404,
                                detail=f"This server has no steering bundle ({state.engine.steer_hint}).")
        return controls_document(bundle, speaker_embedding_dim=state.config.speaker_embedding_dim or None)

    @r.post("/v2/phonemize")
    async def phonemize(request: PhonemizeRequest):
        """Gold-lexicon IPA for the words of a text; on a phoneme-token checkpoint a pronunciation with a symbol
        outside its inventory is "unsupported" (no ``ipa``)."""
        lexicon = state.lexicon
        if lexicon is None:
            raise HTTPException(status_code=404, detail="This server has no pronunciation lexicon.")
        try:
            return await asyncio.to_thread(lexicon.phonemize_for, request.text, request.language,
                                           state.config.text_tokenizer_spec, guess=request.guess)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return r
