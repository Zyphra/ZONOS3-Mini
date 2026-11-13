"""GET /v2/capabilities, /v2/controls, /v2/health, /v1/models and POST /v2/phonemize: the shared documents
(zonos3.server.capabilities) over this server's facts, from the GGUF metadata and the loaded companions."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from zonos3.server.capabilities import (
    CapabilityFacts,
    capabilities_document,
    condition_spec,
    models_document,
)
from zonos3.server.schemas import PhonemizeRequest
from zonos3.tts.edit_runtime import EditModel
from zonos3.tts.steering import controls_document, no_steering_message, release_document

from zonos3_mlx.server.generate import PREFIX_STREAM
from zonos3_mlx.server.state import ServerState


def condition_tokens(state: ServerState) -> dict | None:
    c = state.config
    if c.condition_count == 0:
        return None
    return condition_spec(
        names=c.condition_names,
        count=c.condition_count,
        bucket_counts=c.condition_bucket_counts,
        edges=c.condition_edges,
        flags=state.engine.condition_flags,
        units=c.condition_units,
    )


def decoders(state: ServerState) -> list[dict]:
    """Capabilities.decoders: every loaded decoder GGUF, the default first."""
    return [
        {
            "id": dec.id,
            "kind": "istft_mr",
            "label": f"Multi-resolution iSTFT head ({dec.name}); causal",
            "checkpoint": dec.name,
            "expect_step": None,
            "headroom_db": float(dec.headroom_db),
            "causal": True,
            "codec": state.config.audio_codec,
        }
        for dec in state.engine.decoders.values()
    ]


def edit_model(state: ServerState) -> EditModel:
    from zonos3.tts.edit import edit_marker_rows

    c = state.config
    max_spans = edit_marker_rows(c.text_extra_tokens, c.text_tokenizer_spec or None) if c.span_end_id is not None else 0
    return EditModel.from_config(c, hop=state.engine.decoder.hop, max_spans=max_spans)


def facts(state: ServerState) -> CapabilityFacts:
    """This server's facts: CFG (speaker and tag guidance) runs a twin in a second slot, so it needs --batch >= 2; an
    uploaded clip needs the speaker or the reference encoder; speech editing needs FuseAlign and the reference
    encoder."""
    c, engine = state.config, state.engine
    modes = list(engine.speaker_modes)
    twin = state.options.batch >= 2
    infos = decoders(state)
    return CapabilityFacts(
        text_tokenizer_spec=c.text_tokenizer_spec or None,
        text_normalization_enabled=engine.normalizer is not None,
        accents=engine.accent_names,
        speaker_modes=modes,
        speaker_embedding_dim=c.speaker_embedding_dim,
        speaker_embedding_codec=c.speaker_embedding_codec or None,
        n_codebooks=c.n_codebooks,
        codebook_sizes=list(c.codebook_sizes),
        audio_codec=c.audio_codec,
        max_tokens=engine.max_frames,
        prefix_stream=PREFIX_STREAM,
        silence_hop=engine.decoder.hop if engine.silence_available else None,
        decoder=infos[0],
        decoders=infos,
        condition_tokens=condition_tokens(state),
        speaker_audio_upload=bool(modes) and (engine.speaker_encoder is not None or engine.has_reference_encoder),
        speaker_cfg_available=bool(modes) and twin,
        laci_languages=engine.laci_languages,
        plugins=[p.name for p in state.plugins],
        default_voices_enabled=state.options.default_voices_dir is not None,
        edit_model=edit_model(state),
        edit_available=state.options.fusealign and engine.has_reference_encoder,
        edit_transcribe=state.asr.installed,
        lexicon=state.lexicon,
        projects=state.projects.capabilities() if state.projects is not None else {"enabled": False},
        steer_release=release_document(engine.steer_bundle),
    )


def router(state: ServerState) -> APIRouter:
    r = APIRouter()

    @r.get("/v2/health")
    async def health():
        return {"status": "ok"}

    @r.get("/v1/models")
    async def models():
        return models_document(state.options.model_path)

    @r.get("/v2/capabilities")
    async def get_capabilities():
        return capabilities_document(facts(state))

    @r.get("/v2/controls")
    async def controls():
        bundle = state.engine.steer_bundle
        if bundle is None:
            raise HTTPException(status_code=404, detail=no_steering_message(state.engine.steer_hint))
        return controls_document(bundle, speaker_embedding_dim=state.config.speaker_embedding_dim or None)

    @r.post("/v2/phonemize")
    async def phonemize(request: PhonemizeRequest):
        """Gold-lexicon IPA for the words of a text; on a phoneme-token checkpoint a pronunciation with a symbol
        outside its inventory is "unsupported" (no ``ipa``)."""
        lexicon = state.lexicon
        if lexicon is None:
            raise HTTPException(status_code=404, detail="This server has no pronunciation lexicon.")
        try:
            return await asyncio.to_thread(
                lexicon.phonemize_for,
                request.text,
                request.language,
                state.config.text_tokenizer_spec,
                guess=request.guess,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return r
