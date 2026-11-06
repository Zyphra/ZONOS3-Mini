"""GET /v2/capabilities and GET /v1/models of the Python servers: each server states its facts (CapabilityFacts), the
document follows one set of rules (the sgl server's)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from zonos3 import languages
from zonos3.generation import (
    DECODE_CONTEXT_FRAMES,
    SAMPLE_RATE,
    TAG_CFG_MAX_SCALE,
    TAG_CFG_MIN_SCALE,
)
from zonos3.server.speech import STREAM_DECODE_MODES, stream_decode_default
from zonos3.tts.edit_runtime import EditModel, edit_capabilities
from zonos3.tts.markup import NONVERBAL_MARKUP_TAGS
from zonos3.tts.plan import silence_prefill_default


@dataclass(frozen=True)
class CapabilityFacts:
    """What a server knows of its checkpoint, companions and configuration."""

    text_tokenizer_spec: dict | None
    text_normalization_enabled: bool
    accents: list[str]  # the accent adapters a request may name
    speaker_modes: list[
        str
    ]  # the checkpoint's, in order: "reference" (raw reference prefix), "embedding" (speaker row)
    speaker_embedding_dim: int
    speaker_embedding_codec: str | None
    n_codebooks: int
    codebook_sizes: list[int]
    audio_codec: str
    max_tokens: int  # the frame budget of a request
    prefix_stream: bool  # progressive (prefix re-decode) streaming (speech.prefix_stream_supported)
    silence_hop: int | None  # the codec's hop when codec-matched silence frames can be prefilled, else None
    tag_cfg_available: bool
    tag_cfg_trained: bool  # zonos3.tts.markup.trained_on_tags
    decoder: dict | None  # the audio decoder (decoder_capability for the matched codecs)
    condition_tokens: dict | None  # condition_spec; None without condition rows
    speaker_audio_upload: bool  # uploaded clips can condition (an encoder for them is loaded)
    speaker_cfg_available: bool
    laci_languages: Iterable[str]  # languages with LACI heads
    plugins: list[str]
    default_voices_enabled: bool
    edit_model: EditModel
    edit_available: bool  # FuseAlign and the reference encode of speech editing
    edit_transcribe: bool  # the ASR worker is installed
    lexicon: Any  # zonos3.tts.phonemes.PhonemeLexicon, or None
    projects: dict  # zonos3.server.projects.ProjectStore.capabilities(), or {"enabled": False}


def condition_spec(
    *,
    names: Sequence[str],
    count: int,
    bucket_counts: Sequence[int],
    edges: Mapping | None,
    flags: Mapping | None,
    units: Mapping | None,
) -> dict:
    """The checkpoint's condition rows (loudness / fmax / speaking-rate buckets, flag rows): names, per-name bucket
    counts (the count is the unknown id), edges, flag values, units and the prompt layout."""
    names = [str(name) for name in names][:count]
    counts = dict(zip(names, (int(n) for n in bucket_counts)))
    flag_specs = {str(name): dict(spec or {}) for name, spec in (flags or {}).items()}
    return {
        "enabled": True,
        "names": names,
        # flag rows (the encode-start finetunes): ids restricted to the flag's values, no edges
        "flags": {name: flag_specs[name] for name in names if name in flag_specs},
        "count": count,
        "bucket_counts": counts,
        "unknown_ids": dict(counts),
        "edges": {str(name): [float(x) for x in (values or ())] for name, values in (edges or {}).items()},
        "units": dict(units or {}),
        "layout": "[reference prefix | speaker row]" + "".join(f"[{name}]" for name in names) + " BOS text EOS",
    }


def dual_codec_hop(audio_codec: str) -> int | None:
    """The hop of a matched dual codec (the codecs with codec-matched silence frames), else None."""
    from zonos3.tokenizer.matched_codec import codec_names, entry

    codec = str(audio_codec or "")
    if codec not in codec_names() or entry(codec).get("family") != "dual":
        return None
    return int(entry(codec)["hop"])


def silence_prefill_spec(audio_codec: str, hop: int | None) -> dict:
    """Whether codec-matched silence frames can be prefilled into the audio stream (``hop`` is not None), and the server
    default (zonos3.tts.plan.silence_prefill_default)."""
    codec = str(audio_codec or "")
    if hop is None:
        return {"available": False, "default_frames": None, "codec": codec}
    default = silence_prefill_default()
    return {
        "available": True,
        "default_frames": default.frames,
        "default_on": default.on,
        "default_seconds": round(default.frames * hop / SAMPLE_RATE, 3),
        "frame_seconds": hop / SAMPLE_RATE,
        "codec": codec,
        "note": "codec-matched silence frames open the audio stream before the model starts (their audio is not "
        "emitted); a request without silence_prefill_frames gets the server default, "
        "silence_prefill_frames=0 turns it off and silence_prefill_frames=N sets the length",
    }


def tag_cfg_capabilities(available: bool, trained: bool) -> dict:
    return {
        "available": available,
        "trained": trained,
        "tags": list(NONVERBAL_MARKUP_TAGS),
        "min_scale": TAG_CFG_MIN_SCALE,
        "max_scale": TAG_CFG_MAX_SCALE,
        "default_scale": 1.0,
    }


def decoder_capability(audio_codec: str) -> dict | None:
    """The audio decoder behind a matched codec (the causal iSTFT head), with its default headroom; None otherwise."""
    from zonos3.tokenizer.matched_codec import codec_names, decoder_info

    codec = str(audio_codec)
    if codec not in codec_names():
        return None
    try:
        return dict(decoder_info(codec), codec=codec)
    except Exception as exc:  # noqa: BLE001
        return {"codec": codec, "error": str(exc)}


def phoneme_capabilities(text_tokenizer_spec: dict | None, lexicon) -> dict:
    """Pronunciation input: the checkpoint's phoneme tokens (if any) and the gold lexicon (None: no /v2/phonemize)."""
    phonemes = None if not text_tokenizer_spec else text_tokenizer_spec.get("phonemes")
    return {
        # null: "(IPA)" in plain text (paren-trained checkpoints); else the U+E000 IPA U+E001 group syntax.
        "phoneme_tokens": None
        if phonemes is None
        else {
            "open_marker": phonemes["open_marker"],
            "close_marker": phonemes["close_marker"],
            "symbols": list(phonemes["symbols"]),
        },
        "phonemize_enabled": lexicon is not None,
        "phonemize_languages": [] if lexicon is None else lexicon.server_languages,
        "phonemize_guess_languages": [] if lexicon is None else lexicon.guess_languages,
    }


def capabilities_document(f: CapabilityFacts) -> dict:
    """GET /v2/capabilities (api/openapi.yaml Capabilities)."""
    modes = list(f.speaker_modes)
    embedding = "embedding" in modes
    cfg = f.speaker_cfg_available
    return {
        "text_tokenizer_spec": f.text_tokenizer_spec,
        "text_normalization_enabled": f.text_normalization_enabled,
        "text_norm_languages": list(languages.CODES),
        "accents": list(f.accents),
        "speaker_enabled": bool(modes),
        "speaker_conditioning_type": (
            "raw_reference_prefix" if "reference" in modes else "embedding" if embedding else "none"
        ),
        "speaker_embedding_dim": int(f.speaker_embedding_dim),
        "speaker_embedding_codec": f.speaker_embedding_codec,
        "n_codebooks": int(f.n_codebooks),
        "codebook_sizes": list(f.codebook_sizes),
        "audio_codec": f.audio_codec,
        "max_tokens": int(f.max_tokens),
        "stream_decode_modes": list(STREAM_DECODE_MODES) if f.prefix_stream else ["whole"],
        "stream_decode_default": stream_decode_default(f.prefix_stream),
        "silence_prefill": silence_prefill_spec(f.audio_codec, f.silence_hop),
        "tag_cfg": tag_cfg_capabilities(f.tag_cfg_available, f.tag_cfg_trained),
        "decode_context": {
            "available": True,
            "default_frames": DECODE_CONTEXT_FRAMES,
            "max_frames": 450,
            "note": "decode the generated frames behind the last decode_context_frames of the "
            "raw-reference prompt (their audio dropped), so the first generated frame is not the "
            "decoder's position 0; no-op without a reference prompt",
        },
        "speaker_modes": modes,
        "speaker_mode_default": modes[0] if modes else None,
        "decoder": f.decoder,
        "condition_tokens": f.condition_tokens,
        "speaker_audio_upload": f.speaker_audio_upload,
        "speaker_cfg_supported": cfg,
        # both paths: the twin drops the reference prefix or the speaker row
        "speaker_cfg_modes": modes if cfg else [],
        "speaker_cfg_unconditional": "remove_reference_prefix_or_speaker_row_condition_rows_unknown_keep_text_and_"
        "generated_history",
        "speaker_cfg_formula": "uncond + scale * (cond - uncond)",
        # LACI (laci=true): languages with alignment heads (--laci-heads); needs speaker / controls CFG 1
        "laci_languages": sorted(f.laci_languages),
        # server plugins (--plugins): plugin N serves /v2/N/* and /N
        "plugins": list(f.plugins),
        "speaker_embedding_upload": embedding,
        "speaker_embedding_cache": bool(modes),  # cached clips for raw-reference models too
        "speaker_embedding_blend": embedding,
        "default_voices_enabled": f.default_voices_enabled,
        # editing needs the checkpoint's span-end/marker rows and FuseAlign (fusealign_hub); Qwen3-ASR (.venv-asr) is
        # optional and enables transcription (edit_transcribe)
        **edit_capabilities(f.edit_model, f.edit_available, f.edit_transcribe),
        **phoneme_capabilities(f.text_tokenizer_spec, f.lexicon),
        # server-side studio projects (/v2/projects, /v2/media; --studio-projects-dir)
        "projects": f.projects,
    }


def models_document(model_path: str) -> dict:
    """GET /v1/models: the one model, ``root`` the checkpoint's base name."""
    root = PurePosixPath(str(model_path).rstrip("/")).name
    return {
        "object": "list",
        "data": [{"id": "zonos3", "object": "model", "created": 0, "owned_by": "zyphra", "root": root}],
    }
