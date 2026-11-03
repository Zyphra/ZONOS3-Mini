"""Request bodies of the v2 HTTP API, transcribed from api/openapi.yaml (the contract: field names, types, defaults,
enums and ranges). Every model forbids unknown fields, so anything the schema can express fails with FastAPI's 422
``{"detail": [{loc, msg, type}]}``; state- or semantics-dependent checks stay in the handlers (400)."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from zonos3.generation import DECODE_CONTEXT_FRAMES, SAMPLING, TAG_CFG_MAX_SCALE, TAG_CFG_MIN_SCALE


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]
ConditionBuckets = Dict[str, int | None] | List[int | None] | None
ConditionValues = Dict[str, float | None] | List[float | None] | None
Controls = Dict[str, float] | None


# ----------------------------------------------------------------------------------------------------- generation
class ControlSpan(_Strict):
    """Hidden-space controls over the characters [start, end) of a request's text."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    controls: Dict[str, float]


class GenerationOptions(_Strict):
    """The generation fields GenerateRequest and SpeechRequest share."""

    language: str = "en_us"
    text_normalization: bool = True
    # Codec-matched silence frames prefilled into the audio stream (None = the server default; 0 = off; at most the
    # registry's maximum, checked by the request rules: 400).
    silence_prefill_frames: int | None = None
    # Decode the generated frames behind the last decode_context_frames of the reference prompt.
    decode_context: bool = False
    decode_context_frames: int = DECODE_CONTEXT_FRAMES     # 0 = the registry's
    # Sampling: the generation registry's defaults (zonos3.generation.SAMPLING).
    temperature: float = SAMPLING["temperature"]
    topk: int = SAMPLING["topk"]
    top_p: float = SAMPLING["top_p"]
    min_p: float = SAMPLING["min_p"]
    max_tokens: int | None = None            # None = the server's frame budget
    fade_out_ms: float = 0.0
    repetition_window: int = SAMPLING["repetition_window"]
    repetition_penalty: float = SAMPLING["repetition_penalty"]
    repetition_codebooks: int = SAMPLING["repetition_codebooks"]
    seed: Seed | None = None                 # None = drawn by the server (echoed in X-Seed)
    # "prefix" streams prefix re-decodes (matched long-context codecs), "whole" one exact chunk at EOS.
    stream_decode: Literal["prefix", "whole"] | None = None
    speaker_mode: Literal["embedding", "reference"] | None = None
    # Condition rows of the cond3 finetunes: bucket ids per name or raw values mapped through the checkpoint's edges.
    condition_buckets: ConditionBuckets = None
    condition_values: ConditionValues = None
    headroom_db: Annotated[float, Field(ge=0.0, le=24.0)] | None = None   # None = the decoder's registry value
    speaker_audio_base64: str | None = None
    speaker_audio_name: str | None = None
    # Pre-encoded reference prefix codes [[cb0..cbN], ...] for raw-reference checkpoints.
    speaker_reference_codes: Annotated[List[List[Annotated[int, Field(ge=0)]]], Field(min_length=1)] | None = None
    speaker_embedding_base64: str | None = None
    speaker_embedding_name: str | None = None
    speaker_embedding_id: str | None = None
    speaker_blend_embedding_id_a: str | None = None
    speaker_blend_embedding_id_b: str | None = None
    speaker_blend_t: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    # Guidance on the hidden-space delta of controls / control_spans through a twin without it (1 = off; needs
    # speaker_mode=embedding).
    controls_cfg_scale: float = 1.0
    # Reference CFG: uncond + scale * (cond - uncond) (1 = off).
    speaker_cfg_scale: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    # Tagged-vs-plain guidance on nonverbal markup (1 = off; above 1 needs markup and speaker / controls CFG at 1).
    tag_cfg_scale: float = Field(default=1.0, ge=TAG_CFG_MIN_SCALE, le=TAG_CFG_MAX_SCALE)
    # An accent adapter of GET /v2/capabilities "accents" (None = the base model).
    accent: str | None = None
    accent_strength: float = 1.0
    # Steering-bundle controls "<set>.<direction>" -> dose (GET /v2/controls).
    controls: Controls = None
    control_spans: List[ControlSpan] | None = None
    control_span_rate: Annotated[float, Field(gt=0.0)] | None = None     # frames per character (None = 1.0)
    # "reading": the spans follow the LACI alignment heads' reading position (X-Control-Span-Follow says which ran).
    control_span_follow: Literal["rate", "reading"] = "rate"
    control_span_lead: Annotated[float, Field(ge=0.0, le=8.0)] | None = None
    control_span_lag: Annotated[float, Field(ge=-4.0, le=8.0)] | None = None
    # LACI: guard against skips, hallucinations and truncation (servers with --laci-heads).
    laci: bool = False


class GenerateRequest(GenerationOptions):
    text: str
    stream: bool = True
    # "pcm": float32 PCM; "wav": buffered 16-bit WAV (needs stream=false).
    format: Literal["pcm", "wav"] = "pcm"


class SpeechRequest(GenerationOptions):
    """OpenAI-compatible speech request: ``voice`` is a speaker id of GET /v2/speakers."""

    model: str
    input: str
    voice: str | None = None
    response_format: Literal["pcm", "wav"] = "pcm"


# ------------------------------------------------------------------------------------------------------- speakers
class SpeakerCreateRequest(_Strict):
    label: str | None = None
    speaker_audio_base64: str | None = None
    speaker_audio_name: str | None = None
    speaker_embedding_base64: str | None = None
    speaker_embedding_name: str | None = None


# ------------------------------------------------------------------------------------------------------ phonemize
class PhonemizeRequest(_Strict):
    text: str = Field(max_length=5000)
    language: str = "en_us"
    guess: bool = True


# ----------------------------------------------------------------------------------------------------------- edit
class EditTranscribeRequest(_Strict):
    audio_base64: str
    # The audio's language (server code, as /v2/generate): the Qwen3-ASR language and FuseAlign's tokenization.
    language: str = "en_us"


class EditAlignRequest(_Strict):
    audio_base64: str
    text: str
    language: str = "en_us"


class EditWordIn(_Strict):
    """One word of the original with its timing (seconds), as /v2/edit/align returns it."""

    word: str
    start: float
    end: float
    tokens: List[str] | None = None   # edit tokens (zonos3.tts.textmatch.tokens of the word) when missing
    confidence: float | None = None


class EditRequest(_Strict):
    """Edit recorded speech so it says ``target_text``.

    Regions come from ``spans`` ([start, end] seconds) when given, otherwise from a word diff between
    ``original_text`` (Qwen3-ASR recognition when omitted) and ``target_text`` over a FuseAlign forced alignment of the
    audio.
    """

    audio_base64: str
    target_text: str
    original_text: str | None = None
    # Word timings of original_text the caller checked or corrected (in order): used instead of the server's FuseAlign
    # alignment for windowing and expected fill lengths, and returned as `words`.
    original_words: List[EditWordIn] | None = None
    spans: List[Annotated[List[float], Field(min_length=2, max_length=2)]] | None = None
    margin_seconds: float = Field(default=0.08, ge=0.0, le=1.0)
    max_span_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    language: str = "en_us"
    text_normalization: bool = True
    temperature: float = SAMPLING["temperature"]
    topk: int = SAMPLING["topk"]
    top_p: float = SAMPLING["top_p"]
    min_p: float = SAMPLING["min_p"]
    repetition_window: int = SAMPLING["repetition_window"]
    repetition_penalty: float = SAMPLING["repetition_penalty"]
    repetition_codebooks: int = SAMPLING["repetition_codebooks"]
    seed: Seed | None = None
    num_takes: int = Field(default=1, ge=1, le=4)
    # Speaker-embedding checkpoints: embed the original clip into the speaker row in front of the text (False leaves
    # the row out; the unmasked audio alone carries the voice). None = on when the checkpoint has the row: on the
    # release finetune it lowered edited WER and halved collapsed fills.
    speaker_embedding: bool | None = None
    # Condition rows (cond3 checkpoints), as in /v2/generate; unset rows are unknown.
    condition_buckets: ConditionBuckets = None
    condition_values: ConditionValues = None
    # Steering bundle controls ("<set>.<direction>" -> dose, as /v2/generate `controls`) on every regenerated region:
    # raw directions move the clip's speaker-row embedding, hidden ones steer every frame the fill generates (the
    # prefilled text and unmasked context are not steered).
    controls: Controls = None
    # One entry per `spans` entry (explicit spans only): region k uses controls + region_controls[k].
    region_controls: List[Dict[str, float] | None] | None = None
    # Clips longer than max_context_seconds are edited on word-snapped windows: each region plus context_seconds of
    # unedited audio per side (edit training saw 2-20 s clips). Shorter clips are edited whole.
    max_context_seconds: float = Field(default=16.0, gt=0.0, le=20.0)
    context_seconds: float = Field(default=4.0, ge=0.0, le=10.0)
    # The span-end token is masked until a fill reaches min_fill_ratio x its expected length. 0 = off.
    min_fill_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    # Transcribe the edited audio with Qwen3-ASR and report WER/CER against target_text (and its FuseAlign words).
    verify: bool = False
    # Tag CFG is not offered for edits: any value other than 1.0 is refused (400) rather than ignored.
    tag_cfg_scale: float = 1.0


# ------------------------------------------------------------------------------------------------------------ dub
class DubTranscribeRequest(_Strict):
    ref_id: str
    ref_lang: str | None = None


class DubWord(_Strict):
    word: str
    start: float
    end: float


class DubAnalyzeRequest(_Strict):
    ref_id: str
    ref_text: str | None = None
    # Word timings the caller already has (a Scene mixdown, a Direct take): no forced alignment of ref_text.
    ref_words: List[DubWord] | None = None
    ref_lang: str = "en"
    target_text: str
    target_lang: str = "fr"
    min_pause_s: float = 0.30


class DubVoice(_Strict):
    # "source": clone the reference speaker from its own audio; "studio": a studio voice (speaker_embedding_id, in
    # speaker_mode=embedding) with Cast-bench steering controls (raw + hidden doses).
    mode: Literal["source", "studio"] = "source"
    speaker_embedding_id: str | None = None
    controls: Controls = None


class DubSegmentIn(_Strict):
    ref_start: float
    ref_end: float
    target_text: str
    # speaking-rate bucket on the target language's scale (/v2/dub/capabilities.buckets: row buckets 0..n-1 and the
    # tempo-steered ones below / above); None = from the text's rate
    bucket: int | None = None
    voice: DubVoice | None = None  # None = the request's voice
    controls: Controls = None  # the segment's direction, added to its voice's controls


class DubGenerateRequest(_Strict):
    ref_id: str
    target_lang: str
    seed: int = 0
    segments: List[DubSegmentIn]
    lipsync: bool = False  # warp the dub onto the lips of the reference's video (POST /v2/dub/lips first)
    voice: DubVoice | None = None  # None = {"mode": "source"}


class DubRegenerateRequest(_Strict):
    dub_id: str
    segment: int
    target_text: str
    seed: int = 0
    bucket: int | None = None  # as DubSegmentIn.bucket; None = from the text's rate
    lipsync: bool = False
    voice: DubVoice | None = None  # None = the segment's voice
    controls: Controls = None  # None = the segment's controls; {} clears them


# ------------------------------------------------------------------------------------------------------- projects
class ProjectPutRequest(_Strict):
    """PUT /v2/projects/{id}: the store (zonos3.server.projects) checks media references, revisions and sizes."""

    name: str = Field(min_length=1, max_length=200)
    data: Dict[str, Any]
    base_revision: StrictInt | None = None
