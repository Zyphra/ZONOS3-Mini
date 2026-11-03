"""Reference synthesis of one v2 ``GenerateRequest``: the sgl server's request path in plain PyTorch.

``plan_prompt`` ports the request -> prompt path (api_server ``_tts_generate_msg``, the tokenizer worker of
zonos3_sgl.tokenizer.server and ``TTSScheduler._schedule_prefill``): text normalization, the text / silence-prefill rows,
condition rows, the reference prefix or speaker row, the accent adapter, steering-bundle controls and the CFG twin
(speaker, controls or tag CFG). ``synthesize`` then generates with the TTS sampler of zonos3_sgl.engine.sample (seeded
per request), LACI and span-follow on the LM's alignment readout, and decodes the whole utterance with the matched iSTFT
codec as zonos3_sgl.tokenizer.vocoder does at end of stream. Speaker ids (default voices, the session cache) are server
state and are refused here.

v2 hard-wires the former defaults of the dropped legacy fields (no speaking-rate token, no quality / background /
accurate-mode rows): ``Zonos3LMConfig.from_params`` only accepts checkpoints without those rows, so the prompt simply has
none. ``format``, ``stream`` and ``stream_decode`` are accepted and ignored (one whole decode).
"""

from __future__ import annotations

import base64
import bisect
import io
import math
import os
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from zonos3.models.accent_format import AccentAdapter, list_accent_adapters, load_adapter
from zonos3.reference.accents import AccentState
from zonos3.reference.lm import LMCache, ProbeSpec, Zonos3LM
from zonos3.speaker_cfg import guided_logits, speaker_cfg_twin
from zonos3.tts.delay import shear_up
from zonos3.tts.laci import LaciConfig, LaciHeads, LaciSession, RateBuckets
from zonos3.tts.markup import NONVERBAL_MARKUP_TAGS, has_nonverbal_markup, strip_nonverbal_markup
from zonos3.tts.probe import locate_text_span
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig, silence_rows_from_codes
from zonos3.tts.sampler import sample_tts
from zonos3.tts.span_follow import FollowConfig, SpanFollower
from zonos3.tts.steering import SteeringBundle, load_bundle, resolve_controls, schedule_delta

SAMPLE_RATE = 44100
# v2 GenerateRequest (api/openapi.yaml): every field and its default.
GENERATE_DEFAULTS: dict = {
    "text": None, "language": "en_us", "text_normalization": True, "silence_prefill_frames": None,
    "decode_context": False, "decode_context_frames": 30, "temperature": 0.75, "topk": 80, "top_p": 0.69,
    "min_p": 0.0, "max_tokens": None, "fade_out_ms": 0.0, "repetition_window": 96, "repetition_penalty": 2.4,
    "repetition_codebooks": 6, "seed": None, "stream": True, "stream_decode": None, "format": "pcm",
    "speaker_mode": None, "condition_buckets": None, "condition_values": None, "headroom_db": None,
    "speaker_audio_base64": None, "speaker_audio_name": None, "speaker_reference_codes": None,
    "speaker_embedding_base64": None, "speaker_embedding_name": None, "speaker_embedding_id": None,
    "speaker_blend_embedding_id_a": None, "speaker_blend_embedding_id_b": None, "speaker_blend_t": None,
    "controls_cfg_scale": 1.0, "speaker_cfg_scale": 1.0,
    "tag_cfg_scale": 1.0, "accent": None, "accent_strength": 1.0, "controls": None, "control_spans": None,
    "control_span_rate": None, "control_span_follow": "rate", "control_span_lead": None, "control_span_lag": None,
    "laci": False,
}
_SPEAKER_MODES = ("reference", "embedding")
_TAG_CFG_MIN_SCALE, _TAG_CFG_MAX_SCALE = 1.0, 4.0
_DECODE_OVERLAP_FRAMES = 4  # TTSVocoderManager overlap_frames: already-counted rows re-decoded at EOS


class Request(dict):
    """A v2 GenerateRequest with its defaults filled in (attribute access)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def parse_request(request: dict) -> Request:
    unknown = sorted(set(request) - set(GENERATE_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown GenerateRequest field(s) {unknown}")
    if not isinstance(request.get("text"), str):
        raise ValueError("GenerateRequest.text is required")
    return Request({**GENERATE_DEFAULTS, **request})


# ----------------------------------------------------------------------------------------------------- assets


@dataclass
class ReferenceAssets:
    """Everything a request may use besides the LM: the steering bundle, accent adapters, LACI heads per language; the
    codec decoder, the speaker encoder and the text normalizer are loaded on first use."""

    steer_bundle: SteeringBundle | None = None
    accent_adapters: dict[str, AccentAdapter] = field(default_factory=dict)
    laci_heads: dict[str, LaciHeads] = field(default_factory=dict)
    _decoders: dict = field(default_factory=dict, repr=False)
    _speaker_encoders: dict = field(default_factory=dict, repr=False)
    _normalizer: object | None = field(default=None, repr=False)

    @classmethod
    def load(cls, *, steer_bundle=None, accent_adapters=None, laci_heads: dict | None = None) -> "ReferenceAssets":
        adapters = {}
        if accent_adapters:
            for name in list_accent_adapters(accent_adapters):
                adapters[name] = load_adapter(Path(accent_adapters) / f"{name}.pt")
        return cls(
            steer_bundle=load_bundle(steer_bundle) if steer_bundle else None,
            accent_adapters=adapters,
            laci_heads={key: LaciHeads.load(path) for key, path in (laci_heads or {}).items()},
        )

    def steering_sources(self) -> tuple[int, ...]:
        """The Full AttnRes bank sources hidden deltas steer, ascending (sgl Engine._init_steering)."""
        return () if self.steer_bundle is None else tuple(self.steer_bundle.hidden_sources)

    def accent_names(self) -> list[str]:
        return list(self.accent_adapters)

    def normalizer(self):
        if self._normalizer is None:
            from zonos3.tokenizer.textnorm import TTSTextNormalizer

            self._normalizer = TTSTextNormalizer()
        return self._normalizer

    def decoder(self, codec: str, device: torch.device):
        """The matched codec's iSTFT decoder on ``device`` (zonos3.tokenizer.matched_codec)."""
        key = (codec, str(device))
        if key not in self._decoders:
            from zonos3.tokenizer import matched_codec
            from zonos3.utils.release import resolve_file

            entry = matched_codec.entry(codec)
            actual = matched_codec._sha256(resolve_file(entry["checkpoint"]))
            if actual != entry["checkpoint_sha256"]:
                raise ValueError(f"codec generator checkpoint sha256 {actual} != {entry['checkpoint_sha256']}")
            self._decoders[key] = matched_codec.load_model(codec, device)
        return self._decoders[key]

    def speaker_encoder(self, codec: str | None, dim: int, device: torch.device):
        key = (codec, dim, str(device))
        if key not in self._speaker_encoders:
            from zonos3.models.speaker_cloning import create_speaker_embedder

            self._speaker_encoders[key] = create_speaker_embedder(codec, dim, device=str(device))
        return self._speaker_encoders[key]


# ----------------------------------------------------------------------------------------------------- the plan


@dataclass
class Branch:
    """One LM input: the prompt rows, the speaker row (embedding replacing row ``speaker_row``; -1 = none), the
    condition rows (bucket ids from ``condition_token_start``), and whether the request's hidden steering (constant
    delta, span schedule / follower) applies to its generated frames."""

    input_ids: torch.Tensor
    speaker_embedding: torch.Tensor | None = None
    speaker_row: int = -1
    condition_values: list[int] | None = None
    condition_token_start: int = -1
    steered: bool = True

    def speaker_rows(self) -> dict | None:
        return None if self.speaker_embedding is None else {self.speaker_row: self.speaker_embedding}

    def condition_rows(self) -> dict | None:
        if self.condition_values is None or self.condition_token_start < 0:
            return None
        return {self.condition_token_start + i: int(v) for i, v in enumerate(self.condition_values)}


@dataclass
class PromptPlan:
    request: Request
    text: str  # normalized text
    language: str
    cond: Branch
    twin: Branch | None
    cfg_kind: str | None  # "speaker" | "controls" | "tag"
    cfg_scale: float
    reference_prefix_length: int
    prefill_rows: torch.Tensor | None  # [P, n_codebooks + 1] silence rows ending the prompt
    context_codes: torch.Tensor | None  # [K, n_codebooks] reference tail decoded as left context
    sources: tuple[int, ...]  # steered Full AttnRes bank sources
    hidden_delta: torch.Tensor | None  # constant hidden delta, flattened per source
    schedule: torch.Tensor | None  # span schedule [K, 3 + len(sources) * dim]
    follow_tokens: list[list[int]] | None  # token ranges of the schedule rows (control_span_follow "reading")
    follow_gate: list[float] | None
    accent: tuple[str, float] | None  # adapter accent and strength
    max_tokens: int
    seed: int
    laci: str | None  # LACI head key (the language)

    @property
    def speaker_rows(self) -> int:
        """Leading rows before the condition rows: the reference prefix, or the speaker row."""
        return self.reference_prefix_length + (1 if self.cond.speaker_embedding is not None else 0)

    def cond_json(self, branch: Branch | None = None) -> dict:
        """The ``.cond.json`` sidecar of ``zonos3-cli --build-prompt`` (servers/ggml/src/zonos3-cond-io.h)."""
        b = self.cond if branch is None else branch
        doc: dict = {"speaker_row": b.speaker_row if b.speaker_embedding is not None else -1}
        if doc["speaker_row"] >= 0:
            doc["speaker_embedding"] = [float(v) for v in b.speaker_embedding.reshape(-1).tolist()]
        doc["condition_row"] = b.condition_token_start if b.condition_values is not None else -1
        doc["condition_buckets"] = [int(v) for v in (b.condition_values or [])]
        # the speaker-CFG twin drops the reference prefix / speaker row; the tag and controls twins keep them
        doc["speaker_rows"] = 0 if b is not self.cond and self.cfg_kind == "speaker" else self.speaker_rows
        doc["prefill_frames"] = 0 if self.prefill_rows is None else int(len(self.prefill_rows))
        return doc


def decode_base64_blob(data: str, field_name: str) -> bytes:
    payload = data.strip()
    if payload.startswith("data:"):
        if "," not in payload:
            raise ValueError(f"Invalid data URL for {field_name}.")
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError(f"{field_name} must be valid base64-encoded data.") from exc


def load_embedding_vector(embedding_bytes: bytes, *, expected_dim: int, file_name: str | None) -> torch.Tensor:
    try:
        loaded = np.load(io.BytesIO(embedding_bytes), allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"{file_name or 'Speaker embedding file'} must be a valid .npy or .npz file saved without pickle.") from exc
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "emb" in loaded.files:
                arr = loaded["emb"]
            elif len(loaded.files) == 1:
                arr = loaded[loaded.files[0]]
            else:
                raise ValueError("Embedding archive must contain an 'emb' array or exactly one array.")
        finally:
            loaded.close()
    else:
        arr = loaded
    if isinstance(arr, np.ndarray) and arr.dtype.names:
        if "emb" in arr.dtype.names:
            arr = arr["emb"]
        elif len(arr.dtype.names) == 1:
            arr = arr[arr.dtype.names[0]]
        else:
            raise ValueError("Structured embedding arrays must contain an 'emb' field.")
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))
    if arr.ndim == 1:
        vector = arr
    elif arr.ndim == 2:
        if arr.shape[0] == 0:
            raise ValueError("Speaker embedding file is empty.")
        vector = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    else:
        raise ValueError(f"Unsupported embedding array shape {tuple(arr.shape)}. Expected (D,) or (N, D).")
    vector = np.ascontiguousarray(vector, dtype=np.float32)
    if vector.shape[-1] != expected_dim:
        raise ValueError(f"Reference embedding dimension mismatch. Model expects {expected_dim}, "
                         f"but loaded file has {vector.shape[-1]}.")
    return torch.from_numpy(vector).to(dtype=torch.float32, device="cpu")


def _speaker_modes(config) -> list[str]:
    modes = []
    if config.speaker_reference_prefix_enabled:
        modes.append("reference")
    modes.append("embedding")  # the reference's architecture always has the speaker row
    return modes


def resolve_speaker_mode(config, value) -> str:
    modes = _speaker_modes(config)
    if value in (None, ""):
        return modes[0]
    mode = str(value).strip().lower()
    if mode not in _SPEAKER_MODES:
        raise ValueError(f"speaker_mode must be one of {list(_SPEAKER_MODES)} or null")
    if mode not in modes:
        raise ValueError(f"speaker_mode={mode!r} is not available for this checkpoint (supported: {modes})")
    return mode


def resolve_condition_buckets(config, buckets, values) -> list[int]:
    """Bucket id per condition row: explicit ids win, raw values map through the edges, anything else is unknown
    (api_server ``_resolve_condition_buckets``)."""
    names, n_buckets = list(config.condition_names), config.condition_n_buckets

    def as_map(raw, what):
        if raw is None:
            return {}
        if isinstance(raw, dict):
            unknown = [str(k) for k in raw if str(k) not in names]
            if unknown:
                raise ValueError(f"{what}: unknown condition name(s) {unknown}; expected {names}")
            return {str(k): v for k, v in raw.items()}
        raw = list(raw)
        if len(raw) != len(names):
            raise ValueError(f"{what}: expected {len(names)} entries in the order {names}")
        return dict(zip(names, raw))

    ids_by_name = as_map(buckets, "condition_buckets")
    values_by_name = as_map(values, "condition_values")
    out = []
    for name in names:
        if ids_by_name.get(name) is not None:
            try:
                bucket = int(ids_by_name[name])
            except (TypeError, ValueError):
                raise ValueError(f"condition_buckets[{name}] must be an integer") from None
            if not 0 <= bucket <= n_buckets:
                raise ValueError(f"condition_buckets[{name}] must be in [0, {n_buckets - 1}] or {n_buckets} (= unknown)")
        elif values_by_name.get(name) is not None:
            try:
                value = float(values_by_name[name])
            except (TypeError, ValueError):
                raise ValueError(f"condition_values[{name}] must be a number") from None
            if not math.isfinite(value):
                raise ValueError(f"condition_values[{name}] must be finite")
            edges = config.condition_edges.get(name)
            if not edges:
                raise ValueError(f"the checkpoint carries no edges for {name}; send condition_buckets[{name}] instead")
            bucket = int(bisect.bisect_right(edges, value))
        else:
            bucket = n_buckets
        out.append(bucket)
    return out


def validate_tag_cfg(text: str, scale: float, *, speaker_cfg_scale: float, controls_cfg_scale: float) -> None:
    scale = float(scale)
    if not math.isfinite(scale) or not _TAG_CFG_MIN_SCALE <= scale <= _TAG_CFG_MAX_SCALE:
        raise ValueError(f"tag_cfg_scale must be finite and between {_TAG_CFG_MIN_SCALE} and {_TAG_CFG_MAX_SCALE}.")
    if scale == 1.0:
        return
    if not has_nonverbal_markup(text) or strip_nonverbal_markup(text) == text:
        raise ValueError(
            "tag_cfg_scale above 1.0 requires recognized nonverbal markup in the text: point tags "
            + ", ".join(f"<{tag}/>" for tag in NONVERBAL_MARKUP_TAGS)
            + " or the same names as span tags (<shouting>...</shouting>)."
        )
    if float(speaker_cfg_scale) != 1.0:
        raise ValueError("tag_cfg_scale above 1.0 cannot be combined with speaker_cfg_scale; set speaker_cfg_scale to 1.0.")
    if float(controls_cfg_scale) != 1.0:
        raise ValueError("tag_cfg_scale above 1.0 cannot be combined with controls_cfg_scale; set controls_cfg_scale to 1.0.")


def validate_phoneme_groups(spec: dict | None, text: str) -> None:
    from zonos3.tts.superbpe import PHONEME_CLOSE, PHONEME_OPEN, payload_ids

    phonemes = None if spec is None else spec.get("phonemes")
    open_, close = (PHONEME_OPEN, PHONEME_CLOSE) if phonemes is None else (phonemes["open_marker"], phonemes["close_marker"])
    if open_ not in text and close not in text:
        return
    if phonemes is None:
        raise ValueError("This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups")
    try:
        payload_ids(text, spec)
    except ValueError as exc:
        raise ValueError(f"Pronunciation groups: {exc}") from exc


def normalize_language(language: str) -> str:
    from zonos3.tokenizer.textnorm import SERVER_TO_NEMO_LANG

    normalized = str(language or "").strip().lower().replace("-", "_")
    if normalized not in SERVER_TO_NEMO_LANG:
        raise ValueError(f"Unsupported language code: {language!r}. Supported: {', '.join(SERVER_TO_NEMO_LANG)}.")
    return normalized


def _slerp_free_speaker_ids(req: Request) -> None:
    if req.speaker_embedding_id or req.speaker_blend_embedding_id_a or req.speaker_blend_embedding_id_b:
        raise ValueError("speaker ids need a server")


def _speaker_embedding(lm: Zonos3LM, req: Request, assets: ReferenceAssets, device) -> torch.Tensor | None:
    """speaker_mode=embedding: an uploaded embedding file or clip (MECT encoder) -> the speaker row's vector."""
    c = lm.config
    if req.speaker_audio_base64 and req.speaker_embedding_base64:
        raise ValueError("Provide either a reference audio file or a saved embedding file, not both.")
    if req.speaker_embedding_base64:
        return load_embedding_vector(decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64"),
                                      expected_dim=c.speaker_embedding_dim, file_name=req.speaker_embedding_name)
    if not req.speaker_audio_base64:
        return None
    from zonos3.tts.audio import _decode_audio_bytes

    wav, sample_rate = _decode_audio_bytes(decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64"))
    encoder = assets.speaker_encoder(c.speaker_embedding_codec, c.speaker_embedding_dim, device)
    with torch.inference_mode():
        output = encoder(wav, sample_rate)
    candidates = output if isinstance(output, tuple) else (output,)
    for candidate in candidates:
        candidate = candidate.squeeze(0).to(dtype=torch.float32, device="cpu")
        if candidate.numel() == c.speaker_embedding_dim:
            return candidate
    raise ValueError(f"Reference embedding dimension mismatch. Model expects {c.speaker_embedding_dim}.")


def _reference_codes(lm: Zonos3LM, req: Request, mode: str, device) -> torch.Tensor | None:
    """speaker_mode=reference: inline prefix codes, or an uploaded clip encoded with the dual codec."""
    c = lm.config
    if req.speaker_reference_codes is not None:
        codes = torch.as_tensor(req.speaker_reference_codes, dtype=torch.int32)
        if codes.ndim != 2 or codes.shape[0] == 0 or codes.shape[0] > 16384:
            raise ValueError("speaker_reference_codes must be a non-empty [frames, codebooks] list")
        if bool((codes < 0).any()):
            raise ValueError("speaker_reference_codes must be non-negative")
        from zonos3.tokenizer.matched_codec import entry

        max_frames = int(c.speaker_reference_max_seconds * SAMPLE_RATE // int(entry(c.audio_codec)["hop"]))
        return codes[:max_frames].contiguous()
    if mode != "reference" or not req.speaker_audio_base64:
        return None
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("encoding reference audio with the dual codec needs a CUDA device")
    from zonos3.tokenizer.dual_reference import encode_dual_reference
    from zonos3.tts.audio import _decode_audio_bytes

    wav, sample_rate = _decode_audio_bytes(decode_base64_blob(req.speaker_audio_base64, "speaker_audio_base64"))
    index = device.index if device.index is not None else torch.cuda.current_device()
    return encode_dual_reference(wav, sample_rate, max_seconds=c.speaker_reference_max_seconds, device=f"cuda:{index}")


def _accent(assets: ReferenceAssets, req: Request):
    """(adapter accent or None, strength) -- api_server ``_resolve_tts_request_accent``."""
    accent, strength = req.accent, float(req.accent_strength)
    if accent is None or accent == "" or accent == "none":
        return None, 1.0
    names = assets.accent_names()
    if accent not in names:
        loaded = ", ".join(names) or "none (load accent adapters)"
        raise ValueError(f"Unknown accent {accent!r}; loaded accents: {loaded}.")
    if not math.isfinite(strength):
        raise ValueError(f"accent_strength must be finite, got {strength!r}.")
    return accent, strength


def silence_frames(requested) -> int:
    """Silence prefill length: silence_prefill_frames, or the server default (ZONOS3_SILENCE_PREFILL_FRAMES, 3 frames =
    0.2 s, off with ZONOS3_SILENCE_PREFILL_DEFAULT=0)."""
    if requested is not None:
        return max(0, int(requested))
    try:
        default = max(0, int(os.environ.get("ZONOS3_SILENCE_PREFILL_FRAMES", "3")))
    except ValueError:
        default = 3
    on = os.environ.get("ZONOS3_SILENCE_PREFILL_DEFAULT", "1").strip().lower() not in ("0", "false", "no", "off")
    return default if on else 0


def silence_frame(codec: str, device) -> torch.Tensor:
    """The codec-matched silence frame (registry ``silence_frame``, else the dual encoder's steady state of 2 s of
    silence, as TTSVocoderManager.silence_frame_codes)."""
    from zonos3.tokenizer.matched_codec import entry

    frame = entry(codec).get("silence_frame")
    if frame is not None:
        return torch.tensor([int(v) for v in frame], dtype=torch.int32)
    from zonos3.tokenizer.dual_reference import encode_dual_reference

    codes = encode_dual_reference(torch.zeros(1, 2 * SAMPLE_RATE), SAMPLE_RATE, max_seconds=5.0, device=str(device))
    return codes[-2].to(torch.int32).clone()


def prompt_builder(lm: Zonos3LM) -> TTSPromptBuilder:
    from zonos3.tts.superbpe import load_spec

    c = lm.config
    spec = load_spec(lm.root) if lm.root is not None else None
    return TTSPromptBuilder(TTSPromptConfig(n_codebooks=c.n_codebooks, codebook_sizes=c.codebook_sizes,
                                            audio_pad_id=c.audio_pad_id, text_vocab=c.text_vocab,
                                            text_tokenizer_spec=spec))


def normalize_text(text: str, language: str, enabled: bool, assets: ReferenceAssets) -> str:
    from zonos3.tokenizer.textnorm import collapse_line_breaks, normalization_enabled

    text = collapse_line_breaks(text)
    if enabled and normalization_enabled():
        text = assets.normalizer().normalize(text, language)
    return text


@torch.inference_mode()
def plan_prompt(lm: Zonos3LM, request: dict, *, assets: ReferenceAssets, device=None) -> PromptPlan:
    """The prompt rows, conditioning and CFG twin the sgl server builds for ``request`` (no generation)."""
    c = lm.config
    device = torch.device(device if device is not None else lm.device)
    req = parse_request(request)
    validate_tag_cfg(req.text, req.tag_cfg_scale, speaker_cfg_scale=req.speaker_cfg_scale,
                      controls_cfg_scale=req.controls_cfg_scale)
    mode = resolve_speaker_mode(c, req.speaker_mode)
    condition_values = resolve_condition_buckets(c, req.condition_buckets, req.condition_values)
    _slerp_free_speaker_ids(req)
    if mode == "reference":
        if req.speaker_embedding_base64:
            raise ValueError("This checkpoint conditions on raw reference audio and does not accept "
                             "speaker embedding files or embedding blends.")
        embedding = None
    else:
        embedding = _speaker_embedding(lm, req, assets, device)
    reference_codes = _reference_codes(lm, req, mode, device)
    language = normalize_language(req.language)
    if req.max_tokens is not None and int(req.max_tokens) <= 0:
        raise ValueError("max_tokens must be positive.")
    max_tokens = c.max_seqlen if req.max_tokens is None else min(int(req.max_tokens), c.max_seqlen)
    accent, accent_strength = _accent(assets, req)
    sources = assets.steering_sources()
    schedule = span_chars = delta = None
    if req.controls or req.control_spans:
        bundle = assets.steer_bundle
        if bundle is None:
            raise ValueError("This server has no steering bundle (start it with --steer-bundle); `controls` are unavailable.")
        resolved = resolve_controls(bundle, req.controls, list(req.control_spans or []), text=req.text,
                                    speaker_embedding=embedding, sources=sources, hidden_size=c.dim,
                                    frames_per_char=req.control_span_rate)
        embedding, schedule, span_chars = resolved.speaker_embedding, resolved.schedule, resolved.span_chars
        delta = resolved.hidden_delta
    follow_chars = follow_gate = None
    if req.control_span_follow == "reading" and span_chars and language in assets.laci_heads:
        default = FollowConfig()
        follow_gate = [default.lead if req.control_span_lead is None else float(req.control_span_lead),
                       default.lag if req.control_span_lag is None else float(req.control_span_lag)]
        follow_chars = [[int(a), int(b)] for a, b in span_chars]
    if req.speaker_cfg_scale != 1.0:
        has_reference = mode == "reference" and reference_codes is not None
        has_embedding = mode == "embedding" and embedding is not None
        if not (has_reference or has_embedding):
            raise ValueError("Speaker CFG needs a speaker prompt: speaker_mode=reference with reference audio or codes, "
                             "or speaker_mode=embedding with a clip or an embedding")
        if req.controls_cfg_scale != 1.0:
            raise ValueError("Speaker and controls CFG cannot be combined")
    builder = prompt_builder(lm)
    validate_phoneme_groups(builder.config.text_tokenizer_spec, req.text)
    if req.laci:
        if not assets.laci_heads:
            raise ValueError("LACI is not enabled (pass --laci-heads LANG=PATH)")
        if language not in assets.laci_heads:
            raise ValueError(f"No LACI alignment heads for language {language!r} (configured: "
                             f"{', '.join(sorted(assets.laci_heads))})")
        if float(req.speaker_cfg_scale) != 1.0 or float(req.controls_cfg_scale) != 1.0:
            raise ValueError("LACI needs speaker_cfg_scale = controls_cfg_scale = 1")
        if float(req.tag_cfg_scale) != 1.0:
            raise ValueError("LACI needs tag_cfg_scale = 1")

    # ---- tokenizer worker (zonos3_sgl.tokenizer.server)
    text = normalize_text(req.text, language, bool(req.text_normalization), assets)
    frames = silence_frames(req.silence_prefill_frames)
    silence = None
    if frames > 0:
        silence = silence_rows_from_codes(silence_frame(c.audio_codec, device), frames, builder.config)

    def text_prompt_rows(t: str) -> torch.Tensor:
        prompt = builder.build(t)
        return prompt if silence is None else torch.cat((prompt, silence.to(prompt.dtype)), dim=0)

    text_rows = text_prompt_rows(text)
    plain_rows = text_prompt_rows(strip_nonverbal_markup(text)) if float(req.tag_cfg_scale) > 1.0 else None
    follow_tokens = None
    if follow_chars is not None:
        from zonos3.tts.span_follow import span_token_ranges

        follow_tokens = [list(r) for r in span_token_ranges(req.text, text, [tuple(x) for x in follow_chars],
                                                            builder.config.text_tokenizer_spec)]
    t = torch.cat((builder.condition_rows(c.condition_count, dtype=text_rows.dtype), text_rows), dim=0)
    reference_prefix_length = 0
    if reference_codes is not None:
        prefix = builder.speaker_reference_prefix(reference_codes)
        reference_prefix_length = len(prefix)
        t = torch.cat((prefix.to(t.dtype), t), dim=0)
    context_codes = None
    if req.decode_context and reference_codes is not None:
        k = max(0, min(int(req.decode_context_frames or 0), int(reference_codes.shape[0])))
        context_codes = reference_codes[int(reference_codes.shape[0]) - k:] if k > 0 else None
    speaker_row = -1
    if embedding is not None:
        t = torch.cat((builder.speaker_slot(dtype=t.dtype), t), dim=0)
        speaker_row = 0
    condition_token_start = reference_prefix_length + (1 if embedding is not None else 0)
    plain_ids = None
    if plain_rows is not None:
        lead = len(t) - len(text_rows)
        plain_ids = torch.cat((t[:lead], plain_rows.to(t.dtype)), dim=0)

    # ---- scheduler (_schedule_prefill)
    tag_cfg = float(req.tag_cfg_scale) > 1.0
    controls_cfg = delta is not None and embedding is not None and float(req.controls_cfg_scale) != 1.0
    speaker_cfg = float(req.speaker_cfg_scale) != 1.0
    if speaker_cfg and controls_cfg:
        raise ValueError("Speaker and controls CFG cannot be combined")
    if tag_cfg and (speaker_cfg or float(req.controls_cfg_scale) != 1.0):
        raise ValueError("Tag CFG cannot be combined with speaker or controls CFG")
    cond = Branch(t.to(torch.int32), embedding, speaker_row if embedding is not None else -1,
                  condition_values, condition_token_start, steered=True)
    twin = None
    cfg_kind, cfg_scale = None, 1.0
    if speaker_cfg:
        msg = TwinMsg(t, reference_prefix_length, embedding, max(speaker_row, 0))
        spec = speaker_cfg_twin(msg, t, n_codebooks=c.n_codebooks, audio_pad_id=c.audio_pad_id, text_vocab=c.text_vocab,
                                speaker_token_position=max(speaker_row, 0), condition_values=condition_values,
                                condition_token_start=condition_token_start, condition_unknown=c.condition_n_buckets)
        twin = Branch(spec.input_ids.to(torch.int32), None, -1, spec.condition_values,
                      spec.condition_token_start, steered=True)
        cfg_kind, cfg_scale = "speaker", float(req.speaker_cfg_scale)
    elif tag_cfg:
        twin = replace(cond, input_ids=plain_ids.to(torch.int32))
        cfg_kind, cfg_scale = "tag", float(req.tag_cfg_scale)
    elif controls_cfg:
        twin = replace(cond, steered=False)
        cfg_kind, cfg_scale = "controls", float(req.controls_cfg_scale)
    prompt_len = max(len(cond.input_ids), len(twin.input_ids)) if tag_cfg else len(cond.input_ids)
    if prompt_len >= c.max_seqlen:
        raise ValueError(f"TTS input len {prompt_len} exceeds {c.max_seqlen}")
    max_tokens = min(max_tokens, c.max_seqlen - prompt_len)
    seed = int(req.seed) if req.seed is not None else secrets.randbelow(2**31)
    return PromptPlan(
        request=req, text=text, language=language, cond=cond, twin=twin, cfg_kind=cfg_kind, cfg_scale=cfg_scale,
        reference_prefix_length=reference_prefix_length,
        prefill_rows=None if silence is None else silence.to(torch.int32),
        context_codes=context_codes, sources=sources, hidden_delta=delta, schedule=schedule,
        follow_tokens=follow_tokens, follow_gate=follow_gate,
        accent=None if accent is None else (accent, accent_strength), max_tokens=max_tokens, seed=seed,
        laci=language if req.laci else None,
    )


@dataclass
class TwinMsg:
    """The fields zonos3.speaker_cfg.speaker_cfg_twin reads from a request."""

    input_ids: torch.Tensor
    speaker_reference_prefix_length: int
    speaker_embedding: torch.Tensor | None
    speaker_token_position: int


# ----------------------------------------------------------------------------------------------------- generation


@dataclass
class Sampling:
    """Per-request sampler settings (zonos3_sgl.engine.sample.TTSSampler.prepare)."""

    temperature: float = 0.75
    topk: int = 80
    top_p: float = 0.69
    min_p: float = 0.0
    repetition_window: int = 96
    repetition_penalty: float = 2.4
    repetition_codebooks: int = 6
    min_frames: int = 0

    @classmethod
    def from_request(cls, req: Request) -> "Sampling":
        return cls(float(req.temperature), int(req.topk), float(req.top_p), float(req.min_p),
                   int(req.repetition_window), float(req.repetition_penalty), int(req.repetition_codebooks))


class ProbeState:
    """The per-attempt probe state an observer reads and sets (sgl AlignProbe)."""

    def __init__(self, heads, text_start: int, text_len: int, mask=None):
        self.heads, self.text_start, self.text_len, self.mask = heads, text_start, text_len, mask
        self.steps, self.stop = 0, False

    def spec(self) -> ProbeSpec:
        return ProbeSpec(tuple(self.heads), self.text_start, self.text_len, self.mask)


@dataclass
class Attempt:
    rows: list[torch.Tensor]  # generated rows [n_codebooks + 1] (int32), the finishing row included
    eos_frame: int | None  # aligned EOS frame in generated rows


def sample_row(config, logits, sampling: Sampling, history: torch.Tensor, total_generated: int, history_rows: int,
               stop_id: int, generator: torch.Generator) -> torch.Tensor:
    """One sampled row; ``config`` supplies n_codebooks, codebook_size, codebook_sizes, audio_pad_id and text_vocab."""
    c = config
    n_cb = c.n_codebooks
    device = logits.device
    window = max(int(sampling.repetition_window), 0)
    penalty = max(float(sampling.repetition_penalty), 1.0)
    codebooks = n_cb if sampling.repetition_codebooks < 0 else min(max(sampling.repetition_codebooks, 0), n_cb)
    active = min(window, total_generated + history_rows, len(history)) if window > 0 and penalty > 1.0 and codebooks > 0 else 0
    repetition_ids = repetition_penalties = None
    if active > 0:
        recent = history[len(history) - active:, :n_cb].to(device=device, dtype=torch.int64).transpose(0, 1)
        valid = (recent >= 0) & (recent < c.codebook_size)
        valid[codebooks:] = False
        repetition_ids = torch.where(valid, recent, torch.full_like(recent, -1)).unsqueeze(0)
        repetition_penalties = torch.tensor([penalty], dtype=torch.float32, device=device)
    stop_blocked = None
    if sampling.min_frames > 0:
        stop_blocked = torch.tensor([[total_generated < sampling.min_frames + cb for cb in range(n_cb)]], device=device)
    row = sample_tts(
        logits.float().unsqueeze(0),
        torch.tensor([max(sampling.temperature, 1e-6)], dtype=torch.float32, device=device),
        torch.tensor([sampling.topk if sampling.topk >= 1 else c.codebook_size], dtype=torch.int32, device=device),
        torch.tensor([min(max(sampling.top_p, 0.0), 1.0)], dtype=torch.float32, device=device),
        torch.tensor([max(sampling.min_p, 0.0)], dtype=torch.float32, device=device),
        codebook_size=c.codebook_size, audio_pad_id=c.audio_pad_id,
        stop_token_ids=torch.tensor([stop_id], dtype=torch.int64, device=device),
        repetition_token_ids=repetition_ids, repetition_penalties=repetition_penalties,
        codebook_sizes=c.codebook_sizes, text_vocab=c.text_vocab, generators=[generator], stop_blocked=stop_blocked,
    )[0]
    return torch.tensor(row, dtype=torch.int32)


def generate_rows(
    lm: Zonos3LM,
    cond: Branch,
    *,
    sampling: Sampling,
    seed: int,
    max_tokens: int,
    stop_id: int,
    twin: Branch | None = None,
    cfg_scale: float = 1.0,
    history_rows: int = 0,
    accents: AccentState | None = None,
    steer=None,
    probe: ProbeState | None = None,
    observer=None,
    on_row=None,
) -> Attempt:
    """One request's autoregressive generation (TTSScheduler prefill + decode steps, non-overlapped): ``steer(frame)``
    is the hidden delta (flattened per source) for generated frame ``frame`` or None; ``observer.on_step`` gets the probe
    readout of every decode step (LACI, span follow); ``on_row(row, eos_frame)`` sees every generated row."""
    c = lm.config
    n_cb, pad = c.n_codebooks, c.audio_pad_id
    device = lm.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    prompt = cond.input_ids.to(torch.int32)
    has_audio_prefix = bool((prompt[-1, :n_cb] != pad).any())
    branches = [(cond, LMCache())] + ([] if twin is None else [(twin, LMCache())])

    def forward(branch: Branch, cache: LMCache, ids: torch.Tensor, frame: int | None, with_probe: bool):
        hidden = None
        if frame is not None and branch.steered and steer is not None:
            delta = steer(frame)
            if delta is not None:
                hidden = delta
        return lm(ids, cache, speaker_rows=branch.speaker_rows(),
                  condition_rows=branch.condition_rows(), hidden_deltas=hidden, accents=accents,
                  probe=probe.spec() if with_probe and probe is not None else None)

    outs = [forward(b, cache, b.input_ids, None, False) for b, cache in branches]
    rows: list[torch.Tensor] = []
    history = prompt[:, :n_cb].clone()
    eos_frame, countdown = -1, -1
    remain = max_tokens
    step = 0
    while True:
        logits = outs[0].logits[-1].float()
        if twin is not None:
            logits = guided_logits(logits, outs[1].logits[-1].float(), cfg_scale)
        total_generated = len(rows)
        token = sample_row(lm.config, logits, sampling, history, total_generated, history_rows, stop_id, generator)
        if not has_audio_prefix and total_generated < n_cb - 1:  # cold-start delay triangle -> padding
            token[total_generated + 1 : n_cb] = pad
        if step > 0 and probe is not None and observer is not None:
            probe.steps += 1
            if not probe.stop:
                pos, m = outs[0].probe[-1]
                observer.on_step(probe, probe.steps, int(pos), float(m))
            if probe.stop:
                remain = 0
        rows.append(token)
        history = torch.cat((history, token[None, :n_cb]), dim=0)
        remain -= 1
        finished = remain <= 0  # a probe stop zeroed the remaining budget
        # EOS (TTSReq.check_eos): any stop token starts the delayed countdown
        if eos_frame < 0:
            cols = [i for i, tok in enumerate(token[:n_cb].tolist()) if tok == stop_id]
            if cols:
                eos_frame = max(0, total_generated - max(cols))
                countdown = n_cb + 1
        if countdown > 0:
            countdown -= 1
            if countdown == 0:
                finished = True
        if on_row is not None:
            on_row(token, eos_frame if eos_frame >= 0 else None, finished)
        if finished:
            return Attempt(rows, eos_frame if eos_frame >= 0 else None)
        step += 1
        frame = len(rows) + history_rows
        ids = token[None]
        outs = [forward(b, cache, ids, frame, i == 0) for i, (b, cache) in enumerate(branches)]


# ----------------------------------------------------------------------------------------------------- decoding


def decode_rows(lm: Zonos3LM, assets: ReferenceAssets, rows: list, *, prefill_rows: torch.Tensor | None,
                eos_frame: int | None, context_codes: torch.Tensor | None, device) -> np.ndarray:
    """End-of-stream decode of TTSVocoderManager.decode_frames / _decode_final: the prefill rows open the stream, the
    EOS boundary is the earliest of the aligned EOS frame and every row's EOA column, the last overlap rows of the
    prefill are re-decoded with the rest, and the reference tail is decoded as left context at the stream start."""
    from zonos3.tokenizer.matched_codec import decode_model, entry

    c = lm.config
    n_cb = c.n_codebooks
    eoa = c.audio_pad_id - 1
    buffer = [] if prefill_rows is None else [[int(v) for v in r[:n_cb]] for r in prefill_rows.tolist()]
    prefill = len(buffer)
    candidates = [] if eos_frame is None else [max(0, int(eos_frame)) + prefill]
    for row in rows:
        codes = [int(v) for v in row[:n_cb].tolist()]
        buffer.append(codes)
        cols = [i for i, tok in enumerate(codes) if tok == eoa]
        if cols:
            candidates.append(max(0, len(buffer) - 1 - max(cols)))
    total = len(buffer)
    target = total - (n_cb - 1)
    if candidates:
        target = min(target, min(candidates))
    if target <= prefill:
        return np.zeros(0, dtype=np.float32)
    start = prefill - min(_DECODE_OVERLAP_FRAMES, prefill)
    raw_end = min(target + n_cb - 1, total)
    codes = shear_up(torch.tensor(buffer[start:raw_end], dtype=torch.int64), c.audio_pad_id)[: target - start]
    hop = int(entry(c.audio_codec)["hop"])
    context = 0
    if start == 0 and context_codes is not None and len(context_codes):
        context = len(context_codes)
        codes = torch.cat((torch.as_tensor(context_codes, dtype=torch.int64).reshape(-1, n_cb), codes), dim=0)
    if bool(((codes < 0) | (codes >= c.codebook_size)).any()):
        raise ValueError("decode rows hold non-codec tokens")
    model = assets.decoder(c.audio_codec, torch.device(device))
    audio = decode_model(model, codes.unsqueeze(0), entry(c.audio_codec))[0].numpy().astype(np.float32)
    return audio[context * hop:]


def output_gain(codec: str, headroom_db: float | None) -> float:
    if headroom_db is None:
        from zonos3.tokenizer.matched_codec import codec_names, decoder_headroom_db

        headroom_db = decoder_headroom_db(codec) if codec in codec_names() else 0.0
    return float(10.0 ** (-float(headroom_db) / 20.0))


def fade_out(samples: np.ndarray, fade_out_ms: float) -> np.ndarray:
    if fade_out_ms <= 0 or not len(samples):
        return samples
    n = min(len(samples), int(SAMPLE_RATE * fade_out_ms / 1000.0))
    if n <= 0:
        return samples
    samples = samples.copy()
    samples[-n:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    return samples


# ----------------------------------------------------------------------------------------------------- synthesize


@dataclass
class SynthResult:
    pcm: np.ndarray  # float32 mono 44100 Hz
    seed: int
    laci: dict | None  # LACI record (LaciSession.record) of a laci request
    rows: list = field(default_factory=list)  # generated rows (int32 [n_codebooks + 1])
    eos_frame: int | None = None
    plan: PromptPlan | None = None


def steer_fn(plan: PromptPlan, follower: SpanFollower | None, base_delta=None):
    """Hidden delta of generated frame f: the request's constant delta plus the span deltas active at f (the follower's
    or the character-rate schedule's) -- TTSReq.steer_delta; None outside hidden mode."""
    if not plan.sources:
        return None
    delta0 = plan.hidden_delta if base_delta is None else base_delta

    def steer(frame: int):
        if follower is not None:
            span = follower.delta(frame)
        elif plan.schedule is not None:
            span = schedule_delta(plan.schedule, frame)
        else:
            span = None
        delta = delta0 if span is None else (span if delta0 is None else delta0 + span)
        if delta is None:
            return None
        per_source = delta.reshape(len(plan.sources), -1)
        return {source: per_source[j] for j, source in enumerate(plan.sources)}

    return steer


@torch.inference_mode()
def synthesize(lm: Zonos3LM, request: dict, *, assets: ReferenceAssets, device=None,
               dtype: torch.dtype | None = None) -> SynthResult:
    """Generate and decode one v2 GenerateRequest."""
    device = torch.device(device if device is not None else lm.device)
    if lm.device != device:
        lm.to(device)
    if dtype is not None and lm.dtype != dtype:
        lm.cast(dtype)
    plan = plan_prompt(lm, request, assets=assets, device=device)
    c = lm.config
    req = plan.request
    sampling = Sampling.from_request(req)
    accents = None
    if plan.accent is not None:
        accents = AccentState.for_model(lm, assets.accent_adapters[plan.accent[0]], plan.accent[1])
    follower = None
    if plan.follow_tokens is not None and plan.schedule is not None:
        lead, lag = plan.follow_gate
        follower = SpanFollower(plan.follow_tokens, plan.schedule[:, 3:], FollowConfig(lead=float(lead), lag=float(lag)))
    text_start, text_len = locate_text_span(plan.cond.input_ids, c.n_codebooks)
    record = None
    if plan.laci is None:
        probe = None
        if follower is not None:
            probe = ProbeState(assets.laci_heads[plan.language].heads, text_start, text_len)
        attempt = generate_rows(lm, plan.cond, sampling=sampling, seed=plan.seed, max_tokens=plan.max_tokens,
                                stop_id=c.eoa_id, twin=plan.twin, cfg_scale=plan.cfg_scale, accents=accents,
                                steer=steer_fn(plan, follower), probe=probe, observer=follower)
        rows, eos_frame = attempt.rows, attempt.eos_frame
    else:
        rows, eos_frame, record = _laci(lm, plan, assets, sampling, accents, follower, text_start, text_len)
    pcm = decode_rows(lm, assets, rows, prefill_rows=plan.prefill_rows, eos_frame=eos_frame,
                      context_codes=plan.context_codes, device=device)
    pcm = fade_out(pcm * np.float32(output_gain(c.audio_codec, req.headroom_db)), float(req.fade_out_ms))
    return SynthResult(pcm.astype(np.float32), plan.seed, record, rows, eos_frame, plan)


def _laci(lm, plan: PromptPlan, assets, sampling, accents, follower, text_start, text_len):
    """LACI (zonos3.tts.laci): attempts until the session's result is final; a rollback re-prefills the prompt plus the
    kept rows (TTSScheduler._laci_rows) under seed + 1000 * attempt."""
    c = lm.config
    names = list(c.condition_names)
    edges = c.condition_edges.get("speaking_rate")
    rates = (RateBuckets(names.index("speaking_rate"), tuple(edges), c.condition_n_buckets)
             if c.condition_count and "speaking_rate" in names and edges else None)
    width = c.n_codebooks + 1
    prefill = plan.prefill_rows if plan.prefill_rows is not None else torch.zeros((0, width), dtype=torch.int32)
    session = LaciSession(heads=assets.laci_heads[plan.laci], cfg=LaciConfig(), prefill_rows=prefill, seed=plan.seed,
                          max_tokens=plan.max_tokens, n_codebooks=c.n_codebooks, rates=rates,
                          condition_values=None if plan.cond.condition_values is None else list(plan.cond.condition_values),
                          follower=follower)
    template = plan.cond.input_ids
    prompt = template[: len(template) - session.n_prefill]
    branch, seed, max_tokens, history_rows, delta = plan.cond, plan.seed, plan.max_tokens, 0, plan.hidden_delta
    while True:
        probe = ProbeState(session.heads.heads, text_start, text_len, mask=session.probe_mask())
        state = {}

        def on_row(row, eos, finished):
            session.append(row, eos)
            if finished:
                state["continuation"] = session.finish_attempt()
                session.drain(final=state["continuation"] is None)
            else:
                session.drain()

        generate_rows(lm, branch, sampling=sampling, seed=seed, max_tokens=max_tokens, stop_id=c.eoa_id,
                      history_rows=history_rows, accents=accents, steer=steer_fn(plan, follower, delta),
                      probe=probe, observer=session, on_row=on_row)
        cont = state["continuation"]
        if cont is None:
            break
        branch = replace(branch, input_ids=torch.cat((prompt, cont.rows.to(prompt.dtype)), dim=0),
                         condition_values=cont.condition_values if cont.condition_values is not None
                         else branch.condition_values)
        seed, max_tokens, history_rows = cont.seed, cont.max_tokens, cont.history_rows
        if cont.swap_delta:
            delta = cont.steering_delta
    rows = session.stream[session.n_prefill:]
    return rows, session.eos_frame(), session.record()
