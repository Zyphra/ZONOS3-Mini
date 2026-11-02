"""The MLX server's models, companions and request planning.

``Engine`` owns the loaded networks (LM, codec decoder, speaker encoder, the lazily loaded reference encoder) and the
companion request logic (steering bundle, emotion and accent directions with their calibrations, accent adapters, LACI
heads, the text normalizer). ``plan_request`` turns a v2 GenerateRequest plus its pre-resolved speaker inputs into the
``zonos3.reference.synth.PromptPlan`` the scheduler runs: a port of ``zonos3.reference.synth.plan_prompt`` with the
sgl server's request rules (api_server ``_tts_generate_msg``, ``_apply_request_emotion``, ``_apply_steering_accent``,
``_apply_request_controls``, ``_span_follow``) and the tokenizer worker's prompt layout.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from zonos3.reference.synth import (
    Branch,
    PromptPlan,
    TwinMsg,
    normalize_language,
    parse_request,
    resolve_condition_buckets,
    silence_frames,
    validate_phoneme_groups,
    validate_tag_cfg,
)
from zonos3.speaker_cfg import speaker_cfg_twin
from zonos3.tts.emotion import (
    HIDDEN_SPACES,
    EmotionCalibration,
    EmotionDirections,
    apply_emotion,
    emotion_hidden_delta,
)
from zonos3.tts.laci import LaciHeads
from zonos3.tts.markup import strip_nonverbal_markup
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig, silence_rows_from_codes
from zonos3.tts.span_follow import FollowConfig
from zonos3.tts.steering import HIDDEN_SPACE, RAW_SPACE, SteeringBundle, resolve_controls

from zonos3_mlx import companions
from zonos3_mlx.models.codec_decoder import CodecDecoder
from zonos3_mlx.models.lm import LMConfig, Zonos3MLX

SAMPLE_RATE = 44100
MAX_SILENCE_PREFILL_FRAMES = 30
MAX_REFERENCE_FRAMES = int(os.environ.get("ZONOS3_MAX_REFERENCE_FRAMES", "900"))
COMPANIONS = {
    "dac": "dac.gguf", "spk_encoder": "spk-encoder.gguf", "dac_encoder": "dac-encoder.gguf",
    "ssl_teacher": "ssl-teacher.gguf", "steer": "steer.gguf", "emotion_directions": "emotion.gguf",
    "accent_adapters": "accents.gguf", "accent_directions": "accent-directions.gguf",
}


# ----------------------------------------------------------------------------------------------------- text norm


class TextNormalizer:
    """JSON-lines client of model/src/zonos3/workers/textnorm.py in its own interpreter (servers/ggml textnorm.cpp):
    one request per line, answered in order; restarted once after the worker died."""

    def __init__(self, python: str, python_src: Path):
        self.python, self.python_src = str(python), Path(python_src)
        self.script = self.python_src / "zonos3" / "workers" / "textnorm.py"
        if not self.script.is_file():
            raise FileNotFoundError(f"{self.script} does not exist (--python-src)")
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._next_id = 0

    def _start(self) -> subprocess.Popen:
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env["PYTHONPATH"] = str(self.python_src)
        return subprocess.Popen([self.python, str(self.script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=None, env=env, text=True, encoding="utf-8", bufsize=1)

    def _call(self, request: dict) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = self._start()
        self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise BrokenPipeError("text normalization worker exited")
        return json.loads(line)

    def normalize(self, text: str, language: str) -> str:
        with self._lock:
            self._next_id += 1
            request = {"id": self._next_id, "text": text, "language": language}
            for attempt in range(2):
                try:
                    response = self._call(request)
                    break
                except (OSError, ValueError) as exc:
                    if self._proc is not None:
                        self._proc.kill()
                    self._proc = None
                    if attempt:
                        raise RuntimeError(f"text normalization failed: {exc}") from exc
        if "error" in response:
            raise ValueError(f"text normalization failed: {response['error']}")
        if not isinstance(response.get("text"), str):
            raise RuntimeError("text normalization failed: the worker sent no text")
        return response["text"]

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                self._proc.kill()
                self._proc = None


# ----------------------------------------------------------------------------------------------------- engine


def find_companions(model_path: str | Path, given: dict[str, str | None]) -> dict[str, str | None]:
    """Companion paths: the flags, else the files of ``COMPANIONS`` next to the backbone (ggml main.cpp)."""
    root = Path(model_path).resolve().parent
    out = {}
    for key, name in COMPANIONS.items():
        path = given.get(key)
        if not path and (root / name).is_file():
            path = str(root / name)
        if path and not Path(path).is_file():
            raise FileNotFoundError(f"companion file {path} does not exist")
        out[key] = path
    return out


@dataclass
class ResolvedSpeaker:
    """A request's speaker inputs resolved by the HTTP layer: the mode, the speaker embedding (embedding mode) and the
    reference prefix codes (reference mode)."""

    mode: str
    embedding: torch.Tensor | None = None
    reference_codes: torch.Tensor | None = None


class Engine:
    def __init__(self, lm: Zonos3MLX, decoder: CodecDecoder, *, paths: dict[str, str | None],
                 laci_heads: dict[str, str] | None = None, max_frames: int = 2000, normalizer=None,
                 reference_idle_unload: float = 60.0):
        self.lm = lm
        self.config: LMConfig = lm.config
        self.decoder = decoder
        self.paths = paths
        self.max_frames = int(max_frames)
        self.normalizer: TextNormalizer | None = normalizer
        self.speaker_encoder = None
        if paths.get("spk_encoder"):
            from zonos3_mlx.models.speaker_encoder import SpeakerEncoder

            self.speaker_encoder = SpeakerEncoder.load(paths["spk_encoder"])
        self.bundle: SteeringBundle | None = companions.load_steer(paths["steer"]) if paths.get("steer") else None
        self.emotion: EmotionDirections | None = None
        self.emotion_calibration: EmotionCalibration | None = None
        if paths.get("emotion_directions"):
            emotion, calibration = companions.load_directions(paths["emotion_directions"])
            if not emotion.is_empty():
                self.emotion, self.emotion_calibration = emotion, calibration
        self.accent_directions: EmotionDirections | None = None
        self.accent_calibration: EmotionCalibration | None = None
        if paths.get("accent_directions"):
            directions, calibration = companions.load_directions(paths["accent_directions"])
            if directions.space not in (HIDDEN_SPACE, RAW_SPACE):
                raise ValueError(f"{paths['accent_directions']}: accent directions need an attnres_source or raw set")
            self.accent_directions, self.accent_calibration = directions, calibration
        self.accents = companions.AccentBank.load(paths["accent_adapters"]) if paths.get("accent_adapters") else None
        clash = sorted(set(self.adapter_accents()) & set(self.steering_accents()))
        if clash:
            raise ValueError(f"accent names in both the accent adapters and the accent directions: {clash}")
        if self.emotion is not None and self.emotion.space == "proj" and self.steering_sources():
            raise ValueError("hidden-space steering cannot be combined with proj-space emotion directions")
        self.laci_heads = {lang: LaciHeads.load(path) for lang, path in (laci_heads or {}).items()}
        self.laci_paths = dict(laci_heads or {})
        for lang, heads in self.laci_heads.items():
            for layer, head in heads.heads:
                if not (0 <= layer < self.config.n_layers) or self.config.uses_gdn(layer) or not (
                        0 <= head < self.config.n_heads):
                    raise ValueError(f"--laci-heads {lang}={self.laci_paths[lang]}: head ({layer}, {head}) is not an "
                                     "attention head of this model")
        self._reference_lock = threading.Lock()
        self._reference_encoder = None
        self._reference_last_use = 0.0
        self.reference_idle_unload = float(reference_idle_unload)
        spec = self.config.text_tokenizer_spec
        self._builder = TTSPromptBuilder(TTSPromptConfig(
            n_codebooks=self.config.n_codebooks, codebook_sizes=self.config.codebook_sizes,
            audio_pad_id=self.config.audio_pad_id, text_vocab=self.config.text_vocab, text_tokenizer_spec=spec))

    @classmethod
    def load(cls, model: str | Path, *, dac: str | None = None, **kwargs) -> "Engine":
        given = {"dac": dac, **{k: kwargs.pop(k) for k in list(kwargs) if k in COMPANIONS}}
        paths = find_companions(model, given)
        if not paths.get("dac"):
            raise FileNotFoundError(f"--dac <dac.gguf> is required (also auto-detected when dac.gguf sits next to {model})")
        lm = Zonos3MLX.load(model)
        return cls(lm, CodecDecoder.load(paths["dac"]), paths=paths, **kwargs)

    # -- companions -------------------------------------------------------------------------------------------------

    def prompt_builder(self) -> TTSPromptBuilder:
        return self._builder

    def adapter_accents(self) -> list[str]:
        return [] if self.accents is None else list(self.accents.names)

    def steering_accents(self) -> list[str]:
        d = self.accent_directions
        return [] if d is None else d.emotion_names + d.axis_names

    def accent_names(self) -> list[str]:
        return self.adapter_accents() + self.steering_accents()

    def steering_sources(self) -> tuple[int, ...]:
        sources = set()
        for directions in (self.emotion, self.accent_directions):
            if directions is not None and directions.space == HIDDEN_SPACE:
                sources.add(int(directions.source_index))
        if self.bundle is not None:
            sources.update(self.bundle.hidden_sources)
        return tuple(sorted(sources))

    @property
    def has_reference_encoder(self) -> bool:
        return bool(self.paths.get("dac_encoder") and self.paths.get("ssl_teacher"))

    def reference_encoder(self):
        """The codec encoder + SSL teacher, loaded on first use (call on the scheduler worker thread)."""
        with self._reference_lock:
            if self._reference_encoder is None:
                from zonos3_mlx.models.codec_encoder import CodecEncoder
                from zonos3_mlx.models.ssl_teacher import SSLTeacher

                self._reference_encoder = CodecEncoder.load(self.paths["dac_encoder"],
                                                            SSLTeacher.load(self.paths["ssl_teacher"]))
            self._reference_last_use = time.monotonic()
            return self._reference_encoder

    def unload_idle_reference_encoder(self) -> bool:
        """Free the reference encoder after --reference-idle-unload idle seconds (0 keeps it); True when freed."""
        with self._reference_lock:
            if (self._reference_encoder is None or self.reference_idle_unload <= 0
                    or time.monotonic() - self._reference_last_use < self.reference_idle_unload):
                return False
            self._reference_encoder = None
        import mlx.core as mx

        mx.clear_cache()
        return True

    def output_gain(self, headroom_db: float | None) -> float:
        return self.decoder.output_gain(headroom_db)

    # -- text ---------------------------------------------------------------------------------------------------------

    def normalize_text(self, text: str, language: str, enabled: bool) -> str:
        from zonos3.tokenizer.textnorm import collapse_line_breaks

        text = collapse_line_breaks(text)
        if self.normalizer is not None and enabled:
            text = self.normalizer.normalize(text, language)
        return text


# ----------------------------------------------------------------------------------------------------- planning


def apply_request_emotion(engine: Engine, embedding, *, speaker_id, enabled, sliders, valence, arousal, strength,
                          calibrated):
    """(speaker embedding, hidden delta) of the emotion fields (sgl ``_apply_request_emotion``)."""
    if not enabled:
        return embedding, None
    valence, arousal = float(valence), float(arousal)
    if not (bool(sliders) or valence != 0.0 or arousal != 0.0):
        return embedding, None
    directions = engine.emotion
    if directions is None:
        raise ValueError("Emotion control requested but no emotion directions are configured for this model. "
                         "Place emotion.gguf next to the backbone or pass --emotion-directions.")
    if embedding is None and directions.space != HIDDEN_SPACE:
        return embedding, None
    cal = engine.emotion_calibration
    if cal is not None and calibrated:
        sliders = {e: float(w) * cal.pole_strength(speaker_id, e, float(w)) for e, w in (sliders or {}).items()}
        valence = valence * cal.pole_strength(speaker_id, "valence", valence)
        arousal = arousal * cal.pole_strength(speaker_id, "arousal", arousal)
    if directions.space in HIDDEN_SPACES:
        return embedding, emotion_hidden_delta(directions, sliders=sliders, valence=valence, arousal=arousal,
                                               strength=strength, strict=True)
    return apply_emotion(embedding, directions=directions, sliders=sliders, valence=valence, arousal=arousal,
                         strength=strength, strict=True), None


def layout_steering_delta(engine: Engine, emotion_delta, accent_delta):
    """The emotion / accent deltas in the per-source layout of ``steering_sources`` (sgl ``_layout_steering_delta``)."""
    if emotion_delta is None and accent_delta is None:
        return None
    sources = engine.steering_sources()
    if len(sources) <= 1:
        if emotion_delta is None:
            return accent_delta
        return emotion_delta if accent_delta is None else emotion_delta + accent_delta
    hidden = (emotion_delta if emotion_delta is not None else accent_delta).numel()
    out = torch.zeros(len(sources), hidden, dtype=torch.float32)
    if emotion_delta is not None:
        out[sources.index(int(engine.emotion.source_index))] += emotion_delta
    if accent_delta is not None:
        out[sources.index(int(engine.accent_directions.source_index))] += accent_delta
    return out.flatten()


def resolve_accent(engine: Engine, accent, strength):
    """(accent or None, strength) of a request (sgl ``_resolve_tts_request_accent``)."""
    if accent is None or accent == "" or accent == "none":
        return None, 1.0
    names = engine.accent_names()
    if accent not in names:
        loaded = ", ".join(names) or "none (place accents.gguf or accent-directions.gguf next to the backbone)"
        raise ValueError(f"Unknown accent {accent!r}; loaded accents: {loaded}.")
    if not math.isfinite(float(strength)):
        raise ValueError(f"accent_strength must be finite, got {strength!r}.")
    return accent, float(strength)


def apply_steering_accent(engine: Engine, accent, strength, speaker_id, embedding, emotion_delta, calibrated):
    """(adapter accent or None, speaker embedding, delta) (sgl ``_apply_steering_accent``)."""
    steering = engine.accent_directions
    if accent is None or steering is None or accent not in engine.steering_accents():
        return accent, embedding, layout_steering_delta(engine, emotion_delta, None)
    cal = engine.accent_calibration
    dose = cal.strength(speaker_id, accent) if (cal is not None and calibrated) else 1.0
    if not float(strength) or not float(dose):
        return None, embedding, layout_steering_delta(engine, emotion_delta, None)
    if steering.space == RAW_SPACE:
        if embedding is None:
            raise ValueError(f"Accent {accent!r} is a speaker-embedding direction and needs speaker_mode=embedding.")
        embedding = apply_emotion(embedding, directions=steering, sliders={accent: float(strength) * float(dose)},
                                  strict=True)
        return None, embedding, layout_steering_delta(engine, emotion_delta, None)
    vec = steering.named.get(accent, steering.axes.get(accent))
    return None, embedding, layout_steering_delta(engine, emotion_delta, (float(strength) * float(dose)) * vec)


@dataclass
class LaciTake:
    """A dub take's LACI duration target and its speaking-rate scale (sgl _dub_take_rate_steering)."""

    target_frames: float
    adjust_rate: bool = True
    rate_edges: tuple[float, ...] | None = None
    rate_first: int = 0
    rate_bucket: int | None = None
    rate_deltas: list | None = None


@dataclass
class EditLayout:
    """A speech-edit region's prompt inputs: the clip codes, the spans, the fills so far and the region's min frames."""

    codes: torch.Tensor
    spans: list
    fills: list
    min_frames: int


@dataclass
class Plan(PromptPlan):
    """The reference PromptPlan plus the server-side fields: the request's speaker key (calibration), its LACI take
    (dub), and its edit layout (speech editing, codes-only)."""

    speaker_id: str | None = None
    laci_take: LaciTake | None = None
    edit: EditLayout | None = None
    follow_language: str | None = None


def plan_request(engine: Engine, request: dict, speaker: ResolvedSpeaker | None, *, laci_take: LaciTake | None = None,
                 edit: EditLayout | None = None, hidden_delta=None) -> Plan:
    """The prompt rows, conditioning and CFG twin of ``request`` (zonos3.reference.synth.plan_prompt with the sgl
    server's request rules). ``edit``: a speech-edit region (the edit layout, no silence prefill, stop on span-end;
    ``hidden_delta`` its steering delta). Raises ValueError for a request the server refuses (HTTP 400)."""
    c = engine.config
    req = parse_request(request)
    validate_tag_cfg(req.text, req.tag_cfg_scale, speaker_cfg_scale=req.speaker_cfg_scale,
                     emotion_cfg_scale=req.emotion_cfg_scale)
    condition_values = resolve_condition_buckets(c, req.condition_buckets, req.condition_values)
    mode = speaker.mode if speaker is not None else "reference"
    embedding = None if speaker is None else speaker.embedding
    reference_codes = None if speaker is None else speaker.reference_codes
    speaker_id = req.speaker_embedding_id
    embedding, delta = apply_request_emotion(
        engine, embedding, speaker_id=speaker_id, enabled=req.emotion_enabled, sliders=req.emotion_sliders,
        valence=req.emotion_valence, arousal=req.emotion_arousal, strength=req.emotion_strength,
        calibrated=req.emotion_calibrated)
    language = normalize_language(req.language)
    if req.max_tokens is not None and int(req.max_tokens) <= 0:
        raise ValueError("max_tokens must be positive.")
    max_tokens = engine.max_frames if req.max_tokens is None else min(int(req.max_tokens), engine.max_frames)
    accent, accent_strength = resolve_accent(engine, req.accent, req.accent_strength)
    accent, embedding, delta = apply_steering_accent(engine, accent, accent_strength, speaker_id, embedding, delta,
                                                     req.emotion_calibrated)
    sources = engine.steering_sources()
    schedule = span_chars = None
    if req.controls or req.control_spans:
        bundle = engine.bundle
        if bundle is None:
            raise ValueError("This server has no steering bundle (place steer.gguf next to the backbone or pass "
                             "--steer); `controls` are unavailable.")
        if req.emotion_enabled and (req.emotion_sliders or req.emotion_valence or req.emotion_arousal):
            raise ValueError("`controls` and the emotion_* fields cannot be combined; send the emotion as controls.")
        resolved = resolve_controls(bundle, req.controls, list(req.control_spans or []), text=req.text,
                                    speaker_embedding=embedding, sources=sources, hidden_size=c.dim,
                                    frames_per_char=req.control_span_rate)
        embedding, schedule, span_chars = resolved.speaker_embedding, resolved.schedule, resolved.span_chars
        if resolved.hidden_delta is not None:
            delta = resolved.hidden_delta if delta is None else delta + resolved.hidden_delta
    if hidden_delta is not None:
        delta = hidden_delta if delta is None else delta + hidden_delta
    follow_chars = follow_gate = None
    if req.control_span_follow == "reading" and span_chars and language in engine.laci_heads and edit is None:
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
        if req.emotion_cfg_scale != 1.0:
            raise ValueError("Speaker and emotion CFG cannot be combined")
    builder = engine.prompt_builder()
    validate_phoneme_groups(builder.config.text_tokenizer_spec, req.text)
    if req.laci:
        if not engine.laci_heads:
            raise ValueError("LACI is not enabled on this server (start it with --laci-heads LANG=PATH)")
        if language not in engine.laci_heads:
            raise ValueError(f"No LACI alignment heads for language {language!r} (configured: "
                             f"{', '.join(sorted(engine.laci_heads))})")
        if float(req.speaker_cfg_scale) != 1.0 or float(req.emotion_cfg_scale) != 1.0:
            raise ValueError("LACI needs speaker_cfg_scale = emotion_cfg_scale = 1")
        if float(req.tag_cfg_scale) != 1.0:
            raise ValueError("LACI needs tag_cfg_scale = 1")
    elif laci_take is not None:
        raise ValueError("A LACI duration target needs laci=true")

    # ---- prompt rows (the tokenizer worker)
    text = engine.normalize_text(req.text, language, bool(req.text_normalization))
    silence = None
    if edit is None:
        requested = req.silence_prefill_frames
        if requested is not None:
            requested = max(0, min(int(requested), MAX_SILENCE_PREFILL_FRAMES))
        frames = silence_frames(requested)
        if frames > 0:
            silence = silence_rows_from_codes(torch.tensor(c.silence_frame, dtype=torch.int32), frames, builder.config)

    def text_prompt_rows(t: str) -> torch.Tensor:
        prompt = builder.build(t)
        if edit is not None:
            from zonos3.tts.edit import edit_prompt_tokens

            if c.span_end_id is None:
                raise ValueError("Speech editing needs a checkpoint with a span-end audio token")
            return edit_prompt_tokens(prompt, edit.codes, edit.spans, edit.fills, builder.config,
                                      span_end_id=c.span_end_id)
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
        max_ref = int(c.speaker_reference_max_seconds * SAMPLE_RATE // engine.decoder.hop)
        reference_codes = reference_codes[:max_ref]
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

    # ---- CFG twin (the scheduler's _schedule_prefill)
    hidden = bool(sources)
    tag_cfg = float(req.tag_cfg_scale) > 1.0
    emotion_cfg = delta is not None and embedding is not None and float(req.emotion_cfg_scale) != 1.0
    speaker_cfg = float(req.speaker_cfg_scale) != 1.0
    if speaker_cfg and emotion_cfg:
        raise ValueError("Speaker and emotion CFG cannot be combined")
    if tag_cfg and (speaker_cfg or float(req.emotion_cfg_scale) != 1.0):
        raise ValueError("Tag CFG cannot be combined with speaker or emotion CFG")
    cond = Branch(t.to(torch.int32), embedding, speaker_row if embedding is not None else -1,
                  None if hidden else delta, condition_values, condition_token_start, steered=True)
    twin = None
    cfg_kind, cfg_scale = None, 1.0
    if speaker_cfg:
        msg = TwinMsg(t, reference_prefix_length, embedding, max(speaker_row, 0))
        spec = speaker_cfg_twin(msg, t, n_codebooks=c.n_codebooks, audio_pad_id=c.audio_pad_id, text_vocab=c.text_vocab,
                                speaker_token_position=max(speaker_row, 0), condition_values=condition_values,
                                condition_token_start=condition_token_start, condition_unknown=c.condition_n_buckets)
        twin = Branch(spec.input_ids.to(torch.int32), None, -1, None, spec.condition_values,
                      spec.condition_token_start, steered=True)
        cfg_kind, cfg_scale = "speaker", float(req.speaker_cfg_scale)
    elif tag_cfg:
        twin = replace(cond, input_ids=plain_ids.to(torch.int32))
        cfg_kind, cfg_scale = "tag", float(req.tag_cfg_scale)
    elif emotion_cfg:
        twin = replace(cond, proj_delta=None, steered=False)
        cfg_kind, cfg_scale = "emotion", float(req.emotion_cfg_scale)
    prompt_len = max(len(cond.input_ids), len(twin.input_ids)) if twin is not None else len(cond.input_ids)
    if prompt_len >= c.max_seqlen:
        raise ValueError(f"TTS input len {prompt_len} exceeds {c.max_seqlen}")
    max_tokens = min(max_tokens, c.max_seqlen - prompt_len)
    seed = int(req.seed) if req.seed is not None else secrets.randbelow(2**31)
    return Plan(
        request=req, text=text, language=language, cond=cond, twin=twin, cfg_kind=cfg_kind, cfg_scale=cfg_scale,
        reference_prefix_length=reference_prefix_length,
        prefill_rows=None if silence is None else silence.to(torch.int32),
        context_codes=context_codes, sources=sources, hidden_delta=delta if hidden else None, schedule=schedule,
        follow_tokens=follow_tokens, follow_gate=follow_gate,
        accent=None if accent is None else (accent, accent_strength), max_tokens=max_tokens, seed=seed,
        laci=language if req.laci else None, speaker_id=speaker_id, laci_take=laci_take, edit=edit,
        follow_language=language if follow_tokens is not None else None,
    )
