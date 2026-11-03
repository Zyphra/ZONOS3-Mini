"""The MLX server's models, companions and request planning.

``Engine`` owns the loaded networks (LM, codec decoder, speaker encoder, the lazily loaded reference encoder) and the
companion request logic (steering bundle, accent adapters, LACI heads, the text normalizer). ``plan_request`` turns a
v2 GenerateRequest plus its pre-resolved speaker inputs into the ``zonos3.reference.synth.PromptPlan`` the scheduler
runs: a port of ``zonos3.reference.synth.plan_prompt`` with the sgl server's request rules (api_server
``_tts_generate_msg``, ``_apply_request_controls``, ``_span_follow``) and the tokenizer worker's prompt layout.
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
from zonos3 import languages
from zonos3.reference.synth import (
    Branch,
    PromptPlan,
    TwinMsg,
    parse_request,
    resolve_condition_buckets,
    silence_frames,
    validate_phoneme_groups,
    validate_tag_cfg,
)
from zonos3.speaker_cfg import speaker_cfg_twin
from zonos3.tts.laci import LaciHeads
from zonos3.tts.markup import strip_nonverbal_markup
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig, silence_rows_from_codes
from zonos3.tts.span_follow import FollowConfig
from zonos3.tts.steering import SteeringBundle, resolve_controls

from zonos3_mlx import companions
from zonos3_mlx.models.codec_decoder import CodecDecoder
from zonos3_mlx.models.lm import LMConfig, Zonos3MLX

SAMPLE_RATE = 44100
MAX_SILENCE_PREFILL_FRAMES = 30
MAX_REFERENCE_FRAMES = int(os.environ.get("ZONOS3_MAX_REFERENCE_FRAMES", "900"))
COMPANIONS = {
    "dac": "dac.gguf", "spk_encoder": "spk-encoder.gguf", "dac_encoder": "dac-encoder.gguf",
    "ssl_teacher": "ssl-teacher.gguf", "steer": "steer.gguf", "accent_adapters": "accents.gguf",
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
        self.accents = companions.AccentBank.load(paths["accent_adapters"]) if paths.get("accent_adapters") else None
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

    def accent_names(self) -> list[str]:
        return [] if self.accents is None else list(self.accents.names)

    def steering_sources(self) -> tuple[int, ...]:
        return () if self.bundle is None else tuple(self.bundle.hidden_sources)

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


def resolve_accent(engine: Engine, accent, strength):
    """(accent adapter or None, strength) of a request (sgl ``_resolve_tts_request_accent``)."""
    if accent is None or accent == "" or accent == "none":
        return None, 1.0
    names = engine.accent_names()
    if accent not in names:
        loaded = ", ".join(names) or "none (load accent adapters)"
        raise ValueError(f"Unknown accent {accent!r}; loaded accents: {loaded}.")
    if not math.isfinite(float(strength)):
        raise ValueError(f"accent_strength must be finite, got {strength!r}.")
    return accent, float(strength)


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
    """The reference PromptPlan plus the server-side fields: its LACI take (dub) and its edit layout (speech editing,
    codes-only)."""

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
                     controls_cfg_scale=req.controls_cfg_scale)
    condition_values = resolve_condition_buckets(c, req.condition_buckets, req.condition_values)
    mode = speaker.mode if speaker is not None else "reference"
    embedding = None if speaker is None else speaker.embedding
    reference_codes = None if speaker is None else speaker.reference_codes
    language = languages.normalize(req.language)
    if req.max_tokens is not None and int(req.max_tokens) <= 0:
        raise ValueError("max_tokens must be positive.")
    max_tokens = engine.max_frames if req.max_tokens is None else min(int(req.max_tokens), engine.max_frames)
    accent, accent_strength = resolve_accent(engine, req.accent, req.accent_strength)
    sources = engine.steering_sources()
    delta = schedule = span_chars = None
    if req.controls or req.control_spans:
        bundle = engine.bundle
        if bundle is None:
            raise ValueError("This server has no steering bundle (place steer.gguf next to the backbone or pass "
                             "--steer); `controls` are unavailable.")
        resolved = resolve_controls(bundle, req.controls, list(req.control_spans or []), text=req.text,
                                    speaker_embedding=embedding, sources=sources, hidden_size=c.dim,
                                    frames_per_char=req.control_span_rate)
        embedding, schedule, span_chars = resolved.speaker_embedding, resolved.schedule, resolved.span_chars
        delta = resolved.hidden_delta
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
        if req.controls_cfg_scale != 1.0:
            raise ValueError("Speaker and controls CFG cannot be combined")
    builder = engine.prompt_builder()
    validate_phoneme_groups(builder.config.text_tokenizer_spec, req.text)
    if req.laci:
        if not engine.laci_heads:
            raise ValueError("LACI is not enabled on this server (start it with --laci-heads LANG=PATH)")
        if language not in engine.laci_heads:
            raise ValueError(f"No LACI alignment heads for language {language!r} (configured: "
                             f"{', '.join(sorted(engine.laci_heads))})")
        if float(req.speaker_cfg_scale) != 1.0 or float(req.controls_cfg_scale) != 1.0:
            raise ValueError("LACI needs speaker_cfg_scale = controls_cfg_scale = 1")
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
    tag_cfg = float(req.tag_cfg_scale) > 1.0
    controls_cfg = delta is not None and embedding is not None and float(req.controls_cfg_scale) != 1.0
    speaker_cfg = float(req.speaker_cfg_scale) != 1.0
    if speaker_cfg and controls_cfg:
        raise ValueError("Speaker and controls CFG cannot be combined")
    if tag_cfg and (speaker_cfg or float(req.controls_cfg_scale) != 1.0):
        raise ValueError("Tag CFG cannot be combined with speaker or controls CFG")
    cond = Branch(t.to(torch.int32), embedding, speaker_row if embedding is not None else -1,
                  condition_values=condition_values, condition_token_start=condition_token_start)
    twin = None
    cfg_kind, cfg_scale = None, 1.0
    if speaker_cfg:
        msg = TwinMsg(t, reference_prefix_length, embedding, max(speaker_row, 0))
        spec = speaker_cfg_twin(msg, t, n_codebooks=c.n_codebooks, audio_pad_id=c.audio_pad_id, text_vocab=c.text_vocab,
                                speaker_token_position=max(speaker_row, 0), condition_values=condition_values,
                                condition_token_start=condition_token_start, condition_unknown=c.condition_n_buckets)
        twin = Branch(spec.input_ids.to(torch.int32), condition_values=spec.condition_values,
                      condition_token_start=spec.condition_token_start)
        cfg_kind, cfg_scale = "speaker", float(req.speaker_cfg_scale)
    elif tag_cfg:
        twin = replace(cond, input_ids=plain_ids.to(torch.int32))
        cfg_kind, cfg_scale = "tag", float(req.tag_cfg_scale)
    elif controls_cfg:
        twin = replace(cond, steered=False)
        cfg_kind, cfg_scale = "controls", float(req.controls_cfg_scale)
    prompt_len = max(len(cond.input_ids), len(twin.input_ids)) if twin is not None else len(cond.input_ids)
    if prompt_len >= c.max_seqlen:
        raise ValueError(f"TTS input len {prompt_len} exceeds {c.max_seqlen}")
    max_tokens = min(max_tokens, c.max_seqlen - prompt_len)
    seed = int(req.seed) if req.seed is not None else secrets.randbelow(2**31)
    return Plan(
        request=req, text=text, language=language, cond=cond, twin=twin, cfg_kind=cfg_kind, cfg_scale=cfg_scale,
        reference_prefix_length=reference_prefix_length,
        prefill_rows=None if silence is None else silence.to(torch.int32),
        context_codes=context_codes, sources=sources, hidden_delta=delta, schedule=schedule,
        follow_tokens=follow_tokens, follow_gate=follow_gate,
        accent=None if accent is None else (accent, accent_strength), max_tokens=max_tokens, seed=seed,
        laci=language if req.laci else None, laci_take=laci_take, edit=edit,
        follow_language=language if follow_tokens is not None else None,
    )
