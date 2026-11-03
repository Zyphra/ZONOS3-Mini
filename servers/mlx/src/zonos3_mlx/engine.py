"""The MLX server's models, companions and request planning.

``Engine`` owns the loaded networks (LM, codec decoder, speaker encoder, the lazily loaded reference encoder) and the
companion request logic (steering bundle, accent adapters, LACI heads, the text normalizer), and is the
``zonos3.tts.plan.PromptContext`` of the shared request rules. ``plan_request`` turns a v2 GenerateRequest plus its
pre-resolved speaker inputs into the ``Plan`` the scheduler runs: ``zonos3.tts.plan.plan_prompt`` plus the server-side
fields (a dub take's LACI target, a speech-edit layout).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from zonos3.generation import MAX_FRAMES, REFERENCE_MAX_FRAMES
from zonos3.tts.laci import LaciHeads
from zonos3.tts.plan import PromptPlan, Speaker, plan_prompt
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig
from zonos3.tts.steering import SteeringBundle

from zonos3_mlx import companions
from zonos3_mlx.models.codec_decoder import CodecDecoder
from zonos3_mlx.models.lm import LMConfig, Zonos3MLX

COMPANIONS = {
    "zac": "zac.gguf", "spk_encoder": "spk-encoder.gguf", "zac_encoder": "zac-encoder.gguf",
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


class Engine:
    """The loaded networks and companions, and the ``zonos3.tts.plan.PromptContext`` of the request rules: the
    checkpoint's rows from the backbone GGUF, the loaded companions and the server limits."""

    steer_hint = "place steer.gguf next to the backbone or pass --steer"
    reference_max_frames = REFERENCE_MAX_FRAMES  # R7: the GGUF servers' reference prefix limit
    silence_available = True  # every backbone GGUF carries its codec's silence frame
    condition_flags: Mapping = MappingProxyType({})  # the GGUF converter refuses checkpoints with flag rows

    def __init__(self, lm: Zonos3MLX, decoder: CodecDecoder, *, paths: dict[str, str | None],
                 laci_heads: dict[str, str] | None = None, max_frames: int = MAX_FRAMES, normalizer=None,
                 reference_idle_unload: float = 60.0):
        self.lm = lm
        self.config: LMConfig = lm.config
        self.decoder = decoder
        self.paths = paths
        self.max_frames = int(max_frames)  # --max: the frame budget of a request (R5)
        self.normalizer: TextNormalizer | None = normalizer
        self.speaker_encoder = None
        if paths.get("spk_encoder"):
            from zonos3_mlx.models.speaker_encoder import SpeakerEncoder

            self.speaker_encoder = SpeakerEncoder.load(paths["spk_encoder"])
        self.steer_bundle: SteeringBundle | None = companions.load_steer(paths["steer"]) if paths.get("steer") else None
        self.steering_sources = () if self.steer_bundle is None else tuple(self.steer_bundle.hidden_sources)
        self.accents = companions.AccentBank.load(paths["accent_adapters"]) if paths.get("accent_adapters") else None
        self.accent_names = [] if self.accents is None else list(self.accents.names)
        self.laci_heads = {lang: LaciHeads.load(path) for lang, path in (laci_heads or {}).items()}
        self.laci_paths = dict(laci_heads or {})
        self.laci_languages = frozenset(self.laci_heads)
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
        c = self.config
        self.n_codebooks, self.codebook_size, self.hidden_size = c.n_codebooks, c.codebook_size, c.dim
        self.condition_names, self.condition_n_buckets, self.condition_edges = (
            c.condition_names, c.condition_n_buckets, c.condition_edges)
        # R2: the reference prefix first, then the speaker row (ggml checkpoint_speaker_modes)
        self.speaker_modes = [mode for mode, on in (("reference", c.speaker_reference_prefix_enabled),
                                                    ("embedding", c.speaker_embedding_dim > 0)) if on]
        self.text_tokenizer_spec = c.text_tokenizer_spec or None
        self.audio_pad_id, self.text_vocab, self.span_end_id, self.max_seqlen = (
            c.audio_pad_id, c.text_vocab, c.span_end_id, c.max_seqlen)
        self._builder = TTSPromptBuilder(TTSPromptConfig(
            n_codebooks=c.n_codebooks, codebook_sizes=c.codebook_sizes, audio_pad_id=c.audio_pad_id,
            text_vocab=c.text_vocab, text_tokenizer_spec=c.text_tokenizer_spec))

    @classmethod
    def load(cls, model: str | Path, *, zac: str | None = None, **kwargs) -> "Engine":
        given = {"zac": zac, **{k: kwargs.pop(k) for k in list(kwargs) if k in COMPANIONS}}
        paths = find_companions(model, given)
        if not paths.get("zac"):
            raise FileNotFoundError(f"--zac <zac.gguf> is required (also auto-detected when zac.gguf sits next to {model})")
        lm = Zonos3MLX.load(model)
        return cls(lm, CodecDecoder.load(paths["zac"]), paths=paths, **kwargs)

    # -- companions -------------------------------------------------------------------------------------------------

    def prompt_builder(self) -> TTSPromptBuilder:
        return self._builder

    def silence_frame(self) -> torch.Tensor:
        """The codec's silence frame [n_codebooks] (the backbone GGUF's ``silence_frame``)."""
        return torch.tensor(self.config.silence_frame, dtype=torch.int32)

    @property
    def has_reference_encoder(self) -> bool:
        return bool(self.paths.get("zac_encoder") and self.paths.get("ssl_teacher"))

    def reference_encoder(self):
        """The codec encoder + SSL teacher, loaded on first use (call on the scheduler worker thread)."""
        with self._reference_lock:
            if self._reference_encoder is None:
                from zonos3_mlx.models.codec_encoder import CodecEncoder
                from zonos3_mlx.models.ssl_teacher import SSLTeacher

                self._reference_encoder = CodecEncoder.load(self.paths["zac_encoder"],
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
    """The shared PromptPlan plus the server-side fields: its LACI take (dub), its edit layout (speech editing,
    codes-only) and the language whose LACI heads drive the span follower."""

    laci_take: LaciTake | None = None
    edit: EditLayout | None = None
    follow_language: str | None = None


def plan_request(engine: Engine, request: Mapping, speaker: Speaker | None, *, laci_take: LaciTake | None = None,
                 edit: EditLayout | None = None, hidden_delta=None) -> Plan:
    """The scheduler plan of ``request``: ``zonos3.tts.plan.plan_prompt`` over the engine with the speaker inputs the
    HTTP layer resolved (None: none, in the request's speaker mode). ``laci_take``: a dub take's LACI duration target;
    ``edit``: a speech-edit region (the edit layout, no silence prefill, stop on span-end; ``hidden_delta`` its
    steering delta). Raises ValueError for a request the server refuses (HTTP 400)."""
    plan = plan_prompt(engine, request, lambda req, mode: Speaker(mode) if speaker is None else speaker,
                       hidden_delta=hidden_delta, edit=edit, duration_target=laci_take is not None)
    return Plan(**vars(plan), laci_take=laci_take, edit=edit,
                follow_language=plan.language if plan.follow_tokens is not None else None)
