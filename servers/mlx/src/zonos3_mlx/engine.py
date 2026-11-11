"""The MLX server's models, companions and request planning.

``Engine`` owns the loaded networks (LM, codec decoder, speaker encoder, the lazily loaded reference encoder) and the
companion request logic (steering bundle, accent adapters, LACI heads, the text normalizer), and is the
``zonos3.tts.plan.PromptContext`` of the shared request rules. ``plan_request`` turns a v2 GenerateRequest plus its
pre-resolved speaker inputs into the ``Plan`` the scheduler runs: ``zonos3.tts.plan.plan_prompt`` plus the server-side
fields (a LACI take's duration target, a speech-edit layout).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from zonos3.generation import MAX_FRAMES, REFERENCE_MAX_FRAMES
from zonos3.server.speech import resolve_decoder
from zonos3.tokenizer.textnorm import TTSTextNormalizer, normalize_request_text
from zonos3.tts.laci import LaciHeads
from zonos3.tts.plan import PromptPlan, Speaker, plan_prompt
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig
from zonos3.tts.steering import SteeringBundle

from zonos3_mlx import companions
from zonos3_mlx.models.codec_decoder import CodecDecoder
from zonos3_mlx.models.lm import LMConfig, Zonos3MLX

# Decoders found next to the backbone without --zac, in order (the first found is the default): the keep2_plain head,
# the full-size full_mr head.
DECODERS = ("zac-keep2.gguf", "zac.gguf")
COMPANIONS = {
    "spk_encoder": "spk-encoder.gguf",
    "zac_encoder": "zac-encoder.gguf",
    "ssl_teacher": "ssl-teacher.gguf",
    "steer": "steer.gguf",
    "accent_adapters": "accents.gguf",
}


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


def find_decoders(model_path: str | Path, given: Sequence[str] | None) -> list[str]:
    """Decoder GGUF paths: the --zac flags (the first is the default), else the ``DECODERS`` next to the backbone."""
    if given:
        for path in given:
            if not Path(path).is_file():
                raise FileNotFoundError(f"decoder file {path} does not exist")
        return [str(path) for path in given]
    root = Path(model_path).resolve().parent
    return [str(root / name) for name in DECODERS if (root / name).is_file()]


class Engine:
    """The loaded networks and companions, and the ``zonos3.tts.plan.PromptContext`` of the request rules: the
    checkpoint's rows from the backbone GGUF, the loaded companions and the server limits."""

    steer_hint = "place steer.gguf next to the backbone or pass --steer"
    reference_max_frames = REFERENCE_MAX_FRAMES  # R7: the GGUF servers' reference prefix limit
    silence_available = True  # every backbone GGUF carries its codec's silence frame
    condition_flags: Mapping = MappingProxyType({})  # the GGUF converter refuses checkpoints with flag rows

    def __init__(
        self,
        lm: Zonos3MLX,
        decoders: Sequence[CodecDecoder],
        *,
        paths: dict[str, str | None],
        laci_heads: dict[str, str] | None = None,
        max_frames: int = MAX_FRAMES,
        normalizer=None,
        reference_idle_unload: float = 60.0,
    ):
        self.lm = lm
        self.config: LMConfig = lm.config
        # every decoder a request may pick, by id (mossdual_istft.head.variant), the default (``decoder``) first
        self.decoders: dict[str, CodecDecoder] = {}
        for dec in decoders:
            if dec.id in self.decoders:
                raise ValueError(
                    f"{dec.path}: decoder {dec.id} (mossdual_istft.head.variant) is already loaded from "
                    f"{self.decoders[dec.id].path}"
                )
            self.decoders[dec.id] = dec
        self.decoder = decoders[0]
        self.paths = paths
        self.max_frames = int(max_frames)  # --max: the frame budget of a request (R5)
        self.normalizer: TTSTextNormalizer | None = normalizer
        self.speaker_encoder = None
        if paths.get("spk_encoder"):
            from zonos3_mlx.models.speaker_encoder import SpeakerEncoder

            self.speaker_encoder = SpeakerEncoder.load(paths["spk_encoder"])
        self.steer_bundle: SteeringBundle | None = companions.load_steer(paths["steer"]) if paths.get("steer") else None
        self.steering_sources = () if self.steer_bundle is None else tuple(self.steer_bundle.hidden_sources)
        self.accents = (
            companions.AccentBank.load(
                paths["accent_adapters"], head_rows=lm.config.n_codebooks * lm.config.audio_vocab
            )
            if paths.get("accent_adapters")
            else None
        )
        self.accent_names = [] if self.accents is None else list(self.accents.accent_names)
        if self.accents is not None and self.accents.edit is not None and lm.config.span_end_id is None:
            raise ValueError(
                f"{paths['accent_adapters']}: {self.accents.edit} is a speech-edit adapter; the backbone "
                "lacks its vocabulary (convert it with tools/convert_gguf.py lm --edit-adapter)"
            )
        self.laci_heads = {lang: LaciHeads.load(path) for lang, path in (laci_heads or {}).items()}
        self.laci_paths = dict(laci_heads or {})
        self.laci_languages = frozenset(self.laci_heads)
        for lang, heads in self.laci_heads.items():
            for layer, head in heads.heads:
                if (
                    not (0 <= layer < self.config.n_layers)
                    or self.config.uses_gdn(layer)
                    or not (0 <= head < self.config.n_heads)
                ):
                    raise ValueError(
                        f"--laci-heads {lang}={self.laci_paths[lang]}: head ({layer}, {head}) is not an "
                        "attention head of this model"
                    )
        self._reference_lock = threading.Lock()
        self._reference_encoder = None
        self._reference_last_use = 0.0
        self.reference_idle_unload = float(reference_idle_unload)
        c = self.config
        self.n_codebooks, self.codebook_size, self.hidden_size = c.n_codebooks, c.codebook_size, c.dim
        self.condition_names, self.condition_bucket_counts, self.condition_edges = (
            c.condition_names,
            c.condition_bucket_counts,
            c.condition_edges,
        )
        # R2: the reference prefix first, then the speaker row (ggml checkpoint_speaker_modes)
        self.speaker_modes = [
            mode
            for mode, on in (
                ("reference", c.speaker_reference_prefix_enabled),
                ("embedding", c.speaker_embedding_dim > 0),
            )
            if on
        ]
        self.text_tokenizer_spec = c.text_tokenizer_spec or None
        self.audio_pad_id, self.text_vocab, self.span_end_id, self.max_seqlen = (
            c.audio_pad_id,
            c.text_vocab,
            c.span_end_id,
            c.max_seqlen,
        )
        self._builder = TTSPromptBuilder(
            TTSPromptConfig(
                n_codebooks=c.n_codebooks,
                codebook_sizes=c.codebook_sizes,
                audio_pad_id=c.audio_pad_id,
                text_vocab=c.text_vocab,
                text_tokenizer_spec=c.text_tokenizer_spec,
            )
        )

    @classmethod
    def load(cls, model: str | Path, *, zac: Sequence[str] | None = None, **kwargs) -> "Engine":
        """``zac``: the decoder GGUFs (the first is the default); None = those of ``DECODERS`` next to ``model``."""
        paths = find_companions(model, {k: kwargs.pop(k) for k in list(kwargs) if k in COMPANIONS})
        decoders = find_decoders(model, zac)
        if not decoders:
            raise FileNotFoundError(
                f"--zac <zac.gguf> is required (also auto-detected when {' / '.join(DECODERS)} sits next to {model})"
            )
        lm = Zonos3MLX.load(model)
        return cls(lm, [CodecDecoder.load(path) for path in decoders], paths=paths, **kwargs)

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

                self._reference_encoder = CodecEncoder.load(
                    self.paths["zac_encoder"], SSLTeacher.load(self.paths["ssl_teacher"])
                )
            self._reference_last_use = time.monotonic()
            return self._reference_encoder

    def unload_idle_reference_encoder(self) -> bool:
        """Free the reference encoder after --reference-idle-unload idle seconds (0 keeps it); True when freed."""
        with self._reference_lock:
            if (
                self._reference_encoder is None
                or self.reference_idle_unload <= 0
                or time.monotonic() - self._reference_last_use < self.reference_idle_unload
            ):
                return False
            self._reference_encoder = None
        import mlx.core as mx

        mx.clear_cache()
        return True

    def decoder_of(self, decoder: str | None) -> CodecDecoder:
        """A request's decoder (None = the default); ValueError (HTTP 400) for an id this server does not serve."""
        return self.decoders[resolve_decoder(decoder, self.decoders)]

    def output_gain(self, headroom_db: float | None, decoder: str | None) -> float:
        """The output gain of ``headroom_db``, None = the headroom of the decoder ``decoder`` (None = the default)."""
        return self.decoder_of(decoder).output_gain(headroom_db)

    # -- text ---------------------------------------------------------------------------------------------------------

    def normalize_text(self, text: str, language: str, enabled: bool) -> str:
        return normalize_request_text(text, language, self.normalizer if enabled else None)


# ----------------------------------------------------------------------------------------------------- planning


@dataclass
class LaciTake:
    """A LACI take's duration target and rate scale (zonos3.server.plugins.LaciTakeRequest)."""

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
    """The shared PromptPlan plus the server-side fields: its LACI take, its edit layout (speech editing,
    codes-only) and the language whose LACI heads drive the span follower."""

    laci_take: LaciTake | None = None
    edit: EditLayout | None = None
    follow_language: str | None = None


def plan_request(
    engine: Engine,
    request: Mapping,
    speaker: Speaker | None,
    *,
    laci_take: LaciTake | None = None,
    edit: EditLayout | None = None,
    hidden_delta=None,
) -> Plan:
    """The scheduler plan of ``request``: ``zonos3.tts.plan.plan_prompt`` over the engine with the speaker inputs the
    HTTP layer resolved (None: none, in the request's speaker mode). ``laci_take``: a LACI take's duration target
    and rate scale (zonos3.server.plugins.LaciTakeRequest);
    ``edit``: a speech-edit region (the edit layout, no silence prefill, stop on span-end; ``hidden_delta`` its
    steering delta; it runs on the speech-edit adapter at strength 1 when accents.gguf has one). Raises ValueError for
    a request the server refuses (HTTP 400)."""
    plan = plan_prompt(
        engine,
        request,
        lambda req, mode: Speaker(mode) if speaker is None else speaker,
        hidden_delta=hidden_delta,
        edit=edit,
        duration_target=laci_take is not None,
    )
    if edit is not None and engine.accents is not None and engine.accents.edit is not None:
        plan.accent = (engine.accents.edit, 1.0)
    return Plan(
        **vars(plan),
        laci_take=laci_take,
        edit=edit,
        follow_language=plan.language if plan.follow_tokens is not None else None,
    )
