import os
from contextlib import nullcontext
from functools import cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    Wav2Vec2BertConfig,
    Wav2Vec2BertModel,
)

try:
    from zonos.utils import DEFAULT_DEVICE
except Exception:
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Qwen3SpeakerEmbedding(nn.Module):
    """Qwen3 voice embedding extractor used by 2048D speaker-conditioned checkpoints."""

    MODEL_NAME = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
    TARGET_SAMPLE_RATE = 24_000
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_MELS = 128
    F_MIN = 0.0
    F_MAX = 12_000.0

    def __init__(self, device: str = DEFAULT_DEVICE):
        super().__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.TARGET_SAMPLE_RATE,
            n_fft=self.N_FFT,
            win_length=self.WIN_LENGTH,
            hop_length=self.HOP_LENGTH,
            f_min=self.F_MIN,
            f_max=self.F_MAX,
            n_mels=self.N_MELS,
            power=1.0,
            center=False,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)

        self.requires_grad_(False).eval()

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @cache
    def _get_resampler(self, orig_sample_rate: int):
        return torchaudio.transforms.Resample(orig_sample_rate, self.TARGET_SAMPLE_RATE).to(self.device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert wav.ndim < 3
        if wav.ndim == 2:
            wav = wav.mean(0, keepdim=True)
        wav = wav.to(self.device, torch.float32)
        if sample_rate != self.TARGET_SAMPLE_RATE:
            wav = self._get_resampler(sample_rate)(wav)
        return wav

    def _make_mel(self, wav: torch.Tensor) -> torch.Tensor:
        # Mirror the standalone demo preprocessing: reflect-pad, magnitude mel, log clamp.
        pad = (self.N_FFT - self.HOP_LENGTH) // 2
        wav = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        mel = self.mel_transform(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.transpose(1, 2)

    def forward(self, wav: torch.Tensor, sample_rate: int):
        wav = self.prepare_input(wav, sample_rate)
        mel = self._make_mel(wav)
        return self.model(input_values=mel).last_hidden_state.to(torch.float32)


class _AttentiveStatisticsPooling(nn.Module):
    """Pooling layer used by the published W2V-BERT speaker checkpoint."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x.transpose(1, 2)).transpose(1, 2)
        mean = torch.sum(x * weights, dim=1)
        std = torch.sqrt(
            (torch.sum(x.square() * weights, dim=1) - mean.square()).clamp(min=1e-5)
        )
        return torch.cat([mean, std], dim=1)


class W2VBertSVSpeakerEmbedding(nn.Module):
    """Published 256D encoder behind the ``w2vbert_sv`` training records.

    The public checkpoint contains the full W2V-BERT 2.0 encoder together with
    the per-layer adapters, attentive statistics pooling, and 256D bottleneck.
    Its forward path intentionally mirrors the data-generation implementation
    instead of calling the generic Transformers forward method.
    """

    MODEL_NAME = "facebook/w2v-bert-2.0"
    CHECKPOINT_REPO = "zl389/w2v-bert-2.0_SV"
    CHECKPOINT_FILE = "model_base_0.23.pth"
    TARGET_SAMPLE_RATE = 16_000
    MAX_SECONDS = 60
    N_HIDDEN_STATES = 25
    ENCODER_DIM = 1024
    ADAPTER_DIM = 128
    EMBEDDING_DIM = 256

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        checkpoint_path: str | Path | None = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_NAME)

        # Instantiate the architecture only. All weights, including the base
        # encoder, are restored from model_base_0.23.pth below.
        config = Wav2Vec2BertConfig.from_pretrained(self.MODEL_NAME)
        config.apply_spec_augment = False
        self.front = nn.Module()
        self.front.encoder = Wav2Vec2BertModel(config)
        # The training wrapper removes this unused SpecAugment parameter before
        # saving, so doing the same gives us a strict state-dict load.
        if hasattr(self.front.encoder, "masked_spec_embed"):
            delattr(self.front.encoder, "masked_spec_embed")

        self.adapter_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.ENCODER_DIM, self.ADAPTER_DIM),
                    nn.LayerNorm(self.ADAPTER_DIM),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.ADAPTER_DIM, self.ADAPTER_DIM),
                )
                for _ in range(self.N_HIDDEN_STATES)
            ]
        )
        pooled_dim = self.N_HIDDEN_STATES * self.ADAPTER_DIM
        self.pooling = _AttentiveStatisticsPooling(pooled_dim, self.ADAPTER_DIM)
        self.bottleneck = nn.Linear(pooled_dim * 2, self.EMBEDDING_DIM)

        if checkpoint_path is None:
            checkpoint_path = hf_hub_download(
                self.CHECKPOINT_REPO,
                self.CHECKPOINT_FILE,
            )
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        try:
            state = checkpoint["modules"]["spk_model"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "W2V-BERT speaker checkpoint must contain modules.spk_model."
            ) from exc
        self.load_state_dict(state, strict=True)
        del checkpoint, state

        self.to(self.device)
        self.requires_grad_(False).eval()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @cache
    def _get_resampler(self, orig_sample_rate: int):
        return torchaudio.transforms.Resample(
            orig_sample_rate,
            self.TARGET_SAMPLE_RATE,
        ).to(self.device)

    def prepare_input(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if wav.ndim not in (1, 2):
            raise ValueError(f"Expected mono/stereo waveform, got shape {tuple(wav.shape)}.")
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(self.device, torch.float32)
        if sample_rate != self.TARGET_SAMPLE_RATE:
            wav = self._get_resampler(sample_rate)(wav)
        return wav[..., : self.MAX_SECONDS * self.TARGET_SAMPLE_RATE]

    def _extract_features(self, wav: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(
            wav.squeeze(0).detach().cpu().numpy(),
            sampling_rate=self.TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        return features.input_features.to(self.device, dtype=self.dtype)

    def forward(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav = self.prepare_input(wav, sample_rate)
        input_features = self._extract_features(wav)

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            x = self.front.encoder.feature_projection(input_features)[0]
            hidden_states = [x]
            for layer in self.front.encoder.encoder.layers:
                x = layer(x)[0]
                hidden_states.append(x)

            adapted = [
                adapter(hidden)
                for adapter, hidden in zip(self.adapter_layers, hidden_states, strict=True)
            ]
            x = torch.cat(adapted, dim=-1)
            x = self.pooling(x)
            x = self.bottleneck(x)
        return x.to(torch.float32)


def speaker_embedder_device() -> str:
    """The servers' speaker-encoder device: ``ZONOS3_SPEAKER_EMBEDDER_DEVICE``, else the CPU (it does not occupy GPU
    memory next to the TTS model)."""
    return os.environ.get("ZONOS3_SPEAKER_EMBEDDER_DEVICE", "cpu")


def resolve_speaker_embedding_backend(codec: str | None, expected_dim: int) -> str:
    """Resolve a checkpoint codec to an installed reference-audio encoder."""
    normalized = str(codec or "").strip().lower().replace("-", "_")
    aliases = {
        "qwen3": "qwen3",
        "qwen3_voice_embedding": "qwen3",
        "w2vbert_sv": "w2vbert_sv",
        "w2v_bert_sv": "w2vbert_sv",
    }
    if normalized:
        if normalized in aliases:
            return aliases[normalized]
        # codecs beyond the built-in encoders (e.g. ownmect_s2) come from the speaker encoder registry, keyed by the
        # exact training codec name; the backend key is that name
        from zonos3.models.mect_speaker import registry_entry, registry_path

        entry = registry_entry(codec)
        if entry is not None:
            if int(entry.get("embed_dim", expected_dim)) != int(expected_dim):
                raise ValueError(
                    f"Speaker embedding codec {codec!r}: the checkpoint expects {expected_dim} dimensions but the "
                    f"registry entry produces {entry.get('embed_dim')} ({registry_path()})."
                )
            return entry["codec"]
        raise ValueError(
            f"Unsupported speaker embedding codec: {codec!r} (not a built-in encoder and not in the speaker encoder "
            f"registry {registry_path()})."
        )

    # Backward compatibility for older converted checkpoints that predate the
    # explicit codec field. New conversions always preserve the training codec.
    if expected_dim == 2048:
        return "qwen3"
    if expected_dim == 256:
        return "w2vbert_sv"
    raise ValueError(
        "Checkpoint does not identify its speaker embedding codec and no "
        f"legacy backend is known for {expected_dim} dimensions."
    )


def create_speaker_embedder(
    codec: str | None,
    expected_dim: int,
    device: str = DEFAULT_DEVICE,
) -> nn.Module:
    backend = resolve_speaker_embedding_backend(codec, expected_dim)
    if backend == "qwen3":
        return Qwen3SpeakerEmbedding(device=device)
    if backend == "w2vbert_sv":
        return W2VBertSVSpeakerEmbedding(device=device)
    from zonos3.models.mect_speaker import create_registry_speaker_embedder, registry_entry

    entry = registry_entry(backend)
    if entry is not None:
        return create_registry_speaker_embedder(entry, device=device)
    raise AssertionError(f"Unhandled speaker embedding backend: {backend}")
# MECT_SPEAKER_ENCODER_20261016
