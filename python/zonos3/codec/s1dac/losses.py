"""DAC GAN/reconstruction objectives and Mimi-style averaged cosine distillation."""
import torch
from torch import nn
from torch.nn import functional as F
from audiotools import AudioSignal
from dac.nn.loss import MelSpectrogramLoss


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = MelSpectrogramLoss(
            n_mels=[5,10,20,40,80,160,320],
            window_lengths=[32,64,128,256,512,1024,2048],
            mel_fmin=[0]*7, mel_fmax=[None]*7,
            pow=1.0, clamp_eps=1e-5, mag_weight=0.0)

    def forward(self, fake, real):
        with torch.autocast(fake.device.type, enabled=False):
            return self.mel(AudioSignal(fake.float(),44100), AudioSignal(real.float(),44100))


def _families(fake, real):
    if not isinstance(fake, dict):
        fake, real = {"all": fake}, {"all": real}
    if fake.keys() != real.keys():
        raise ValueError("Real/fake discriminator families differ")
    for name in fake:
        if not fake[name] or len(fake[name]) != len(real[name]):
            raise ValueError(f"Missing/mismatched discriminator branches: {name}")
        yield name, fake[name], real[name]


def discriminator_loss(fake, real, return_families=False):
    values = {
        name: sum(x[-1].float().square().mean() + (1-y[-1].float()).square().mean()
                  for x, y in zip(fs, rs))
        for name, fs, rs in _families(fake, real)
    }
    total = sum(values.values())
    return (total, values) if return_families else total


def generator_losses(fake, real, return_families=False):
    adversarial, feature = {}, {}
    for name, fs, rs in _families(fake, real):
        adversarial[name] = sum((1-x[-1].float()).square().mean() for x in fs)
        feature[name] = sum(
            F.l1_loss(x.float(), y.detach().float())
            for fake_branch, real_branch in zip(fs, rs)
            for x, y in zip(fake_branch[:-1], real_branch[:-1]))
    result = (sum(adversarial.values()), sum(feature.values()))
    if return_families:
        parts = {**{"adversarial/"+k: v for k, v in adversarial.items()},
                 **{"feature_matching/"+k: v for k, v in feature.items()}}
        return (*result, parts)
    return result


def average_ssl_frames(target, student_frames, window_ratio=2.0):
    """Noncausal box averages, including all overlapping SSL frame cells.

    Mimi uses kernel=8, stride=4 for 50 -> 12.5 Hz. Here a student interval
    occupies target_frames/student_frames SSL frame cells, and the window is
    window_ratio intervals wide, centered on that interval. With ratio 4 and
    window_ratio 2 this is avg_pool1d(kernel=8,stride=4,padding=2,
    count_include_pad=False).

    Fractional boundary cells contribute in proportion to their overlap.
    Integral prefix sums implement exact piecewise-constant box integration
    without interpolating feature vectors or rounding away any teacher frames.
    Crop-edge windows are clipped and renormalized over available frames.
    """
    if target.ndim != 3 or target.shape[1] < 1 or student_frames < 1:
        raise ValueError("Expected nonempty SSL features [batch,frames,channels]")
    if window_ratio <= 0:
        raise ValueError("window_ratio must be positive")
    with torch.autocast(target.device.type, enabled=False):
        values = target.detach().float().transpose(1, 2)
        teacher_frames = values.shape[-1]
        ratio = teacher_frames / student_frames
        centers = (torch.arange(student_frames, device=target.device, dtype=torch.float32) + 0.5) * ratio
        half_width = 0.5 * window_ratio * ratio
        left = (centers - half_width).clamp(0, teacher_frames)
        right = (centers + half_width).clamp(0, teacher_frames)
        prefix = F.pad(values.cumsum(dim=-1), (1, 0))
        def integral(position):
            index = position.floor().long()
            fraction = position - index
            return (prefix.index_select(-1, index)
                    + values.index_select(-1, index.clamp(max=teacher_frames-1)) * fraction)
        return (integral(right) - integral(left)) / (right-left)


def semantic_loss(prediction, target, window_ratio=2.0, eps=1e-8):
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("Prediction/SSL features must be rank three")
    if prediction.shape[:2] != (target.shape[0], target.shape[2]):
        raise ValueError("Prediction and SSL batch/channel dimensions differ")
    with torch.autocast(prediction.device.type, enabled=False):
        pooled = average_ssl_frames(target, prediction.shape[-1], window_ratio)
        cosine = F.cosine_similarity(prediction.float(), pooled, dim=1, eps=eps)
        return (1 - cosine.clamp(-1, 1)).mean()


class SemanticTeacher(nn.Module):
    def __init__(self, path, layer=16, dtype=torch.bfloat16):
        super().__init__()
        if layer != 16:
            raise ValueError("The recipe targets transformer layer 16")
        from transformers import Wav2Vec2BertModel
        self.model = Wav2Vec2BertModel.from_pretrained(
            path, dtype=dtype, local_files_only=True)
        self.model.encoder.layers = nn.ModuleList(list(self.model.encoder.layers[:layer]))
        self.model.config.num_hidden_layers = layer
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, features, attention_mask):
        self.model.eval()
        return self.model(
            input_features=features.to(next(self.model.parameters()).dtype),
            attention_mask=attention_mask, return_dict=True).last_hidden_state
