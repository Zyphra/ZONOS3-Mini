#!/usr/bin/env python3
"""Build every zonos3.cpp GGUF from the PyTorch modules of the `zonos3` model package.

    T="uv run --package zonos3-tools python tools/convert_gguf.py"
    $T lm RELEASE/lm/rate30_20k_r2grpo [--outtype f32|f16|q8_0] [--silence-frame J] -o zonos3-f16.gguf
    $T lm RELEASE/lm/rate30_20k_r2grpo --outtype q4_0-exp-q8_0-mix -o zonos3-ptq.gguf   (PTQ)
    $T lm lm/mect_60k_q4q8_qat -o zonos3-qat.gguf               (QAT directory: bake.json present)
    $T codec-decoder RELEASE/autoencoder/generator.pt --head RELEASE/autoencoder/heads/full_mr -o zac.gguf
    $T codec-encoder RELEASE/autoencoder/generator.pt -o zac-encoder.gguf
    $T ssl-teacher RELEASE/autoencoder/teacher -o ssl-teacher.gguf
    $T speaker-encoder RELEASE/speaker_encoders/ownmect_s2/ckpt_0037.pth -o spk-encoder.gguf
    $T steer assets/steering/studio_bundle_mect60k_x10.json -o steer.gguf
    $T accents ACCENT_ADAPTER_DIR -o accents.gguf

The server finds `steer.gguf` and `accents.gguf` beside the backbone GGUF.
Every subcommand loads its weights into the module that runs them (zonos3.reference.lm.Zonos3LM, ISTFTCodec,
MossCodec, transformers' Wav2Vec2BertModel, MECTFbankSpeakerEmbedding) with a strict load, so an unknown or missing
checkpoint tensor is an error; `--help` on a subcommand describes its GGUF layout.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import gguf.quants as gquants
import numpy as np
import torch
import yaml
from gguf import GGMLQuantizationType, GGUFWriter
from zonos3.utils.hashing import sha256_bytes


def write_gguf(writer: GGUFWriter, progress: bool = False) -> None:
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=progress)
    writer.close()


# ===================================================================================================== lm

LM_DOC = """Export a ZONOS3 Mini LM directory (release lm/mect_60k or lm/rate30_20k_r2grpo, a derived edit / NV /
phoneme checkpoint, or the Q4_0-expert/Q8_0-mixer QAT derivative) to zonos3 GGUF.

The directory holds params.json and model.pth (zonos3.reference.lm.Zonos3LM.load: strict) plus the tokenizer named by
params.text_tokenizer_spec.path; conversion.json and config.yaml are validated when present. QAT directories
(bake.json present, step 3462) hold BF16 weights already fake-quantized by the QAT forward: routed experts on the Q4_0
grid, GDN in_proj and attention wq/wkv/wo/gater on the Q8_0 grid. They export only with --outtype q4_0-exp-q8_0-mix
(the default for them), which stores exactly those tensors as Q4_0/Q8_0 and the rest F16/F32; other directories
export with f32|f16|q8_0 (default f16), or with q4_0-exp-q8_0-mix, which quantizes the same tensors post-training
(round to nearest from the checkpoint weights; general.name gains " Q4_0-expert/Q8_0-mixer PTQ"). The condition
tables' bucket counts (params condition_bucket_counts, else condition_n_buckets for every table) are written as
zonos3.condition.bucket_counts. The silence frame defaults to
<lm_dir>/../../registries/silence_frame_<audio_codec>.json; pass --silence-frame when the directory sits elsewhere.
The silence frame (codec-silence prefill), condition edges/units, the tokenizer and the tokenizer spec are embedded
verbatim. Checkpoints fine-tuned on nonverbal markup (zonos3.tts.markup.trained_on_tags: a config.yaml data source with
`nonverbal: true`) get `zonos3.tag_cfg_trained = true`; the key is absent otherwise. --edit-adapter <.pt> grows the
model to a zonos-accent-lora/3 speech-edit adapter's vocabulary first (zonos3.reference.lm.Zonos3LM.grow: span_end_id,
the edit marker rows; plain requests never read them): serve it with that adapter in accents.gguf. --phoneme-rows <.pt>
then appends zonos-text-extra/1 phoneme tokens after them (zonos3.models.text_extra; the spec gains their phonemes):
text without pronunciation groups never reads them. Tensors are written in LM_NAMES order."""

ARCH = "zonos3"
MODEL_NAME = "bingua2-gdn-fullattnres-sbpe8k-375k-zv3ft-srcmix-cond3-mect-60k-20261015"
GENERAL_NAME = "ZONOS3 Mini srcmix-cond3-mect-60k step 60000"
RELEASE_STEP = 60000
MIX_OUTTYPE = "q4_0-exp-q8_0-mix"
QAT_STEP = 3462
QAT_GENERAL_NAME = "ZONOS3 Mini srcmix-cond3-mect-60k Q4_0-expert/Q8_0-mixer QAT step 3462"
QAT_COUNTS = {"q4_0": 20, "q8_0": 26}
# bake.json key suffix (lingua names) -> (release state-dict suffix, GGUF tensor names)
QAT_TARGETS = {
    "feed_forward.experts.w13": ("feed_forward.experts.gate_up_proj", ("ffn_gate_exps.weight", "ffn_up_exps.weight")),
    "feed_forward.experts.w2": ("feed_forward.experts.down_proj", ("ffn_down_exps.weight",)),
    "attention.in_proj.weight": ("attention.in_proj.weight", ("gdn_in.weight",)),
    "attention.wo.weight": ("attention.wo.weight", ("attn_output.weight",)),
    "attention.wq.weight": ("attention.wq.weight", ("attn_q.weight",)),
    "attention.wkv.weight": ("attention.wkv.weight", ("attn_k.weight", "attn_v.weight")),
    "attention.gater.weight": ("attention.gater.weight", ("attn_gate.weight",)),
}
CODEC = "opus_dual_3n_600k_sched600k_r4096_step600000"
CONDITION_NAMES = ["loudness", "fmax", "speaking_rate"]
TOKENIZER_SPEC = {"kind": "superbpe", "offset": 4, "bos": 2, "eos": 3}
SPEAKER_REFERENCE_MAX_SECONDS = 60.0


def _unbind0(t):
    return t.unbind(0)


def _halves1(t):
    return t.chunk(2, 1)


def _squeeze1(t):
    return t.squeeze(1)


def _abs_flat(t):
    return t.reshape(-1).abs()


# The GGUF name contract: (Zonos3LM parameter regex, GGUF names, transform). A transform returning a tuple splits the
# parameter into the names in order. Global entries precede the per-layer ones ("layers.{L}."), and the GGUF lists
# tensors in this order: global entries, then layer by layer the per-layer entries (an entry's numbered parameters,
# e.g. the embedders, in index order). Embedder n_codebooks is the text table.
LM_NAMES: tuple[tuple[str, tuple[str, ...], object], ...] = (
    (r"multi_embedder\.embedders\.(?P<i>\d+)\.weight", ("audio_embd.{i}.weight",), None),
    (r"out_norm\.weight", ("output_norm.weight",), None),
    (r"multi_output\.weight", ("output.weight",), None),
    (r"block_attnres_queries", ("attnres_q.weight",), None),
    (r"speaker_projection\.weight", ("speaker_proj.weight",), None),
    (r"speaker_projection\.bias", ("speaker_proj.bias",), None),
    (r"condition_embeddings\.(?P<i>\d+)\.weight", ("cond_embd.{i}.weight",), None),
    (r"layers\.(?P<L>\d+)\.attention_norm\.weight", ("blk.{L}.attn_norm.weight",), None),
    (r"layers\.(?P<L>\d+)\.ffn_norm\.weight", ("blk.{L}.ffn_norm.weight",), None),
    (r"layers\.(?P<L>\d+)\.attention\.in_proj\.weight", ("blk.{L}.gdn_in.weight",), None),
    (r"layers\.(?P<L>\d+)\.attention\.out_weight", ("blk.{L}.gdn_norm.weight",), None),
    (r"layers\.(?P<L>\d+)\.attention\.A_log", ("blk.{L}.gdn_A_log",), None),
    (r"layers\.(?P<L>\d+)\.attention\.dt_bias", ("blk.{L}.gdn_dt_bias",), None),
    (r"layers\.(?P<L>\d+)\.attention\.conv\.weight", ("blk.{L}.gdn_conv.weight",), _squeeze1),
    (r"layers\.(?P<L>\d+)\.attention\.wq\.weight", ("blk.{L}.attn_q.weight",), None),
    (r"layers\.(?P<L>\d+)\.attention\.wkv\.weight", ("blk.{L}.attn_k.weight", "blk.{L}.attn_v.weight"), _unbind0),
    (r"layers\.(?P<L>\d+)\.attention\.temp", ("blk.{L}.attn_temp",), _abs_flat),
    (r"layers\.(?P<L>\d+)\.attention\.gater\.weight", ("blk.{L}.attn_gate.weight",), None),
    (r"layers\.(?P<L>\d+)\.attention\.wo\.weight", ("blk.{L}.attn_output.weight",), None),
    (r"layers\.(?P<L>\d+)\.feed_forward\.w_in\.weight", ("blk.{L}.ffn_up.weight", "blk.{L}.ffn_gate.weight"), _unbind0),
    (r"layers\.(?P<L>\d+)\.feed_forward\.w_out\.weight", ("blk.{L}.ffn_down.weight",), None),
    # gate_up_proj rows [0, H) are the gate and [H, 2H) the up projection (tools/convert_lingua_checkpoint.py:
    # cat(w13[:, 0::2], w13[:, 1::2])).
    (
        r"layers\.(?P<L>\d+)\.feed_forward\.experts\.gate_up_proj",
        ("blk.{L}.ffn_gate_exps.weight", "blk.{L}.ffn_up_exps.weight"),
        _halves1,
    ),
    (r"layers\.(?P<L>\d+)\.feed_forward\.experts\.down_proj", ("blk.{L}.ffn_down_exps.weight",), None),
    (r"layers\.(?P<L>\d+)\.feed_forward\.latent_down\.weight", ("blk.{L}.ffn_latent_down.weight",), None),
    (r"layers\.(?P<L>\d+)\.feed_forward\.latent_up\.weight", ("blk.{L}.ffn_latent_up.weight",), None),
    (r"layers\.(?P<L>\d+)\.feed_forward\.router\.router_linear\.weight", ("blk.{L}.router.weight",), None),
    (r"layers\.(?P<L>\d+)\.feed_forward\.router\.rmsnorm_eda\.weight", ("blk.{L}.router_norm.weight",), None),
    (r"layers\.(?P<L>\d+)\.feed_forward\.router\.balancing_biases", ("blk.{L}.router_bias",), None),
)
_LM_PATTERNS = tuple(re.compile(pattern) for pattern, _, _ in LM_NAMES)


def lm_entry(name: str, n_codebooks: int | None = None):
    """(table index, layer or -1, numbered index or -1, GGUF names, transform) of a Zonos3LM parameter."""
    hits = [(i, m) for i, p in enumerate(_LM_PATTERNS) if (m := p.fullmatch(name))]
    if len(hits) != 1:
        raise ValueError(f"LM_NAMES maps parameter {name!r} {len(hits)} times")
    index, match = hits[0]
    groups = match.groupdict()
    _, names, transform = LM_NAMES[index]
    if index == 0:
        if n_codebooks is None:
            raise ValueError("embedder names need n_codebooks")
        if int(groups["i"]) == n_codebooks:
            names = ("text_embd.weight",)
    names = tuple(n.format(**groups) for n in names)
    return index, int(groups.get("L", -1)), int(groups.get("i", -1)), names, transform


def mapped_tensors(model):
    """(GGUF name, tensor) of every Zonos3LM parameter, in LM_NAMES order."""
    n_codebooks = model.config.n_codebooks
    entries = []
    for name, tensor in model.state_dict().items():
        index, layer, number, names, transform = lm_entry(name, n_codebooks)
        entries.append(((layer, index, number), names, transform, tensor))
    for _, names, transform, tensor in sorted(entries, key=lambda e: e[0]):
        parts = transform(tensor) if transform is not None else tensor
        parts = tuple(parts) if isinstance(parts, tuple) else (parts,)
        if len(parts) != len(names):
            raise ValueError(f"{names}: transform produced {len(parts)} tensors")
        yield from zip(names, parts)


def unsupported(key, value):
    raise ValueError(f"unsupported checkpoint: {key}={value!r}")


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def validate_conditions(model, bucket_counts):
    if model.get("condition_names") != CONDITION_NAMES:
        unsupported("params.condition_names", model.get("condition_names"))
    edges, units = model.get("condition_edges"), model.get("condition_units")
    if not isinstance(edges, dict) or set(edges) != set(CONDITION_NAMES):
        unsupported("params.condition_edges", edges)
    if (
        not isinstance(units, dict)
        or set(units) != set(CONDITION_NAMES)
        or not all(isinstance(units[n], str) and units[n] for n in CONDITION_NAMES)
    ):
        unsupported("params.condition_units", units)
    for name, count in zip(CONDITION_NAMES, bucket_counts):
        values = edges[name]
        if (
            not isinstance(values, list)
            or len(values) != count - 1
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
            or any(b <= a for a, b in zip(values, values[1:]))
        ):
            unsupported(f"params.condition_edges.{name}", values)
    return {
        "names": list(CONDITION_NAMES),
        "units": [units[n] for n in CONDITION_NAMES],
        "edges": {n: [float(v) for v in edges[n]] for n in CONDITION_NAMES},
    }


def _run_name(conversion: dict) -> str | None:
    """Training run of conversion.source_checkpoint: <run>/runs/main/checkpoints/<step> or <run>/checkpoints/<step>."""
    parts = Path(str(conversion.get("source_checkpoint", ""))).parts
    if "checkpoints" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("checkpoints")
    if i >= 3 and parts[i - 2 : i] == ("runs", "main"):
        return parts[i - 3]
    return parts[i - 1] if i >= 1 else None


@dataclass
class LMSource:
    model: object  # zonos3.reference.lm.Zonos3LM, float32 weights
    params: dict  # params.json "model" table
    tokenizer: str  # tokenizer JSON text
    spec: dict  # params.text_tokenizer_spec
    meta: dict  # name, qat, conditions, silence_frame, speaker_dim, bucket_counts, tag_cfg_trained


def load_lm(lm_dir, silence_frame=None) -> LMSource:
    """Validate and strictly load an LM directory (see LM_DOC)."""
    from zonos3.reference.lm import Zonos3LM, Zonos3LMConfig
    from zonos3.tts.markup import trained_on_tags

    lm_dir = Path(lm_dir).resolve()
    bake = read_json(lm_dir / "bake.json") if (lm_dir / "bake.json").exists() else None
    conversion = read_json(lm_dir / "conversion.json") if (lm_dir / "conversion.json").exists() else None
    if bake and conversion is None:
        raise ValueError(f"{lm_dir}: a QAT directory needs conversion.json")
    if conversion is not None:
        for key, wanted in {"format": "minisgl-zonos2", "output_dtype": "bfloat16"}.items():
            if conversion.get(key) != wanted:
                unsupported("conversion." + key, conversion.get(key))
        if bake and conversion.get("checkpoint_step") != QAT_STEP:
            unsupported("conversion.checkpoint_step", conversion.get("checkpoint_step"))
    params = read_json(lm_dir / "params.json")["model"]
    config_path = lm_dir / "config.yaml"
    if config_path.exists():
        with config_path.open() as f:
            config = yaml.safe_load(f)
        trained = config.get("model", config)
        for key, value in params.items():
            if key in trained and trained[key] != value:
                unsupported(f"params.json {key} (config.yaml has {trained[key]!r})", value)
        Zonos3LMConfig.from_params({**trained, **params})  # the training config's keys meet the same rules
        data = config.get("data", {})
        tspec = params.get("text_tokenizer_spec") or {}
        for key, wanted in {
            "codec": params.get("audio_codec"),
            "text_tokenizer": "superbpe",
            "text_tokenizer_sha256": tspec.get("sha256"),
            "text_tokenizer_vocab_size": tspec.get("vocab_size"),
            "speaker_embedding_codec": params.get("speaker_embedding_codec"),
        }.items():
            if data.get(key) != wanted:
                unsupported("data." + key, data.get(key))
    cfg = Zonos3LMConfig.from_params(params)
    if cfg.n_codebooks != 10 or cfg.condition_count != len(CONDITION_NAMES):
        unsupported("n_codebooks/condition_count", (cfg.n_codebooks, cfg.condition_count))
    conditions = validate_conditions(params, cfg.condition_bucket_counts)
    if params.get("speaker_reference_max_seconds") != SPEAKER_REFERENCE_MAX_SECONDS:
        unsupported("params.speaker_reference_max_seconds", params.get("speaker_reference_max_seconds"))
    spec = params.get("text_tokenizer_spec")
    if not isinstance(spec, dict) or any(spec.get(k) != v for k, v in TOKENIZER_SPEC.items()) or not spec.get("path"):
        unsupported("params.text_tokenizer_spec", spec)
    raw = (lm_dir / spec["path"]).read_bytes()
    if spec.get("sha256") and sha256_bytes(raw) != spec["sha256"]:
        unsupported("tokenizer SHA256", sha256_bytes(raw))
    silence_path = (
        Path(silence_frame)
        if silence_frame
        else lm_dir / ".." / ".." / "registries" / f"silence_frame_{cfg.audio_codec}.json"
    )
    silence = read_json(silence_path)
    frame = silence.get("frame")
    if silence.get("codec") != cfg.audio_codec:
        unsupported("silence frame codec", silence.get("codec"))
    if (
        not isinstance(frame, list)
        or len(frame) != cfg.n_codebooks
        or not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < cfg.codebook_size for v in frame)
    ):
        unsupported("silence frame", frame)
    model = Zonos3LM.load(lm_dir, dtype=torch.float32)
    qat = qat_targets(bake, model.state_dict()) if bake else None
    if bake:
        name = QAT_GENERAL_NAME
    elif (
        conversion is not None
        and conversion.get("checkpoint_step") == RELEASE_STEP
        and _run_name(conversion) == MODEL_NAME
    ):
        name = GENERAL_NAME
    else:
        run = (conversion and _run_name(conversion)) or lm_dir.name
        step = conversion.get("checkpoint_step") if conversion else None
        name = f"ZONOS3 {run}" + (f" step {step}" if step is not None else "")
    meta = {
        "name": name,
        "qat": qat,
        "conditions": conditions,
        "silence_frame": [int(v) for v in frame],
        "speaker_dim": cfg.speaker_embedding_dim,
        "bucket_counts": list(cfg.condition_bucket_counts),
        "tag_cfg_trained": trained_on_tags(lm_dir),
    }
    return LMSource(model, params, raw.decode("utf-8"), spec, meta)


def qat_targets(bake, sd):
    """{GGUF tensor name: "q4_0"|"q8_0"} for every tensor the QAT forward quantizes (from bake.json)."""
    baked = bake.get("baked", {})
    counts = {s: sum(v == s for v in baked.values()) for s in QAT_COUNTS}
    if counts != QAT_COUNTS or bake.get("model_tensors") != 187:
        unsupported("bake.json tensor counts", counts)
    targets = {}
    for key, scheme in baked.items():
        layer, _, rest = key.removeprefix("layers.").partition(".")
        if rest not in QAT_TARGETS or not layer.isdigit():
            unsupported("bake.json key", key)
        src, names = QAT_TARGETS[rest]
        if scheme != ("q4_0" if ".experts." in rest else "q8_0") or f"layers.{layer}.{src}" not in sd:
            unsupported("bake.json entry", f"{key}={scheme}")
        for name in names:
            targets[f"blk.{layer}.{name}"] = scheme
    return targets


def mix_targets(sd):
    """{GGUF tensor name: "q4_0"|"q8_0"} of a plain directory under --outtype q4_0-exp-q8_0-mix: the tensors the QAT
    forward quantizes (QAT_TARGETS), rounded to nearest from the checkpoint's weights (post-training quantization)."""
    sources = dict(QAT_TARGETS.values())
    targets = {}
    for key in sd:
        layer, _, rest = key.removeprefix("layers.").partition(".")
        if key.startswith("layers.") and layer.isdigit() and rest in sources:
            for name in sources[rest]:
                targets[f"blk.{layer}.{name}"] = "q4_0" if ".experts." in rest else "q8_0"
    if not targets:
        unsupported("Q4_0-expert/Q8_0-mixer targets", 0)
    return targets


def is_high_precision(name):
    return (
        name in ("output.weight", "text_embd.weight", "attnres_q.weight", "speaker_proj.weight")
        or name.startswith(("audio_embd.", "cond_embd."))
        or name.endswith(".router.weight")
        or name.endswith(".gdn_in.weight")
    )


def pick_qtype(name, ndim, ne0, outtype, targets=None):
    if ndim <= 1 or name.endswith(".gdn_conv.weight"):
        return "f32"
    if outtype == MIX_OUTTYPE:
        scheme = targets.get(name)
        if scheme:
            if ne0 % 32:
                raise ValueError(f"{name}: row length {ne0} is not a multiple of the 32-element quant block")
            return GGMLQuantizationType.Q4_0 if scheme == "q4_0" else GGMLQuantizationType.Q8_0
        return "f16"
    if outtype in ("f32", "f16"):
        return outtype
    if outtype == "q8_0":
        return "f16" if is_high_precision(name) or ne0 % 32 else GGMLQuantizationType.Q8_0
    raise ValueError(f"unknown outtype {outtype}")


def fit_baked_blocks(data, qtype):
    """Recover the exact Q4_0/Q8_0 blocks of a QAT-baked tensor.

    The QAT forward uses bf16(q * d16) with an fp16 scale d16; storing the bf16 values loses scale bits, so
    quantizing them afresh would shift every block scale by up to ~0.4%. Search the fp16 scales around the
    bf16-derived one for the (q, d16) that reproduces the baked block bit-for-bit. Returns (raw blocks, number of
    blocks without an exact reproduction)."""
    q4 = qtype == GGMLQuantizationType.Q4_0
    lo, hi = (-8, 7) if q4 else (-127, 127)
    blocks = torch.from_numpy(data.reshape(-1, 32))
    if q4:
        d0 = blocks.gather(1, blocks.abs().argmax(1, keepdim=True)).squeeze(1) / -8
    else:
        d0 = blocks.abs().amax(1) / 127
    bits0 = d0.to(torch.float16).view(torch.int16)
    best_err = torch.full((blocks.shape[0],), 1 << 30, dtype=torch.int32)
    best_bits = bits0.clone()
    best_q = torch.zeros(blocks.shape, dtype=torch.int16)
    for k in range(-8, 9):
        bits = (bits0 + k).to(torch.int16)
        d = bits.view(torch.float16).float()
        inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d)).unsqueeze(1)
        q = torch.round(blocks * inv).clamp_(lo, hi)
        err = ((q * d.unsqueeze(1)).to(torch.bfloat16).float() != blocks).sum(1).to(torch.int32)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_bits = torch.where(better, bits, best_bits)
        best_q = torch.where(better.unsqueeze(1), q.to(torch.int16), best_q)
    scale = best_bits.numpy().astype("<i2").view(np.uint8).reshape(-1, 2)
    if q4:
        u = (best_q + 8).numpy().astype(np.uint8)
        codes = (u[:, :16] | (u[:, 16:] << 4)).astype(np.uint8)
    else:
        codes = best_q.numpy().astype(np.int8).view(np.uint8)
    raw = np.concatenate([scale, codes], axis=1)
    return raw, int((best_err > 0).sum())


def add_tensor(writer, name, tensor, qtype, baked=False):
    data = np.ascontiguousarray(tensor.detach().float().numpy())
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite values in {name}")
    if qtype == "f32":
        writer.add_tensor(name, data)
    elif qtype == "f16":
        half = data.astype(np.float16)
        if not np.isfinite(half).all():
            raise ValueError(f"F16 overflow in {name}: max|x|={np.abs(data).max()}")
        writer.add_tensor(name, half)
    elif data.ndim in (2, 3) and baked:
        raw, inexact = fit_baked_blocks(data, qtype)
        writer.add_tensor(name, raw.reshape(*data.shape[:-1], -1), raw_dtype=qtype)
        return inexact
    elif data.ndim in (2, 3):
        # A nonzero scale that underflows F16 dequantizes to zero; gguf's reciprocal overflows
        # there, so store the zero code (lingua's result) instead of NaN-to-integer casts.
        with np.errstate(over="ignore", invalid="ignore"):
            quant = gquants.quantize(data.reshape(-1, data.shape[-1]), qtype)
        blocks = quant.reshape(-1, gquants.GGML_QUANT_SIZES[qtype][1])
        dead = blocks[:, :2].copy().view("<f2")[:, 0] == 0
        blocks[dead, 2:] = 0x88 if qtype == GGMLQuantizationType.Q4_0 else 0
        writer.add_tensor(name, quant.reshape(*data.shape[:-1], -1), raw_dtype=qtype)
    else:
        raise ValueError(f"unsupported quantized rank for {name}: {data.ndim}")
    return 0


def write_lm_metadata(writer, src: LMSource, outtype):
    from zonos3.reference.lm import GDN_HEAD_DIM, GDN_HEADS, GDN_KERNEL

    c, meta, spec = src.model.config, src.meta, src.spec
    writer.add_string("general.name", meta["name"])
    ints = {
        "block_count": c.n_layers,
        "context_length": c.max_seqlen,
        "embedding_length": c.dim,
        "head_dim": c.head_dim,
        "attention.head_count": c.n_heads,
        "attention.head_count_kv": c.n_kv_heads,
        "feed_forward_length": c.ffn_hidden,
        "expert_count": c.n_experts,
        "n_codebooks": c.n_codebooks,
        "codebook_size": c.codebook_size,
        "audio_vocab": c.audio_vocab,
        "eoa_id": c.eoa_id,
        "audio_pad_id": c.audio_pad_id,
        "text_vocab": c.text_vocab,
        "gdn.head_count": GDN_HEADS,
        "gdn.head_dim": GDN_HEAD_DIM,
        "gdn.conv_kernel": GDN_KERNEL,
        "speaker.embedding_length": meta["speaker_dim"],
        "condition.count": c.condition_count,
    }
    for key, value in ints.items():
        writer.add_uint32(f"{ARCH}.{key}", value)
    for key, value in {
        "rope.freq_base": c.rope_theta,
        "attention.layer_norm_rms_epsilon": c.norm_eps,
        "attention.qk_norm_epsilon": 1e-6,
        "logit_softcap": c.loss_softcap,
    }.items():
        writer.add_float32(f"{ARCH}.{key}", float(value))
    writer.add_array(f"{ARCH}.expert_used_count_per_layer", [0] + [c.moe_topk] * (c.n_layers - 1))
    # one bucket count per condition table (table i: counts[i] + 1 rows, id counts[i] = unknown)
    writer.add_array(f"{ARCH}.condition.bucket_counts", [int(n) for n in meta["bucket_counts"]])
    for key, value in {
        "codec": "mossdual",
        "moe.balancing": "quantile",
        "attention.pattern": src.params["attention_pattern"],
        "residual.type": "full_attnres",
        "tokenizer.json": src.tokenizer,
    }.items():
        writer.add_string(f"{ARCH}.{key}", value)
    for key in ("offset", "bos", "eos"):
        writer.add_uint32(f"{ARCH}.tokenizer.{key}", spec[key])
    conditions = meta["conditions"]
    writer.add_array(f"{ARCH}.condition.names", conditions["names"])
    writer.add_array(f"{ARCH}.condition.units", conditions["units"])
    for name in conditions["names"]:
        writer.add_array(f"{ARCH}.condition.edges.{name}", [float(np.float32(v)) for v in conditions["edges"][name]])
    writer.add_array(f"{ARCH}.silence_frame", meta["silence_frame"])
    writer.add_string(f"{ARCH}.text_tokenizer_spec", json.dumps(spec, ensure_ascii=False))
    if c.span_end_id is not None:
        writer.add_uint32(f"{ARCH}.span_end_id", c.span_end_id)
    if meta["tag_cfg_trained"]:
        writer.add_bool(f"{ARCH}.tag_cfg_trained", True)
    writer.add_file_type({"f32": 0, "f16": 1, "q8_0": 7, MIX_OUTTYPE: 2}[outtype])


def cmd_lm(args, parser):
    src = load_lm(args.lm_dir, args.silence_frame)
    if args.edit_adapter:
        from zonos3.models.accent_format import load_adapter

        adapter = load_adapter(args.edit_adapter)
        if adapter.grown is None:
            parser.error(
                f"--edit-adapter {args.edit_adapter}: a {adapter.format} adapter carries no vocabulary "
                "(zonos-accent-lora/3 needed)"
            )
        src.model.grow(adapter.grown)
        src.meta["name"] += f" + {adapter.name} vocabulary"
    if args.phoneme_rows:
        from zonos3.models.text_extra import TextExtraRows

        extra = TextExtraRows.load(args.phoneme_rows)
        existing = src.model.config.text_extra_tokens
        src.model.append_text_rows(extra)
        src.spec = extra.spec(src.spec, text_vocab=src.model.config.text_vocab, existing=existing)
        src.meta["name"] += " + phoneme tokens"
    qat = src.meta["qat"]
    args.outtype = args.outtype or (MIX_OUTTYPE if qat else "f16")
    if qat and args.outtype != MIX_OUTTYPE:
        parser.error(
            f"{args.lm_dir} is a Q4_0-expert/Q8_0-mixer QAT checkpoint; export it with --outtype {MIX_OUTTYPE}"
        )
    targets = qat if qat else mix_targets(src.model.state_dict()) if args.outtype == MIX_OUTTYPE else None
    if targets is not None and not qat:
        src.meta["name"] += " Q4_0-expert/Q8_0-mixer PTQ"
    writer = GGUFWriter(args.out, ARCH)
    write_lm_metadata(writer, src, args.outtype)
    counts, inexact, seen = {}, 0, set()
    for name, tensor in mapped_tensors(src.model):
        qtype = pick_qtype(name, tensor.ndim, tensor.shape[-1], args.outtype, targets)
        inexact += add_tensor(writer, name, tensor, qtype, baked=bool(qat) and name in qat)
        seen.add(name)
        label = qtype if isinstance(qtype, str) else qtype.name.lower()
        counts[label] = counts.get(label, 0) + 1
    if qat:
        if set(qat) - seen:
            raise ValueError(f"QAT targets not exported: {sorted(set(qat) - seen)}")
        print(f"QAT grid check: {inexact} blocks without an exact (q, fp16 scale) reproduction", flush=True)
        if inexact:
            raise ValueError("baked weights do not lie on the Q4_0/Q8_0 grid; refusing to export")
    print(
        f"Exported {len(src.model.state_dict())} model tensors as {len(seen)} GGUF tensors; storage={counts}",
        flush=True,
    )
    write_gguf(writer, progress=True)
    print(f"Wrote {args.out}", flush=True)


# ===================================================================================================== codec-decoder

CODEC_DECODER_DOC = """Convert the ZONOS3 Mini multi-resolution iSTFT decoder to a mossdual_istft GGUF.

<head_dir> is `full_mr` or `keep2_plain`: it holds exactly one config_*.json and step_0050000.pt. The model is
zonos3.codec.istft_head_a_20261002.model.ISTFTCodec: generator.pt supplies the quantizers (RVQ tables folded through
their 1x1 out_proj), the semantic ConvNeXt head and the 44160->44100 resampler kernel; the head checkpoint's
`trainable` dict overrides every decoder weight (transformer stages 0,2,4,6,8 and the iSTFT head). Weight-normed convs
are materialized (w = g * v / ||v||). Transformer, semantic-head and iSTFT-head matrices follow --outtype; every other
tensor is f32 (the Hann windows are stored so the runtime uses torch's exact values)."""

DEC_ARCH = "mossdual_istft"
DEC_STEP = 600000
HEAD_STEP = 50000
PATCHES = [2, 2, 2, 2, 23]  # generator patches (the last, x23, fed the removed conv tail)
STAGE_PATCH = [2, 2, 2, 2, 1]  # patches the runtime applies
VARIANTS = ("full_mr", "keep2_plain")
HEAD_CHECKS = {
    "type": "mr_istft",
    "depth": 8,
    "hidden": 2048,
    "hop": 184,
    "frame_rate": 240,
    "n_ffts": [736, 2208],
    "fir_kernel": 127,
    "trainable_stages": list(range(9)),
}
HEADROOM_DB = 3.0
LAYER_KEYS = [
    "norm1.weight",
    "norm1.bias",
    "qkv.weight",
    "attention_out.weight",
    "attention_scale",
    "norm2.weight",
    "norm2.bias",
    "ffn.0.weight",
    "ffn.3.weight",
    "ffn_scale",
]
LAYER_NAMES = [
    "norm1.weight",
    "norm1.bias",
    "qkv.weight",
    "wo.weight",
    "attn_scale",
    "norm2.weight",
    "norm2.bias",
    "ffn1.weight",
    "ffn2.weight",
    "ffn_scale",
]
LAYER_MAT = {"qkv.weight", "wo.weight", "ffn1.weight", "ffn2.weight"}


def codec_fail(msg):
    raise SystemExit(f"unsupported codec checkpoint: {msg} (zonos3.cpp runs only the ZONOS3 Mini iSTFT decoder)")


def validate_generator(state):
    if state.get("step") != DEC_STEP:
        codec_fail(f"step={state.get('step')}")
    cfg = state["config"]["model"]
    for key, want in {
        "architecture": "s1_semantic_rvq_moss_v2_hybrid",
        "sample_rate": 44100,
        "internal_sample_rate": 44160,
        "frame_rate": 15,
        "internal_hop_length": 2944,
        "external_samples_per_frame": 2940,
    }.items():
        if cfg.get(key) != want:
            codec_fail(f"{key}={cfg.get(key)}")
    bn = cfg["bottleneck"]
    for key, want in {
        "type": "dual_encoding_semantic_residual_rvq",
        "semantic_codebooks": 1,
        "residual_codebooks": 9,
        "semantic_codebook_size": 4096,
        "residual_codebook_size": 4096,
        "codebook_dim": 8,
    }.items():
        if bn.get(key) != want:
            codec_fail(f"bottleneck.{key}={bn.get(key)}")
    sh = bn["semantic_head"]
    if (sh.get("type"), sh.get("causal"), sh.get("width"), sh.get("out_channels")) != (
        "ConvNeXtDecoder",
        True,
        768,
        1024,
    ):
        codec_fail(f"semantic_head={sh}")
    stages = cfg["decoder_stages"]
    kinds = [s["type"] for s in stages]
    want_kinds = ["MossProjectedTransformer", "PatchedPretransform"] * 5 + ["S1DecoderBlock"] * 2 + ["S1WaveformHead"]
    if kinds != want_kinds:
        codec_fail(f"decoder stage types {kinds}")
    moss = stages[0:10:2]
    for s in moss:
        if (s["causal"], s["norm"], s["positional_embedding"], s["max_period"], s["gating"], s["activation"]) != (
            True,
            "layer_norm",
            "rope",
            10000,
            "none",
            "gelu",
        ):
            codec_fail(f"moss stage {s}")
    patches = [s["patch_size"] for s in stages[1:10:2]]
    if patches != PATCHES or any(s["direction"] != "up" for s in stages[1:10:2]):
        codec_fail(f"patches {patches}")
    return cfg, moss, sh


def find_head_files(head_dir: Path):
    variant = head_dir.name
    if variant not in VARIANTS:
        codec_fail(f"head directory name '{variant}' is not one of {VARIANTS}")
    configs = sorted(head_dir.glob("config_*.json"))
    if len(configs) != 1:
        codec_fail(f"{head_dir} must contain exactly one config_*.json (found {[c.name for c in configs]})")
    ckpt = head_dir / f"step_{HEAD_STEP:07d}.pt"
    if not ckpt.is_file():
        codec_fail(f"missing {ckpt}")
    return variant, configs[0], ckpt


def validate_head(hc, ck, gen_cfg, moss, cfg_name):
    if ck.get("step") != HEAD_STEP:
        codec_fail(f"head checkpoint step={ck.get('step')}")
    if hc.get("model") != gen_cfg:
        codec_fail(f"{cfg_name}: model config differs from generator.pt's")
    head = hc["head"]
    for key, want in HEAD_CHECKS.items():
        if head.get(key) != want:
            codec_fail(f"head.{key}={head.get(key)}")
    layers = []
    for s in range(0, 9, 2):
        if "stage_layers" in head and str(s) in head["stage_layers"]:
            n = len(head["stage_layers"][str(s)])
        else:
            n = int(moss[s // 2]["num_layers"])
        layers.append(n)
    want = set()
    for s, n in zip(range(0, 9, 2), layers):
        want.add(f"stages.{s}.input_proj.weight")
        if s < 8:
            want.add(f"stages.{s}.output_proj.weight")
        for layer in range(n):
            want.update(f"stages.{s}.layers.{layer}.{k}" for k in LAYER_KEYS)
    for b in range(head["depth"]):
        want.update(
            f"head.blocks.{b}.{k}"
            for k in (
                "dwconv.weight",
                "dwconv.bias",
                "norm.weight",
                "norm.bias",
                "pwconv1.weight",
                "pwconv1.bias",
                "pwconv2.weight",
                "pwconv2.bias",
            )
        )
    want.update(
        [
            "head.norm.weight",
            "head.norm.bias",
            "head.outs.0.weight",
            "head.outs.0.bias",
            "head.outs.1.weight",
            "head.outs.1.bias",
            "head.mixer.weight",
        ]
    )
    got = set(ck["trainable"])
    if got != want:
        codec_fail(
            f"head checkpoint keys differ: missing {sorted(want - got)[:8]}, unexpected {sorted(got - want)[:8]}"
        )
    return head, layers


def wn(g, v):
    """torch weight norm over every dim but 0."""
    return g * v / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1))


def _wn_key(get, sd, prefix):
    if prefix + ".weight_g" in sd:
        return wn(get(prefix + ".weight_g"), get(prefix + ".weight_v"))
    return wn(get(prefix + ".parametrizations.weight.original0"), get(prefix + ".parametrizations.weight.original1"))


def cmd_codec_decoder(args, parser):
    from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec

    state = torch.load(args.generator, map_location="cpu", mmap=True, weights_only=True)
    gen_cfg, moss, sh = validate_generator(state)
    variant, cfg_path, ckpt_path = find_head_files(Path(args.head))
    hc = json.loads(cfg_path.read_text())
    ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    head, layers = validate_head(hc, ck, gen_cfg, moss, cfg_path.name)
    model = ISTFTCodec(hc["model"], hc["head"])
    model.load_pretrained(state["generator"])
    model.load_trainable_state_dict(ck["trainable"])
    del state, ck
    sd = model.state_dict()
    consumed = set()

    def get(key):
        consumed.add(key)
        return sd[key].float()

    w = GGUFWriter(args.out, DEC_ARCH)
    w.add_string("general.name", f"ZONOS3 Mini MR-iSTFT decoder ({variant}, step {HEAD_STEP})")
    for key, val in {
        "sample_rate": 44100,
        "internal_sample_rate": 44160,
        "frame_rate": 15,
        "hop": 2940,
        "internal_hop": 2944,
        "n_codebooks": 10,
        "codebook_size": 4096,
        "latent_dim": 1024,
        "semantic_dim": sh["width"],
        "sem_head.depth": sh["depth"],
        "head.depth": head["depth"],
        "head.hidden": head["hidden"],
        "head.hop": head["hop"],
        "head.fir_kernel": head["fir_kernel"],
        "head.step": HEAD_STEP,
    }.items():
        w.add_uint32(f"{DEC_ARCH}.{key}", val)
    w.add_array(f"{DEC_ARCH}.head.n_fft", [int(n) for n in head["n_ffts"]])
    w.add_float32(f"{DEC_ARCH}.head.max_log_magnitude", math.log(100.0))
    w.add_float32(f"{DEC_ARCH}.headroom_db", HEADROOM_DB)
    w.add_string(f"{DEC_ARCH}.head.variant", variant)
    w.add_float32(f"{DEC_ARCH}.ln_eps", 1e-5)
    w.add_float32(f"{DEC_ARCH}.convnext_eps", 1e-6)
    for key, field in {
        "in_dim": "input_dimension",
        "out_dim": "output_dimension",
        "d_model": "d_model",
        "n_head": "num_heads",
        "n_ff": "dim_feedforward",
        "context": "context_frames",
    }.items():
        vals = [int(s[field]) for s in moss]
        if key == "out_dim":
            vals[4] = 0  # decoder.8.output_proj is an Identity in the iSTFT codec
        w.add_array(f"{DEC_ARCH}.stage.{key}", vals)
    w.add_array(f"{DEC_ARCH}.stage.n_layer", layers)
    w.add_array(f"{DEC_ARCH}.stage.patch", STAGE_PATCH)

    kernel = get("to_external.kernel")  # [new, 1, 2*width + orig]
    new, _, kl = kernel.shape
    orig, width = 736, (kl - 736) // 2
    if new != 735 or kl != 2 * width + orig:
        codec_fail(f"to_external.kernel shape {tuple(kernel.shape)}")
    w.add_uint32(f"{DEC_ARCH}.resample.orig", orig)
    w.add_uint32(f"{DEC_ARCH}.resample.new", new)
    w.add_uint32(f"{DEC_ARCH}.resample.width", width)

    count = 0

    def put(name, t, mat=False):
        nonlocal count
        a = t.detach().cpu().float().numpy()
        if not np.isfinite(a).all():
            raise SystemExit(f"non-finite values in {name}")
        if mat and args.outtype == "f16":
            h = a.astype(np.float16)
            if not np.isfinite(h).all():
                raise SystemExit(f"F16 overflow in {name}")
            w.add_tensor(name, np.ascontiguousarray(h))
        else:
            w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32))
        count += 1

    def convnext(o, p):
        put(o + ".dw.weight", get(p + ".dwconv.weight").squeeze(1))
        put(o + ".dw.bias", get(p + ".dwconv.bias"))
        put(o + ".norm.weight", get(p + ".norm.weight"))
        put(o + ".norm.bias", get(p + ".norm.bias"))
        put(o + ".pw1.weight", get(p + ".pwconv1.weight"), mat=True)
        put(o + ".pw1.bias", get(p + ".pwconv1.bias"))
        put(o + ".pw2.weight", get(p + ".pwconv2.weight"), mat=True)
        put(o + ".pw2.bias", get(p + ".pwconv2.bias"))

    with torch.no_grad():
        # RVQ codebooks folded through their 1x1 out_proj: table[id] = codebook[id] @ W^T.
        def table(prefix):
            return (
                get(prefix + ".codebook.weight") @ _wn_key(get, sd, prefix + ".out_proj").squeeze(-1).T,
                get(prefix + ".out_proj.bias"),
            )

        t, b = table("quantizer.semantic_quantizer.quantizers.0")
        put("sem.table", t)
        put("sem.bias", b)
        # res.bias = the nine out_proj biases summed (what a complete frame adds); res.{i}.bias = each one alone, which
        # a decode leaving residual codebook i out of a frame (early partial-frame decode) subtracts.
        total = torch.zeros(1024, dtype=torch.float64)
        for i in range(9):
            t, b = table(f"quantizer.quantizer.quantizers.{i}")
            put(f"res.{i}.table", t)
            put(f"res.{i}.bias", b)
            total += b.double()
        put("res.bias", total.float())
        consumed.update(k for k in sd if ".in_proj." in k and k.startswith("quantizer."))  # encode-only

        for b in range(sh["depth"]):
            convnext(f"sem_head.{b}", f"semantic_head.{b}")
        put("sem_head.out.weight", _wn_key(get, sd, f"semantic_head.{sh['depth']}").squeeze(-1), mat=True)
        put("sem_head.out.bias", get(f"semantic_head.{sh['depth']}.bias"))

        for si, s in enumerate(range(0, 9, 2)):
            p, o = f"stages.{s}", f"stage.{si}"
            put(o + ".in.weight", get(p + ".input_proj.weight"), mat=True)
            if s < 8:
                put(o + ".out.weight", get(p + ".output_proj.weight"), mat=True)
            for layer in range(layers[si]):
                for src, dst in zip(LAYER_KEYS, LAYER_NAMES):
                    put(f"{o}.l{layer}.{dst}", get(f"{p}.layers.{layer}.{src}"), mat=dst in LAYER_MAT)

        for b in range(head["depth"]):
            convnext(f"head.{b}", f"head.blocks.{b}")
        put("head.norm.weight", get("head.norm.weight"))
        put("head.norm.bias", get("head.norm.bias"))
        for i, n in enumerate(head["n_ffts"]):
            ow = get(f"head.outs.{i}.weight")
            if tuple(ow.shape) != (n + 2, 768):
                codec_fail(f"head.outs.{i}.weight shape {tuple(ow.shape)}")
            put(f"head.out.{i}.weight", ow, mat=True)
            put(f"head.out.{i}.bias", get(f"head.outs.{i}.bias"))
        mixer = get("head.mixer.weight")
        if tuple(mixer.shape) != (1, len(head["n_ffts"]), head["fir_kernel"]):
            codec_fail(f"head.mixer.weight shape {tuple(mixer.shape)}")
        put("head.mixer.weight", mixer.squeeze(0))
        for i, n in enumerate(head["n_ffts"]):
            put(f"head.window.{i}", torch.hann_window(n, periodic=True))
        put("resample.kernel", kernel.squeeze(1))

    leftover = sorted(k for k in sd if k not in consumed)
    if leftover:
        raise SystemExit(f"{len(leftover)} ISTFTCodec tensors not mapped, e.g.: {leftover[:8]}")
    write_gguf(w)
    print(f"wrote {count} {DEC_ARCH} tensors to {args.out}")


# ===================================================================================================== codec-encoder

CODEC_ENCODER_DOC = """Export the frozen dual codec encode path (zonos3.codec MossCodec, the model MossAutoencoder encodes
with) to a separate mossdual_encoder GGUF.

Stores original RVQ codevectors and both projections in F32, never an inverse of folded decode tables. Acoustic
convolutions are tap-major [K,OC,IC], ConvNeXt kernels [K,C], matching the decoder's native repacked graph convention."""

ENC_ARCH = "mossdual_encoder"


def cmd_codec_encoder(args, parser):
    from zonos3.codec.hybrid10_dual_encoding.model import MossCodec

    state = torch.load(args.generator, map_location="cpu", mmap=True, weights_only=True)
    cfg, _, sh = validate_generator(state)
    specs = cfg["encoder_stages"]
    if [s["type"] for s in specs] != ["S1CausalWNConv1d", "S1EncoderBlock", "S1EncoderBlock"] + [
        "PatchedPretransform",
        "MossProjectedTransformer",
    ] * 5:
        raise SystemExit("unsupported encoder stages")
    patches = [s["patch_size"] for s in specs[3::2]]
    if patches != [23, 2, 2, 2, 2] or any(s["direction"] != "down" for s in specs[3::2]):
        raise SystemExit("unsupported encoder patching")
    if [s["stride"] for s in specs[1:3]] != [2, 4] or cfg["dual_encoding"] != {
        "width": 768,
        "hidden": 2048,
        "depth": 4,
        "causal": True,
        "window_ratio": 2.0,
        "strategy": "frozen_ssl_to_q1_decoded_semantic_residual",
        "normalization": "unchanged_baseline",
    }:
        raise SystemExit("unsupported S1/semantic encoder configuration")
    moss = specs[4::2]
    model = MossCodec(cfg)
    model.load_state_dict(state["generator"], strict=True, assign=True)
    sd = model.state_dict()
    consumed = set()

    def get(k):
        consumed.add(k)
        return sd[k].detach().float()

    def wnp(p):
        return _wn_key(get, sd, p)

    w = GGUFWriter(args.out, ENC_ARCH)
    w.add_string("general.name", "fixed600k frozen dual Opus reference encoder")
    for key, value in {
        "sample_rate": 44100,
        "internal_sample_rate": 44160,
        "frame_rate": 15,
        "hop": 2940,
        "internal_hop": 2944,
        "n_codebooks": 10,
        "codebook_size": 4096,
        "latent_dim": 1024,
        "semantic_dim": 768,
        "semantic_depth": 4,
        "lane_frames": 300,
        "context_frames": 150,
    }.items():
        w.add_uint32(f"{ENC_ARCH}.{key}", value)
    w.add_float32(f"{ENC_ARCH}.window_ratio", cfg["dual_encoding"]["window_ratio"])
    w.add_float32(f"{ENC_ARCH}.ln_eps", 1e-5)
    w.add_float32(f"{ENC_ARCH}.convnext_eps", 1e-6)
    for key, field in {
        "in_dim": "input_dimension",
        "out_dim": "output_dimension",
        "d_model": "d_model",
        "n_head": "num_heads",
        "n_layer": "num_layers",
        "n_ff": "dim_feedforward",
        "context": "context_frames",
    }.items():
        w.add_array(f"{ENC_ARCH}.stage.{key}", [int(s[field]) for s in moss])
    w.add_array(f"{ENC_ARCH}.stage.patch", patches)
    count = 0

    def put(name, t, mat=False):
        nonlocal count
        a = t.cpu().numpy().astype(np.float16 if mat and args.outtype == "f16" else np.float32)
        if not np.isfinite(a).all():
            raise SystemExit(f"nonfinite/overflow tensor {name}")
        w.add_tensor(name, np.ascontiguousarray(a))
        count += 1

    def conv(o, p):
        # Raw torch [OC,IC,K] -> ggml [IC,OC,K], taps contiguous matrices.
        put(o + ".weight", wnp(p + ".conv").permute(2, 0, 1))
        put(o + ".bias", get(p + ".conv.bias").reshape(-1, 1))

    def snake(o, p):
        alpha = get(p).reshape(-1, 1)
        put(o + ".alpha", alpha)
        put(o + ".inv_alpha", 1 / (alpha + 1e-9))

    def convnext(o, p):
        put(o + ".dw.weight", get(p + ".dwconv.weight").squeeze(1).T)
        put(o + ".dw.bias", get(p + ".dwconv.bias"))
        for name in ["norm.weight", "norm.bias"]:
            put(o + "." + name, get(p + "." + name))
        for src, dst in [("pwconv1", "pw1"), ("pwconv2", "pw2")]:
            put(o + "." + dst + ".weight", get(p + "." + src + ".weight"))
            put(o + "." + dst + ".bias", get(p + "." + src + ".bias"))

    with torch.no_grad():
        kernel = get("to_internal.kernel").squeeze(1)
        new, kl = kernel.shape
        orig = 735
        width = (kl - orig) // 2
        if new != 736 or kl != 2 * width + orig:
            raise SystemExit("unsupported encoder resampling kernel")
        for key, value in {"orig": orig, "new": new, "width": width}.items():
            w.add_uint32(f"{ENC_ARCH}.resample.{key}", value)
        put("resample.kernel", kernel)
        conv("enc.in", "encoder.0")
        for b in range(2):
            p, o = f"encoder.{b + 1}.module.block", f"enc.b{b}"
            for r in range(3):
                rp, ro = f"{p}.{r}.module.block", f"{o}.res{r}"
                snake(ro + ".snake1", rp + ".0.alpha")
                conv(ro + ".conv1", rp + ".1")
                snake(ro + ".snake2", rp + ".2.alpha")
                conv(ro + ".conv2", rp + ".3")
            snake(o + ".snake", p + ".3.alpha")
            conv(o + ".conv", p + ".4")
        for s, spec in enumerate(moss):
            p, o = f"encoder.{4 + 2 * s}", f"stage.{s}"
            put(o + ".in.weight", get(p + ".input_proj.weight"), mat=True)
            put(o + ".out.weight", get(p + ".output_proj.weight"), mat=True)
            for layer in range(spec["num_layers"]):
                lp, lo = f"{p}.layers.{layer}", f"{o}.l{layer}"
                for src, dst in [
                    ("norm1.weight", "norm1.weight"),
                    ("norm1.bias", "norm1.bias"),
                    ("norm2.weight", "norm2.weight"),
                    ("norm2.bias", "norm2.bias"),
                    ("attention_scale", "attn_scale"),
                    ("ffn_scale", "ffn_scale"),
                    ("qkv.weight", "qkv.weight"),
                    ("attention_out.weight", "wo.weight"),
                    ("ffn.0.weight", "ffn1.weight"),
                    ("ffn.3.weight", "ffn2.weight"),
                ]:
                    put(
                        lo + "." + dst,
                        get(lp + "." + src),
                        mat=dst in ["qkv.weight", "wo.weight", "ffn1.weight", "ffn2.weight"],
                    )
        put("sem_enc.in.weight", wnp("semantic_encoder.0").squeeze(-1))
        put("sem_enc.in.bias", get("semantic_encoder.0.bias"))
        for b in range(sh["depth"]):
            convnext(f"sem_enc.{b}", f"semantic_encoder.{b + 1}")
            convnext(f"sem_head.{b}", f"semantic_head.{b}")
        put("sem_head.out.weight", wnp(f"semantic_head.{sh['depth']}").squeeze(-1))
        put("sem_head.out.bias", get(f"semantic_head.{sh['depth']}.bias"))
        for c in range(10):
            p = "quantizer.semantic_quantizer.quantizers.0" if c == 0 else f"quantizer.quantizer.quantizers.{c - 1}"
            for proj in ["in_proj", "out_proj"]:
                put(f"rvq.{c}.{proj}.weight", wnp(p + "." + proj).squeeze(-1))
                put(f"rvq.{c}.{proj}.bias", get(p + "." + proj + ".bias"))
            put(f"rvq.{c}.codebook", get(p + ".codebook.weight"))
    required = ("encoder.", "semantic_encoder.", "semantic_head.", "quantizer.", "to_internal.")
    leftover = sorted(k for k in sd if k.startswith(required) and k not in consumed)
    if leftover:
        raise SystemExit(f"unmapped encoder tensors: {leftover}")
    write_gguf(w)
    print(f"wrote {count} encoder tensors to {args.out}")


# ===================================================================================================== ssl-teacher

SSL_DOC = """Export the matched sixteen-layer teacher (transformers Wav2Vec2BertModel, teacher/model.safetensors) to
w2vbert_ssl GGUF.

The only non-inference checkpoint tensor, masked_spec_embed, is explicitly consumed and checked: spec augmentation is
disabled in this frozen SemanticTeacher. Frontend tables retain their original float64 precision; resampling
coefficients are the exact default torchaudio.transforms.Resample(44100, 16000) float32 kernel. Use --outtype bf16 for
production matrix storage without runtime weight casts; normalization, biases and depthwise/frontend tables keep their
opmath precision."""

SSL_ARCH = "w2vbert_ssl"
SSL_EXPECTED_CONFIG = {
    "model_type": "wav2vec2-bert",
    "num_hidden_layers": 16,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "feature_projection_input_dim": 160,
    "conv_depthwise_kernel_size": 31,
    "left_max_position_embeddings": 64,
    "right_max_position_embeddings": 8,
    "position_embeddings_type": "relative_key",
    "hidden_act": "swish",
    "layer_norm_eps": 1e-5,
    "add_adapter": False,
    "apply_spec_augment": False,
    "use_intermediate_ffn_before_adapter": False,
}


def ssl_mappings():
    # Each entry contains GGUF name, model parameter name, exact shape,
    # optional squeezed convolution axes, and whether weights must remain F32.
    result = []

    def tensor(dst, src, shape, squeeze=(), sensitive=False):
        result.append((dst, src, tuple(shape), squeeze, sensitive))

    def pair(dst, src, inp, out):
        tensor(dst + ".weight", src + ".weight", (out, inp))
        tensor(dst + ".bias", src + ".bias", (out,), sensitive=True)

    def norm(dst, src, dim):
        tensor(dst + ".weight", src + ".weight", (dim,), sensitive=True)
        tensor(dst + ".bias", src + ".bias", (dim,), sensitive=True)

    norm("fp.ln", "feature_projection.layer_norm", 160)
    pair("fp.proj", "feature_projection.projection", 160, 1024)
    for i in range(16):
        src, dst = f"encoder.layers.{i}.", f"enc.{i}."
        for j in (1, 2):
            norm(dst + f"ffn{j}_ln", src + f"ffn{j}_layer_norm", 1024)
            pair(dst + f"ffn{j}.up", src + f"ffn{j}.intermediate_dense", 1024, 4096)
            pair(dst + f"ffn{j}.down", src + f"ffn{j}.output_dense", 4096, 1024)
        norm(dst + "attn_ln", src + "self_attn_layer_norm", 1024)
        for a, b in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "out")):
            pair(dst + "attn." + a, src + "self_attn.linear_" + b, 1024, 1024)
        tensor(dst + "attn.dist", src + "self_attn.distance_embedding.weight", (73, 64), sensitive=True)
        norm(dst + "conv_ln", src + "conv_module.layer_norm", 1024)
        tensor(dst + "conv.pw1.weight", src + "conv_module.pointwise_conv1.weight", (2048, 1024, 1), (2,))
        tensor(dst + "conv.dw.weight", src + "conv_module.depthwise_conv.weight", (1024, 1, 31), (1,), True)
        norm(dst + "conv.dw_ln", src + "conv_module.depthwise_layer_norm", 1024)
        tensor(dst + "conv.pw2.weight", src + "conv_module.pointwise_conv2.weight", (1024, 1024, 1), (2,))
        norm(dst + "final_ln", src + "final_layer_norm", 1024)
    return result


def cmd_ssl_teacher(args, parser):
    import torchaudio
    from gguf import GGMLQuantizationType as Q
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

    checkpoint = args.checkpoint / "model.safetensors" if args.checkpoint.is_dir() else args.checkpoint
    if checkpoint.name != "model.safetensors":
        raise ValueError("expected the actual teacher/model.safetensors checkpoint")
    config = json.loads((checkpoint.parent / "config.json").read_text())
    for key, expected in SSL_EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"unsupported teacher configuration {key}: {config.get(key)!r}, expected {expected!r}")
    if config.get("architectures") != ["Wav2Vec2BertModel"]:
        raise ValueError("teacher must be the base Wav2Vec2BertModel, not a task head")
    model, info = Wav2Vec2BertModel.from_pretrained(checkpoint.parent, dtype=torch.float32, output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"teacher load is not strict: {info}")
    sd = model.state_dict()
    mapping = ssl_mappings()
    expected = {src: shape for _, src, shape, _, _ in mapping}
    expected["masked_spec_embed"] = (1024,)
    keys = set(sd)
    if keys != set(expected):
        raise ValueError(
            f"teacher tensor schema mismatch: missing={sorted(set(expected) - keys)}, unexpected={sorted(keys - set(expected))}"
        )
    for key, shape in expected.items():
        if tuple(sd[key].shape) != shape:
            raise ValueError(f"wrong shape for {key}: {tuple(sd[key].shape)}, expected {shape}")
    if not torch.isfinite(sd["masked_spec_embed"]).all():
        raise ValueError("nonfinite training-only masked_spec_embed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(args.out), SSL_ARCH)
    w.add_string("general.name", "Matched Wav2Vec2-BERT semantic teacher layer 16")
    meta = {
        "sample_rate": 16000,
        "max_seconds": 60,
        "n_layer": 16,
        "n_embd": 1024,
        "n_head": 16,
        "n_ff": 4096,
        "rel_left": 64,
        "rel_right": 8,
        "conv_kernel": 31,
        "n_mels": 80,
        "frame_length": 400,
        "hop_length": 160,
        "fft_length": 512,
        "stride": 2,
    }
    for key, value in meta.items():
        w.add_uint32(f"{SSL_ARCH}.{key}", value)
    for key, value in {"eps": 1e-5, "preemphasis": 0.97, "mel_floor": 1.192092955078125e-07}.items():
        w.add_float32(f"{SSL_ARCH}.{key}", value)
    w.add_float32(f"{SSL_ARCH}.padding_value", 1.0)
    fe = SeamlessM4TFeatureExtractor(feature_size=80, sampling_rate=16000, num_mel_bins=80, padding_value=1, stride=2)
    for name, value in (("frontend.window", fe.window), ("frontend.mel_filters", fe.mel_filters)):
        w.add_tensor(name, np.ascontiguousarray(value, dtype=np.float64), raw_dtype=Q.F64)
    resampler = torchaudio.transforms.Resample(44100, 16000)
    w.add_tensor("frontend.resample_44100", resampler.kernel[:, 0, :].contiguous().numpy())
    consumed = {"masked_spec_embed"}
    for dst, src, _, squeeze, sensitive in mapping:
        value = sd[src].float().numpy()
        if squeeze:
            value = np.squeeze(value, axis=squeeze)
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite tensor: {src}")
        if args.outtype == "f16" and value.ndim == 2 and not sensitive:
            if np.max(np.abs(value)) >= 65504:
                raise ValueError(f"f16 overflow: {src}")
            value = value.astype(np.float16)
        if args.outtype == "bf16" and value.ndim == 2 and not sensitive:
            bits = torch.from_numpy(np.ascontiguousarray(value)).to(torch.bfloat16).view(torch.uint16).numpy()
            w.add_tensor(dst, bits, raw_dtype=Q.BF16)
        else:
            w.add_tensor(dst, np.ascontiguousarray(value))
        consumed.add(src)
    if consumed != keys:
        raise ValueError(f"unconsumed teacher tensors: {sorted(keys - consumed)}")
    write_gguf(w, progress=True)
    print(
        f"consumed {len(consumed)} checkpoint tensors; exported {len(mapping)} inference tensors and 3 frontend tables -> {args.out}",
        flush=True,
    )


# ===================================================================================================== speaker-encoder

SPK_DOC = """Export the ZONOS3 Mini speaker encoder (ownmect_s2, MECT-B2 + ASTP, 192-d) to GGUF.

The model is zonos3.models.mect_speaker.MECTFbankSpeakerEmbedding (vendored AntSpeaker VPR: MECT_B2 backbone, ASTP
pooling, BatchNorm, linear 192) loaded strictly from the sha256-checked checkpoint named by the registry entry.
Everything is stored F32; BatchNorm (eval) is folded to scale/shift in float64.

Notation. `ne=[a,b,c,d]` is the ggml shape (a = fastest axis). The numpy array handed to the GGUF writer has the
REVERSED shape, so torch layouts map 1:1: torch (OC,IC,KH,KW) == ggml ne [KW,KH,IC,OC] (ggml_conv_2d_direct kernel),
torch Linear weight (out,in) == ggml ne [in,out] (ggml_mul_mat weight), torch (C,1,K) depthwise Conv1d weight squeezed
to (C,K) == ggml ne [K,C]. Activations are WHCN `[T, F, C, 1]`: time = ne0 (torch W), mel/feature axis = ne1 (torch
H), channels = ne2.

Frontend (host, fp32; MECTFbankSpeakerEmbedding.fbank + VPR.cmn)
  wav16k (<=960000 samples) * 32768 -> torchaudio.compliance.kaldi.fbank(num_mel_bins=80, frame_length=25,
  frame_shift=10, dither=0, energy_floor=0, window_type=hamming, snip_edges=True, preemph 0.97, remove_dc_offset,
  round_to_power_of_two (n_fft 512), use_power, low_freq 20, high_freq 0 (=Nyquist), htk_compat False,
  use_energy False) -> fbank [T,80], T = 1 + (N-400)//160
  per frame: x -= mean(x); y[j] = x[j] - 0.97*x[max(j-1,0)]; y *= frontend.window; zero-pad to 512; rfft;
  power = re^2+im^2 (257 bins); mel = frontend.mel_banks[80,257] . power; fbank = log(max(mel, 1.1920928955078125e-07)).
  CMN: fbank_cmn = fbank - mean_over_time(fbank) (per mel bin). Network input x0 = [T, 80, 1, 1].

Head:  x = BN(conv3x3(x0; fe.head.conv, pad 1) + bias) -> [T, 80, 32]

MECT_B2 stages (name fe.b{S}.{K}.*; S = 1..4, K = block index; torch mect_block{S}.{K}):
  stage 1: 2 blocks  K=0,1       stride (F,T) (1,1)        cnn depth 3,3          expansion 4  (OC 128)
  stage 2: 4 blocks  K=0..3      K=0: (2,1)  K>=1: (1,2)   cnn depth 3,3,3,3      expansion 2  (OC 128)
  stage 3: 5 blocks  K=0..4      K=0: (2,1)  K>=1: (1,2)   cnn depth 3,3,3,3,3    expansion 2  (OC 256)
  stage 4: 3 blocks  K=0..2      K=0: (2,1)  K>=1: (1,2)   cnn depth 3,2,2        expansion 1  (OC 256)
  out channels per block: 32 / 64 / 128 / 256 (S=1..4); transformer width H = 64 in every block.
  (F-stride s_f maps to ggml stride1, T-stride s_t to ggml stride0 of dw/pw convs.)

  Block input x_in [T, F_in, C_in]; s_f, s_t = block strides; OC = out_channels*expansion.
   1. cnn = BasicResBlock chain (`cnn.{R}`, R = 0..depth-1); res block R=0 has stride (s_f, s_t) and input channels
      C_in; R>=1 stride 1 and input channels OC. For each res block (IC -> OC):
        t  = grouped3x3(x; conv1, stride (s_f,s_t), pad (1,1))          groups G1 = IC (see below)
        t  = pw1(t) (1x1 conv + bias);  t = relu(BN1(t))
        t  = grouped3x3(t; conv2, stride 1, pad (1,1))                  groups G2 = G1 (conv_groups of the block)
        t  = pw2(t) (1x1 conv + bias);  t = relu(BN2(t))
        out = t + shortcut(x)
      shortcut(x) = BN(conv1x1_nobias(x; stride (s_f,s_t))) (`sc.weight` ne [1,1,IC,OC], `sc.bn.*`) for R=0 of every
      block of stages 2-4 (torch stride is a tuple there, which always creates a shortcut; (1,2) blocks included) and
      of stage 1 (IC 32 != OC 128); R>=1 res blocks have identity shortcut. Only R=0 has `sc.*` tensors.
      Output spatial size per conv with kernel 3 pad 1: n_out = (n + 2 - 3)//s + 1; the 1x1 stride-s shortcut gives
      (n - 1)//s + 1 (identical).
      Grouped conv: torch Conv2d(IC, OC, 3, groups=Gc, bias=False) weight (OC, a, 3, 3), a = IC_conv/Gc. Group g
      covers input channels [g*a, (g+1)*a) and output channels [g*b, (g+1)*b), b = OC/Gc. The GGUF stores one tensor
      per input slot j<a: `w{j}` = weight[:, j:j+1] (torch (OC,1,3,3) == ggml ne [3,3,1,OC], the
      ggml_conv_2d_dw_direct kernel). y = sum_j dwconv(w{j}, x_slot_j repeated b times per group) where x_slot_j =
      channels {g*a+j}_g (view x [T,F,a,G], take slice j -> [T,F,1,G], repeat to [T,F,b,G], reshape [T,F,OC]:
      channel index oc = g*b + (0..b-1)).
      Actual slot counts a: R=0: conv1 a=1 (b=OC/C_in), conv2 a=OC/C_in (G=C_in groups, b=a); R>=1: a=1 (depthwise,
      G=OC, b=1) for both convs. E.g. stage 1: conv2 a=4; stage 2 K=0: a=4, K>=1 (C_in 64): a=2; stage 4 K=0: a=2.
   2. reduce (only expansion>1; `reduce.*`): y = BN(conv1x1(cnn_out; reduce.weight, bias)) with OC -> C' = C_in*s_f.
      expansion==1 (stage 4): identity (OC == C_in*s_f). Result r [T', F', C'] with F' = F_in//s_f, T' = cnn output
      length, F'*C' = 2560 always.
   3. x2 = flatten(r) -> [T', 2560], flat index k = f*C' + c (torch r.permute(0,2,1,3).reshape(b, C'*F', T')):
      in ggml: permute r [T',F',C'] -> [T', C', F'] (c fastest) ... -> reshape [T', 2560].
   4. h = LN_cf(rproj(x2)): `rproj.weight` ggml ne [2560, 64] (torch Conv1d(2560,64,1) squeezed) + `rproj.bias`, then
      channels-first LayerNorm over the 64 channels per time step, eps 1e-6: `rproj.ln.{weight,bias}`.
   5. h = h + PosConv(h): zero pad 56 on both sides of time; depthwise Conv1d k=113 over the 64 channels
      (`pos.dw.weight` ggml ne [113, 64]: tap-major per channel, `pos.dw.bias`); BatchNorm1d (`pos.bn.*`); exact GELU
      (erf); pointwise Conv1d 64->64 (`pos.pw.weight` ggml ne [64,64] + `pos.pw.bias`).
   6. Post-norm TransformerEncoderLayer on [T', 64] (dropout off):
        a = attn(h): q = q_proj(h)*0.25 (head_dim 16, scaling 16^-0.5), k = k_proj(h), v = v_proj(h) (all with bias),
           4 heads x 16 contiguous channels [16*i, 16*i+16), softmax over keys, concat heads, o_proj.
        h = LN1(h + a)                                       (`ln1` = transformer.layer_norm, eps 1e-6)
        m = FFN(h):  r = softmax(moe.gate . h) over 4 experts (`moe.gate.weight` ne [64,4], no bias),
                     e = NewGELU(moe.experts . h) -> 256 values, expert i = rows [64*i, 64*i+64)
                     (`moe.experts.weight` ne [64,256], no bias), moe = sum_i r_i * e_i ([64]),
                     m = ffn_out(NewGELU(moe)) (`ffn_out.{weight,bias}` ne [64,64]; NewGELU applied AGAIN,
                     FeedForward.intermediate_act_fn)
        h = LN2(h + m)                                       (`ln2` = transformer.final_layer_norm, eps 1e-6)
        NewGELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
   7. e = eproj(h) + x2: `eproj.weight` ne [64, 2560] (torch Conv1d(64,2560,1) squeezed) + `eproj.bias`; e [T', 2560].
   8. e = nearest-upsample(e, scale s_t) along time, cropped to T (the block input length); then e = e + flatten(x_in)
      where flatten(x_in) is x_in [T, F_in, C_in] flattened with k = f*C_in + c (same rule as step 3 applied to the
      block INPUT, a pure reshape of the 2560 = F_in*C_in values) -> [T, 2560]. Block output = reshape e back to
      [T, F', C'] with k = f*C' + c (torch reshape (b,f,c,T).permute(0,2,1,3)), i.e. the next block's input is
      [T, F', C'] using the reduce output shape (F', C'); for stage 4 (no reduce) F'*C' = (F_in//s_f) * (C_in*s_f).
  After each STAGE S, feat_S = flatten(x) [T, 2560] (k = f*C + c) is recorded (dump names feat_1..feat_4).
  Per-stage geometry (F, C) of the stage OUTPUT: head (80, 32); stage 1 (80, 32); stage 2 (40, 64); stage 3
  (20, 128); stage 4 (10, 256). Inside a block with s_t=2 the cnn/transformer run at T' = ceil(T/2) but the block
  output is upsampled and cropped back to its input length T, so T is the fbank frame count everywhere.

Dense fusion + pooling (VPR / ASTP):
  x = sum_S w_S (.) feat_S,  w = `fe.dense_w` ggml ne [2560, 4] (softmax over stages precomputed; row S contiguous),
      x [T, 2560].
  ASTP: u = tanh(pool.l1 . x + b1)   (`pool.l1.weight` ne [2560,128], bias 128)   [T,128]
        s = pool.l2 . u + b2         (`pool.l2.weight` ne [128,2560], bias 2560)   [T,2560]
        alpha = softmax over time per channel; mean = sum_t alpha x; var = sum_t alpha x^2 - mean^2;
        std = sqrt(max(var, 1e-7));   pooled = concat(mean, std) [5120]
  embedding = linear(BN(pooled)): `bn.{scale,shift}` (5120), `linear.weight` ne [5120,192] (no bias). NOT normalized.

Tensor names (all F32). BN pair `X.{scale,shift}`: scale = w/sqrt(var+1e-5), shift = b - mean*scale (float64).
  fe.head.conv.{weight,bias}, fe.head.bn.{scale,shift}
  fe.b{S}.{K}.cnn.{R}.conv1.w{j}, conv2.w{j}, pw1.{weight,bias}, pw2.{weight,bias}, bn1.*, bn2.*, sc.weight, sc.bn.*
  fe.b{S}.{K}.reduce.{weight,bias}, reduce.bn.*   (expansion>1 only)
  fe.b{S}.{K}.rproj.{weight,bias}, rproj.ln.{weight,bias}
  fe.b{S}.{K}.pos.dw.{weight,bias}, pos.bn.*, pos.pw.{weight,bias}
  fe.b{S}.{K}.attn.{q,k,v,o}.{weight,bias}, ln1.*, moe.gate.weight, moe.experts.weight, ffn_out.*, ln2.*, eproj.*
  fe.dense_w, pool.l1.*, pool.l2.*, bn.{scale,shift}, linear.weight
  frontend.window [400] (torch.hamming_window(400, periodic=False, alpha=.54, beta=.46)),
  frontend.mel_banks ne [257,80], frontend.resample_44100 ne [K,160] (torchaudio Resample(44100,16000).kernel[:,0,:])."""

SPK_ARCH = "mect_spk"
ENCODER = "ownmect_s2"
PREFIX = "model."
BN_EPS = 1e-5
EXPECTED_BACKBONE_CONFIG = {
    "moe_type": "token",
    "num_experts_shard": 2,
    "num_experts_token": 4,
    "top_k_shard": None,
    "top_k_token": None,
    "shard_scale": 1.0,
}
EXPECTED_FBANK = {
    "num_mel_bins": 80,
    "frame_length": 25,
    "frame_shift": 10,
    "dither": 0.0,
    "energy_floor": 0.0,
    "window_type": "hamming",
    "scale": 32768,
}
HIDDEN = 64
POS_K = 113
N_EXPERTS = 4
FEAT = 2560
IN_C, IN_F = 32, 80
SPK_SOURCE_TENSORS = 1253

# MECT_B2 (backbones/mect.py): per block (cnn_num, out_channels, stride (s_f, s_t), expansion).
STAGES = [
    [(3, 32, (1, 1), 4)] * 2,
    [(3, 64, (2, 1), 2)] + [(3, 64, (1, 2), 2)] * 3,
    [(3, 128, (2, 1), 2)] + [(3, 128, (1, 2), 2)] * 4,
    [(3, 256, (2, 1), 1)] + [(2, 256, (1, 2), 1)] * 2,
]
# Stage 1 blocks use an int stride 1 in torch; stages 2-4 use tuples. A tuple stride always creates a shortcut.
STAGE_STRIDE_IS_TUPLE = [False, True, True, True]


def fail(message):
    raise ValueError(message)


class Mapper:
    """Builds the (dst, src, expected_shape, transform) list; consumed source keys are tracked."""

    def __init__(self):
        self.items = []  # (dst, src, shape, fn)
        self.bn_items = []  # (dst_prefix, src_prefix, channels)
        self.src_keys = []

    def tensor(self, dst, src, shape, fn=None):
        self.items.append((dst, src, tuple(shape), fn))
        self.src_keys.append(src)

    def affine(self, dst, src, dim):
        self.tensor(dst + ".weight", src + ".weight", (dim,))
        self.tensor(dst + ".bias", src + ".bias", (dim,))

    def linear(self, dst, src, i, o, bias=True):
        self.tensor(dst + ".weight", src + ".weight", (o, i))
        if bias:
            self.tensor(dst + ".bias", src + ".bias", (o,))

    def conv1x1(self, dst, src, i, o, bias=True):
        self.tensor(dst + ".weight", src + ".weight", (o, i, 1, 1))
        if bias:
            self.tensor(dst + ".bias", src + ".bias", (o,))

    def conv1d1(self, dst, src, i, o):
        self.tensor(dst + ".weight", src + ".weight", (o, i, 1), lambda t: t.squeeze(-1))
        self.tensor(dst + ".bias", src + ".bias", (o,))

    def bn(self, dst, src, dim):
        self.bn_items.append((dst, src, dim))
        for s in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            self.src_keys.append(f"{src}.{s}")


def build_spk_mapping():
    """Expected checkpoint schema derived from the MECT_B2 definition (mect.py), independent of the ckpt."""
    m = Mapper()
    fe = PREFIX + "feature_extractor."
    m.tensor("fe.head.conv.weight", fe + "head.0.weight", (IN_C, 1, 3, 3))
    m.tensor("fe.head.conv.bias", fe + "head.0.bias", (IN_C,))
    m.bn("fe.head.bn", fe + "head.1", IN_C)
    c_in, f_in = IN_C, IN_F
    for si, stage in enumerate(STAGES):
        S = si + 1
        for K, (depth, out_ch, (s_f, s_t), expansion) in enumerate(stage):
            src = f"{fe}mect_block{S}.{K}."
            dst = f"fe.b{S}.{K}."
            oc = out_ch * expansion
            ic = c_in
            conv_groups = ic  # group_divisor 1: conv_groups = in_channels of the block's first res block
            for R in range(depth):
                cs = f"{src}cnn_block.{R}."
                cd = f"{dst}cnn.{R}."
                ic_r = ic if R == 0 else oc
                groups = conv_groups if R == 0 else oc  # res block R>=1 builds its own groups = its in_channels
                a1 = ic_r // groups
                a2 = oc // groups
                for j in range(a1):
                    m.tensor(f"{cd}conv1.w{j}", cs + "conv1.weight", (oc, a1, 3, 3), lambda t, j=j: t[:, j : j + 1])
                m.conv1x1(cd + "pw1", cs + "conv_pw1", oc, oc)
                m.bn(cd + "bn1", cs + "bn1", oc)
                for j in range(a2):
                    m.tensor(f"{cd}conv2.w{j}", cs + "conv2.weight", (oc, a2, 3, 3), lambda t, j=j: t[:, j : j + 1])
                m.conv1x1(cd + "pw2", cs + "conv_pw2", oc, oc)
                m.bn(cd + "bn2", cs + "bn2", oc)
                stride_tuple = R == 0 and STAGE_STRIDE_IS_TUPLE[si]
                if R == 0 and (stride_tuple or ic_r != oc):
                    m.conv1x1(cd + "sc", cs + "shortcut.0", ic_r, oc, bias=False)
                    m.bn(cd + "sc.bn", cs + "shortcut.1", oc)
            c_out_r, f_out_r = c_in * s_f, f_in // s_f
            if expansion > 1:
                m.conv1x1(dst + "reduce", src + "reduce_conv.0", oc, c_out_r)
                m.bn(dst + "reduce.bn", src + "reduce_conv.1", c_out_r)
            elif oc != c_out_r:
                fail(f"internal: block {S}.{K} identity reduce needs OC == C_in*s_f ({oc} vs {c_out_r})")
            fcdims = f_in * out_ch // s_f
            if fcdims != FEAT or f_out_r * c_out_r != FEAT:
                fail(f"internal: block {S}.{K} flattened dim {fcdims}/{f_out_r * c_out_r} != {FEAT}")
            m.tensor(dst + "rproj.weight", src + "reduce_project.0.weight", (HIDDEN, FEAT, 1), lambda t: t.squeeze(-1))
            m.tensor(dst + "rproj.bias", src + "reduce_project.0.bias", (HIDDEN,))
            m.affine(dst + "rproj.ln", src + "reduce_project.1", HIDDEN)
            m.tensor(
                dst + "pos.dw.weight", src + "pos_conv.pos_conv.0.weight", (HIDDEN, 1, POS_K), lambda t: t.squeeze(1)
            )
            m.tensor(dst + "pos.dw.bias", src + "pos_conv.pos_conv.0.bias", (HIDDEN,))
            m.bn(dst + "pos.bn", src + "pos_conv.pos_conv.1", HIDDEN)
            m.conv1d1(dst + "pos.pw", src + "pos_conv.pos_conv.3", HIDDEN, HIDDEN)
            tr = src + "transformer."
            for a, b in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "out_proj")):
                m.linear(f"{dst}attn.{a}", f"{tr}attention.{b}", HIDDEN, HIDDEN)
            m.affine(dst + "ln1", tr + "layer_norm", HIDDEN)
            moe = tr + "feed_forward.intermediate_dense.token_moe."
            m.tensor(dst + "moe.gate.weight", moe + "gate.weight", (N_EXPERTS, HIDDEN))
            m.tensor(dst + "moe.experts.weight", moe + "experts.weight", (HIDDEN * N_EXPERTS, HIDDEN))
            m.linear(dst + "ffn_out", tr + "feed_forward.output_dense", HIDDEN, HIDDEN)
            m.affine(dst + "ln2", tr + "final_layer_norm", HIDDEN)
            m.conv1d1(dst + "eproj", src + "expansion_project.0", HIDDEN, FEAT)
            c_in, f_in = c_out_r, f_out_r
    m.tensor("fe.dense_w", fe + "dense3.inputs_weights", (1, 4, FEAT, 1), lambda t: torch.softmax(t, dim=1)[0, :, :, 0])
    ep = PREFIX + "encoding_layers."
    m.tensor("pool.l1.weight", ep + "linear1.weight", (128, FEAT, 1), lambda t: t.squeeze(-1))
    m.tensor("pool.l1.bias", ep + "linear1.bias", (128,))
    m.tensor("pool.l2.weight", ep + "linear2.weight", (FEAT, 128, 1), lambda t: t.squeeze(-1))
    m.tensor("pool.l2.bias", ep + "linear2.bias", (FEAT,))
    m.bn("bn", PREFIX + "bn", 2 * FEAT)
    m.tensor("linear.weight", PREFIX + "linear.weight", (192, 2 * FEAT))
    return m


def validate_spk_registry(entry):
    expect = {
        "type": "mect_fbank",
        "backbone_type": "MECT_B2",
        "backbone_config": EXPECTED_BACKBONE_CONFIG,
        "pooling": "ASTP",
        "embed_dim": 192,
        "sample_rate": 16000,
        "max_seconds": 60,
        "fbank": EXPECTED_FBANK,
    }
    for key, value in expect.items():
        if entry.get(key) != value:
            fail(f"speaker encoder registry {key}: {entry.get(key)!r}, expected {value!r}")
    if not entry.get("sha256"):
        fail("speaker encoder registry entry has no sha256")


def cmd_speaker_encoder(args, parser):
    import torchaudio
    from zonos3.models.mect_speaker import MECTFbankSpeakerEmbedding, registry_path

    entry = json.loads((args.registry or registry_path()).read_text())[ENCODER]
    validate_spk_registry(entry)
    # strict load of the sha256-checked checkpoint
    module = MECTFbankSpeakerEmbedding({**entry, "codec": ENCODER, "checkpoint": str(args.checkpoint)}, device="cpu")
    sd = module.state_dict()

    m = build_spk_mapping()
    expected_keys = set(m.src_keys)
    keys = set(sd)
    if keys != expected_keys:
        fail(
            f"speaker encoder schema mismatch: missing={sorted(expected_keys - keys)[:8]}, "
            f"unknown={sorted(keys - expected_keys)[:8]}"
        )
    for dst, src, shape, fn in m.items:
        if tuple(sd[src].shape) != shape:
            fail(f"wrong shape for {src}: {tuple(sd[src].shape)}, expected {shape}")
        if sd[src].dtype != torch.float32:
            fail(f"{src}: dtype {sd[src].dtype}, expected float32")
    for dst, src, dim in m.bn_items:
        for s in ("weight", "bias", "running_mean", "running_var"):
            if tuple(sd[f"{src}.{s}"].shape) != (dim,):
                fail(f"wrong shape for {src}.{s}: {tuple(sd[f'{src}.{s}'].shape)}, expected ({dim},)")
        if sd[f"{src}.num_batches_tracked"].dtype != torch.int64:
            fail(f"{src}.num_batches_tracked is not int64")
    if len(keys) != SPK_SOURCE_TENSORS:
        fail(f"expected {SPK_SOURCE_TENSORS} checkpoint tensors, found {len(keys)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(args.out), SPK_ARCH)
    w.add_string("general.name", f"ZONOS3 Mini speaker encoder {ENCODER} (MECT-B2 + ASTP, 192-d)")
    for key, value in {
        "sample_rate": 16000,
        "max_seconds": 60,
        "embedding_length": 192,
        "n_mels": 80,
        "frame_length": 400,
        "frame_shift": 160,
        "n_fft": 512,
        "n_heads": 4,
        "n_experts": N_EXPERTS,
        "pos_kernel": POS_K,
    }.items():
        w.add_uint32(f"{SPK_ARCH}.{key}", value)
    for key, value in {
        "preemphasis": 0.97,
        "waveform_scale": 32768.0,
        "log_floor": 1.1920928955078125e-07,
        "ln_eps": 1e-6,
        "pool_var_floor": 1e-7,
    }.items():
        w.add_float32(f"{SPK_ARCH}.{key}", value)
    w.add_string(f"{SPK_ARCH}.backbone", "MECT_B2")

    n_out = 0

    def put(name, value):
        nonlocal n_out
        value = np.ascontiguousarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            fail(f"nonfinite tensor: {name}")
        w.add_tensor(name, value)
        n_out += 1

    # Frontend host tables (before the weights so that they are easy to find; order is irrelevant to the loader).
    window = torch.hamming_window(400, periodic=False, alpha=0.54, beta=0.46)
    mel, _ = torchaudio.compliance.kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
    mel = torch.nn.functional.pad(mel, (0, 1), mode="constant", value=0.0)
    assert tuple(mel.shape) == (80, 257) and tuple(window.shape) == (400,)
    put("frontend.window", window.numpy())
    put("frontend.mel_banks", mel.numpy())
    resampler = torchaudio.transforms.Resample(44100, 16000)
    put("frontend.resample_44100", resampler.kernel[:, 0, :].contiguous().numpy())

    for dst, src, shape, fn in m.items:
        t = sd[src]
        t = fn(t) if fn is not None else t
        put(dst, t.contiguous().numpy())
    for dst, src, dim in m.bn_items:
        weight, bias = sd[f"{src}.weight"].double(), sd[f"{src}.bias"].double()
        mean, var = sd[f"{src}.running_mean"].double(), sd[f"{src}.running_var"].double()
        scale = weight / torch.sqrt(var + BN_EPS)
        shift = bias - mean * scale
        put(dst + ".scale", scale.float().numpy())
        put(dst + ".shift", shift.float().numpy())

    w.add_uint32(f"{SPK_ARCH}.source_tensor_count", len(keys))
    write_gguf(w, progress=True)
    print(f"Accounted for {len(keys)} checkpoint tensors; wrote {n_out} GGUF tensors -> {args.out}", flush=True)


# ===================================================================================================== steer

STEER_DOC = """Pack a steering bundle (`studio_bundle_*.json`, zonos3.tts.steering.load_bundle) into one steer.gguf.

The bundle JSON lists direction sets (`{"sets": {name: relative-dir}, "timing": {...}, "ui": {...},
"rate_extension": {...}}`); each set directory holds a `manifest.json`, one `.npy` per direction and optionally a
`calibration.json`. Raw sets move the 192-d speaker embedding; `attnres_source` sets add to one Full AttnRes bank
source of the decoder.

GGUF layout (read by servers/ggml/src/zonos3-steer.cpp):
  general.architecture = "zonos3_steer", general.name
  zonos3_steer.name                    bundle name
  zonos3_steer.speaker_dim             u32, raw-set dim (0 without raw sets)
  zonos3_steer.hidden_size             u32, hidden-set dim (0 without hidden sets)
  zonos3_steer.set.{names,spaces}      [str] bundle order; "raw" | "attnres_source"
  zonos3_steer.set.sources             [i32] bank source index, -1 for raw sets
  zonos3_steer.set.calibrations        [str] per set, its calibration.json text, or "" without one
  zonos3_steer.direction.{ids,kinds}   [str] Python order (per set: named, then axes); "named" | "axis"
  zonos3_steer.timing                  str, JSON {direction id: [[dose, duration factor], ...]}
  zonos3_steer.ui                      str, JSON (key order preserved)
  zonos3_steer.rate_extension          str, JSON of the bundle's rate_extension block, or "" without one
  steer.dir.<i>                        F32 [dim], vector of direction i"""

STEER_ARCH = "zonos3_steer"


def cmd_steer(args, parser):
    from zonos3.tts.steering import HIDDEN_SPACE, RAW_SPACE, load_bundle

    bundle_path = Path(args.bundle)
    spec = json.loads(bundle_path.read_text())
    if not isinstance(spec, dict) or not isinstance(spec.get("sets"), dict) or not spec["sets"]:
        raise SystemExit(f"error: {bundle_path}: expected an object with a non-empty 'sets' mapping")
    if not isinstance(spec.get("ui", {}), dict):
        raise SystemExit(f"error: {bundle_path}: 'ui' must be an object")
    if not isinstance(spec.get("timing", {}), dict):
        raise SystemExit("error: 'timing' must be an object")
    bundle = load_bundle(bundle_path)
    set_names, set_spaces, set_sources, set_calibrations, ids, kinds, vectors, summary = [], [], [], [], [], [], [], []
    hidden_dims = set()
    for set_name, directions in bundle.sets.items():
        set_names.append(set_name)
        set_spaces.append(directions.space)
        set_sources.append(int(directions.source_index) if directions.space == HIDDEN_SPACE else -1)
        calibration = (bundle_path.parent / spec["sets"][set_name]).resolve() / "calibration.json"
        set_calibrations.append(calibration.read_text() if calibration.is_file() else "")
        if directions.space == HIDDEN_SPACE:
            hidden_dims.add(int(directions.dim))
        for kind, group in (("named", directions.named), ("axis", directions.axes)):
            for name, vec in group.items():
                vec = np.ascontiguousarray(vec.numpy(), dtype=np.float32)
                if not np.all(np.isfinite(vec)):
                    raise SystemExit(f"error: direction '{set_name}.{name}' has non-finite values")
                ids.append(f"{set_name}.{name}")
                kinds.append(kind)
                vectors.append(vec)
        summary.append((set_name, directions.space, set_sources[-1], len(directions.named) + len(directions.axes)))
    if len(hidden_dims) > 1:
        raise SystemExit(f"error: hidden sets disagree on the hidden size: {sorted(hidden_dims)}")
    for did, table in bundle.timing.items():
        ok = (
            isinstance(table, list)
            and len(table) > 0
            and all(
                isinstance(p, (list, tuple))
                and len(p) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p)
                for p in table
            )
        )
        if not ok:
            raise SystemExit(f"error: timing['{did}'] must be a non-empty list of finite [dose, factor] pairs")

    w = GGUFWriter(str(args.out), STEER_ARCH)
    w.add_name(f"zonos3 steering bundle {bundle.name}")
    w.add_string(f"{STEER_ARCH}.name", str(bundle.name))
    w.add_uint32(f"{STEER_ARCH}.speaker_dim", bundle.raw_dim or 0)
    w.add_uint32(f"{STEER_ARCH}.hidden_size", next(iter(hidden_dims), 0))
    w.add_array(f"{STEER_ARCH}.set.names", set_names)
    w.add_array(f"{STEER_ARCH}.set.spaces", set_spaces)
    w.add_array(f"{STEER_ARCH}.set.sources", set_sources)
    w.add_array(f"{STEER_ARCH}.set.calibrations", set_calibrations)
    w.add_array(f"{STEER_ARCH}.direction.ids", ids)
    w.add_array(f"{STEER_ARCH}.direction.kinds", kinds)
    w.add_string(f"{STEER_ARCH}.timing", json.dumps(bundle.timing))
    w.add_string(f"{STEER_ARCH}.ui", json.dumps(bundle.ui, ensure_ascii=False))
    w.add_string(
        f"{STEER_ARCH}.rate_extension", json.dumps(spec["rate_extension"]) if spec.get("rate_extension") else ""
    )
    for i, vec in enumerate(vectors):
        w.add_tensor(f"steer.dir.{i}", vec)
    write_gguf(w)
    for set_name, space, source, n in summary:
        print(f"{set_name} {space} {source} {n}")
    n_raw = sum(n for _, sp, _, n in summary if sp == RAW_SPACE)
    print(f"wrote {args.out}: {len(set_names)} sets, {len(ids)} directions ({n_raw} raw, {len(ids) - n_raw} hidden)")


# ===================================================================================================== accents

ACCENTS_DOC = """Pack the accent adapters of a directory (zonos3.models.accent_format: list_accent_adapters order,
zonos-accent-lora/1, /2 and /3) into a zonos3_accents GGUF (accents.gguf beside the backbone).

  zonos3_accents.names    [str] adapter names; slot s = index in this list
  zonos3_accents.format   [str] adapter format per slot
  zonos3_accents.rank     [u32] AccentAdapter.rank per slot
A linear module M maps to its GGUF base tensor B = LM_NAMES name of M.weight minus ".weight":
  acc.<s>.<B>.lora_down   F32 numpy [K, in]
  acc.<s>.<B>.lora_up     F32 numpy [out, K]      (shift = up @ down)
  Split modules share lora_down and split lora_up rows: attention.wkv -> attn_k (first half) / attn_v,
  feed_forward.w_in -> ffn_up (first half) / ffn_gate.
Expert factors (/2), SonicExperts layout de-interleaved:
  acc.<s>.blk.<L>.ffn_gate_exps.lora_up = w13 up[:, 0::2, :], ffn_up_exps.lora_up = w13 up[:, 1::2, :]  [E, H, r];
  both with lora_down = w13 down [E, r, in]; acc.<s>.blk.<L>.ffn_down_exps.lora_{down,up} = w2 down [E, r, H] /
  up [E, out, r].
  acc.<s>.blk.<L>.router_bias  F32 [E], added to the router's balancing biases.
Modules an adapter does not shift are absent.
A zonos-accent-lora/3 adapter (at most one: accent_format.edit_adapter) is the speech-edit adapter: its slot is written
like a /2 one with format "zonos-accent-lora/3"; the servers run edit fills on it at strength 1 and do not offer it as an
accent. Its vocabulary rows are not stored here: export the backbone with `lm --edit-adapter <the same .pt>`."""

ACC_ARCH = "zonos3_accents"


def _base(name: str) -> str:
    return name.removesuffix(".weight")


def accent_tensors(slot: int, adapter):
    """(GGUF name, float32 array) of one adapter."""
    out = []
    for module, (down, up_t) in adapter.linear.items():
        _, _, _, names, transform = lm_entry(module + ".weight")
        up = up_t.T.contiguous()  # [out, K]
        if len(names) == 1:
            if transform is not None:
                raise ValueError(f"accent module {module}: its GGUF tensor is transformed")
            ups = (up,)
        else:
            if up.shape[0] % len(names):
                raise ValueError(f"accent module {module}: {up.shape[0]} rows do not split into {names}")
            ups = up.chunk(len(names), 0)
        for name, part in zip(names, ups):
            out.append((f"acc.{slot}.{_base(name)}.lora_down", down))
            out.append((f"acc.{slot}.{_base(name)}.lora_up", part))
    for module, weights in adapter.experts.items():
        for weight, (up, down) in weights.items():
            param = module + (".gate_up_proj" if weight == "w13" else ".down_proj")
            _, _, _, names, _ = lm_entry(param)
            ups = (up[:, 0::2, :], up[:, 1::2, :]) if weight == "w13" else (up,)
            for name, part in zip(names, ups, strict=True):
                out.append((f"acc.{slot}.{_base(name)}.lora_down", down))
                out.append((f"acc.{slot}.{_base(name)}.lora_up", part))
    for router, bias in adapter.router_bias.items():
        _, _, _, names, _ = lm_entry(router + ".balancing_biases")
        out.append((f"acc.{slot}.{names[0]}", bias))
    return [(name, np.ascontiguousarray(t.float().numpy())) for name, t in out]


def cmd_accents(args, parser):
    from zonos3.models.accent_format import edit_adapter, list_accent_adapters, load_adapter

    directory = Path(args.directory)
    edit_adapter(directory)  # at most one speech-edit (/3) adapter
    adapters = [load_adapter(directory / f"{name}.pt") for name in list_accent_adapters(directory)]
    w = GGUFWriter(str(args.out), ACC_ARCH)
    w.add_name(f"zonos3 accent adapters {directory.name}")
    w.add_array(f"{ACC_ARCH}.names", [a.name for a in adapters])
    w.add_array(f"{ACC_ARCH}.format", [a.format for a in adapters])
    w.add_array(f"{ACC_ARCH}.rank", [int(a.rank) for a in adapters])
    count = 0
    for slot, adapter in enumerate(adapters):
        for name, value in accent_tensors(slot, adapter):
            if not np.isfinite(value).all():
                raise SystemExit(f"error: {name} has non-finite values")
            w.add_tensor(name, value)
            count += 1
    write_gguf(w)
    print(f"wrote {args.out}: {len(adapters)} adapters, {count} tensors")


# ===================================================================================================== CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, doc, fn):
        p = sub.add_parser(
            name, help=doc.split("\n")[0], description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        p.set_defaults(fn=fn)
        return p

    p = command("lm", LM_DOC, cmd_lm)
    p.add_argument(
        "lm_dir", help="release directory lm/mect_60k, a derived checkpoint, or a QAT-baked directory (bake.json)"
    )
    p.add_argument(
        "--silence-frame",
        help="silence-frame registry JSON (default <lm_dir>/../../registries/silence_frame_<audio_codec>.json)",
    )
    p.add_argument(
        "--edit-adapter",
        help="a zonos-accent-lora/3 speech-edit adapter: the backbone gets its vocabulary "
        "(span_end_id and the grown rows)",
    )
    p.add_argument(
        "--phoneme-rows",
        help="zonos-text-extra/1 phoneme tokens appended to the text table (after "
        "--edit-adapter's rows): pronunciation groups",
    )
    p.add_argument("-o", "--out", required=True)
    p.add_argument(
        "--outtype",
        choices=("f32", "f16", "q8_0", MIX_OUTTYPE),
        help=f"default f16 (plain directory) or {MIX_OUTTYPE} (QAT directory; the only valid type for it). "
        f"{MIX_OUTTYPE} on a plain directory quantizes the same tensors post-training (round to nearest)",
    )

    p = command("codec-decoder", CODEC_DECODER_DOC, cmd_codec_decoder)
    p.add_argument("generator", help="autoencoder/generator.pt")
    p.add_argument("--head", required=True, help="autoencoder/heads/full_mr or autoencoder/heads/keep2_plain")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--outtype", choices=["f16", "f32"], default="f16")

    p = command("codec-encoder", CODEC_ENCODER_DOC, cmd_codec_encoder)
    p.add_argument("generator", help="autoencoder/generator.pt")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--outtype", choices=["f32", "f16"], default="f32")

    p = command("ssl-teacher", SSL_DOC, cmd_ssl_teacher)
    p.add_argument("checkpoint", type=Path, help="teacher directory or its model.safetensors")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--outtype", choices=("bf16", "f16", "f32"), default="f32")

    p = command("speaker-encoder", SPK_DOC, cmd_speaker_encoder)
    p.add_argument("checkpoint", type=Path, help=f"speaker_encoders/{ENCODER}/ckpt_0037.pth")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="speaker_encoder_registry.json (default zonos3.models.mect_speaker.registry_path())",
    )

    p = command("steer", STEER_DOC, cmd_steer)
    p.add_argument("bundle", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)

    p = command("accents", ACCENTS_DOC, cmd_accents)
    p.add_argument("directory", type=Path, help="accent adapter directory (<accent>.pt files)")
    p.add_argument("-o", "--out", type=Path, required=True)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sub = parser._subparsers._group_actions[0].choices[args.command]
    args.fn(args, sub)


if __name__ == "__main__":
    sys.exit(main())
