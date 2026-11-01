"""tools/convert_gguf.py on tiny random inputs: lm (Zonos3LM -> GGUF through LM_NAMES), directions, accents."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from gguf import GGUFReader

import convert_gguf
from zonos3.reference.lm import Zonos3LM, Zonos3LMConfig

CODEC = "tiny_codec"


def tiny_params(span_end: bool = False) -> dict:
    params = {
        "attention_pattern": "gdn3_global1", "residual_type": "full_attnres", "use_xsa": False,
        "speaker_enabled": True, "condition_tokens_enabled": True, "moe_router_type": "linear",
        "moe_router_use_eda": False, "moe_balancing_strategy": "quantile", "moe_router_topk": 2,
        "moe_start_from_layer": 1, "moe_end_from_layer": 0,
        "n_layers": 4, "dim": 64, "head_dim": 8, "n_heads": 4, "n_kv_heads": 2,
        "n_codebooks": 10, "codebook_size": 32, "eoa_id": 32, "audio_pad_id": 33,
        "text_vocab": 20, "text_extra_tokens": 2,
        "text_tokenizer_spec": {"kind": "superbpe", "path": "tokenizer.json", "vocab_size": 16,
                                "offset": 4, "bos": 2, "eos": 3},
        "moe_alpha": 0.25, "moe_n_experts": 4, "multiple_of": 32, "ffn_dim_multiplier": 1.0,
        "condition_count": 3, "condition_n_buckets": 4,
        "condition_names": ["loudness", "fmax", "speaking_rate"],
        "condition_edges": {"loudness": [-30.0, -20.0, -10.0], "fmax": [4000.0, 8000.0, 12000.0],
                            "speaking_rate": [10.0, 14.0, 18.0]},
        "condition_units": {"loudness": "LUFS", "fmax": "Hz", "speaking_rate": "phonemes/s"},
        "norm_eps": 1e-5, "rope_theta": 10000.0, "max_seqlen": 256, "loss_softcap": 30.0,
        "speaker_embedding_dim": 8, "speaker_reference_max_seconds": 60.0, "audio_codec": CODEC,
    }
    if span_end:
        params["span_end_id"] = 34
    return params


def make_tiny_lm_dir(path: Path, span_end: bool = False, seed: int = 0) -> tuple[Path, Path]:
    """A random tiny LM checkpoint directory and its silence-frame JSON."""
    path.mkdir(parents=True, exist_ok=True)
    params = tiny_params(span_end)
    model = Zonos3LM(Zonos3LMConfig.from_params(params))
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for name, tensor in model.state_dict().items():
        value = torch.randn(tensor.shape, generator=generator) * 0.05
        state[name] = value if name.endswith("balancing_biases") else value.to(torch.bfloat16)
    torch.save(state, path / "model.pth")
    (path / "params.json").write_text(json.dumps({"model": params}))
    (path / "tokenizer.json").write_text('{"model": {"type": "BPE"}}')
    silence = path / "silence.json"
    silence.write_text(json.dumps({"codec": CODEC, "frame": list(range(10))}))
    return path, silence


@pytest.mark.parametrize("span_end", [False, True])
def test_lm_round_trip(tmp_path, span_end):
    lm_dir, silence = make_tiny_lm_dir(tmp_path / "lm", span_end)
    out = tmp_path / "lm.gguf"
    convert_gguf.main(["lm", str(lm_dir), "--outtype", "f32", "--silence-frame", str(silence), "-o", str(out)])

    model = Zonos3LM.load(lm_dir)
    expected = list(convert_gguf.mapped_tensors(model))
    reader = GGUFReader(out)
    assert [t.name for t in reader.tensors] == [name for name, _ in expected]
    for tensor, (name, value) in zip(reader.tensors, expected):
        assert tuple(int(n) for n in tensor.shape)[::-1] == tuple(value.shape), name
        np.testing.assert_array_equal(np.asarray(tensor.data).reshape(value.shape), value.float().numpy(), err_msg=name)

    # every LM_NAMES entry is exercised (the tiny model has GDN, global, dense and MoE layers)
    hit = {convert_gguf.lm_entry(name, model.config.n_codebooks)[0] for name in model.state_dict()}
    assert hit == set(range(len(convert_gguf.LM_NAMES)))
    names = [t.name for t in reader.tensors]
    assert names[:12] == [f"audio_embd.{i}.weight" for i in range(10)] + ["text_embd.weight", "output_norm.weight"]
    assert ["blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight", "blk.3.attn_temp"] == \
        names[names.index("blk.3.attn_q.weight"):names.index("blk.3.attn_q.weight") + 4]

    fields = reader.fields

    def scalar(key):
        field = fields[key]
        return field.parts[field.data[0]][0]

    def string(key):
        field = fields[key]
        return bytes(field.parts[field.data[0]]).decode()

    assert string("general.architecture") == "zonos3"
    assert scalar("zonos3.block_count") == 4
    assert scalar("zonos3.text_vocab") == 20
    assert scalar("zonos3.audio_vocab") == (35 if span_end else 34)
    assert json.loads(string("zonos3.text_tokenizer_spec")) == tiny_params()["text_tokenizer_spec"]
    assert ("zonos3.span_end_id" in fields) == span_end
    if span_end:
        assert scalar("zonos3.span_end_id") == 34
    text = reader.get_tensor(names.index("text_embd.weight"))
    assert tuple(int(n) for n in text.shape)[::-1] == (20 + 1 + 2, 64)


def release_shapes() -> dict:
    """The release lm/mect_60k flat state dict (177 tensors)."""
    shapes = {f"multi_embedder.embedders.{i}.weight": (4098, 1024) for i in range(10)}
    shapes.update({"multi_embedder.embedders.10.weight": (8197, 1024), "multi_output.weight": (40980, 1024),
                   "out_norm.weight": (1024,), "block_attnres_queries": (23, 1024),
                   "speaker_projection.weight": (1024, 192), "speaker_projection.bias": (1024,)})
    shapes.update({f"condition_embeddings.{i}.weight": (11, 1024) for i in range(3)})
    for layer in range(11):
        p = f"layers.{layer}."
        shapes[p + "attention_norm.weight"] = shapes[p + "ffn_norm.weight"] = (1024,)
        if (layer + 1) % 4:
            attention = {"in_proj.weight": (2056, 1024), "conv.weight": (1536, 1, 4), "out_weight": (128,),
                         "A_log": (4,), "dt_bias": (4,), "wo.weight": (1024, 512)}
        else:
            attention = {"wq.weight": (512, 1024), "wkv.weight": (2, 128, 1024), "wo.weight": (1024, 512),
                         "temp": (1, 16, 1), "gater.weight": (16, 1024)}
        shapes.update({p + "attention." + k: v for k, v in attention.items()})
        if layer == 0:
            ffn = {"w_in.weight": (2, 2048, 1024), "w_out.weight": (1024, 2048)}
        else:
            ffn = {"experts.gate_up_proj": (64, 3072, 256), "experts.down_proj": (64, 256, 1536),
                   "latent_down.weight": (256, 1024), "latent_up.weight": (1024, 256),
                   "router.router_linear.weight": (64, 1024), "router.rmsnorm_eda.weight": (1024,),
                   "router.balancing_biases": (64,)}
        shapes.update({p + "feed_forward." + k: v for k, v in ffn.items()})
    return shapes


def test_release_schema_maps_to_the_gguf_tensors():
    params = dict(tiny_params(), n_layers=11, dim=1024, head_dim=32, n_heads=16, n_kv_heads=4, max_seqlen=32768,
                  codebook_size=4096, eoa_id=4096, audio_pad_id=4097, text_vocab=8196, text_extra_tokens=0,
                  speaker_embedding_dim=192, moe_n_experts=64, ffn_dim_multiplier=2.0, multiple_of=64,
                  moe_ffn_dim_multiplier=1.5, moe_multiple_of=64, condition_n_buckets=10, condition_edges={},
                  text_tokenizer_spec={"kind": "superbpe", "vocab_size": 8192, "offset": 4, "bos": 2, "eos": 3})
    with torch.device("meta"):
        model = Zonos3LM(Zonos3LMConfig.from_params(params))
    assert {name: tuple(t.shape) for name, t in model.state_dict().items()} == release_shapes()
    names = [name for name, _ in convert_gguf.mapped_tensors(model)]
    assert len(names) == len(set(names)) == 190
    assert names[-8:] == [f"blk.10.{n}" for n in ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                                                   "ffn_down_exps.weight", "ffn_latent_down.weight",
                                                   "ffn_latent_up.weight", "router.weight", "router_norm.weight",
                                                   "router_bias")]


@pytest.mark.parametrize("nonverbal", [False, True])
def test_lm_tag_cfg_trained(tmp_path, nonverbal):
    lm_dir, silence = make_tiny_lm_dir(tmp_path / "lm")
    flag = "true" if nonverbal else "false"
    (lm_dir / "config.yaml").write_text(
        f"model: {{}}\ndata:\n  codec: {CODEC}\n  text_tokenizer: superbpe\n  text_tokenizer_vocab_size: 16\n"
        f"  sources:\n    full: {{weight: 0.55, nonverbal: false}}\n    zv3subsample: {{weight: 0.45, nonverbal: {flag}}}\n")
    out = tmp_path / "lm.gguf"
    convert_gguf.main(["lm", str(lm_dir), "--outtype", "f32", "--silence-frame", str(silence), "-o", str(out)])
    fields = GGUFReader(out).fields
    # written only when true, so checkpoints without nonverbal training keep their GGUF unchanged
    assert ("zonos3.tag_cfg_trained" in fields) == nonverbal
    if nonverbal:
        field = fields["zonos3.tag_cfg_trained"]
        assert bool(field.parts[field.data[0]][0])


def test_lm_rejects_qat_outtype_without_bake(tmp_path):
    lm_dir, silence = make_tiny_lm_dir(tmp_path / "lm")
    with pytest.raises(SystemExit):
        convert_gguf.main(["lm", str(lm_dir), "--outtype", convert_gguf.QAT_OUTTYPE, "--silence-frame", str(silence),
                           "-o", str(tmp_path / "x.gguf")])


def test_directions(tmp_path):
    root = tmp_path / "dirs"
    root.mkdir()
    rng = np.random.default_rng(0)
    vectors = {name: rng.standard_normal(16).astype(np.float32) for name in ("happy", "sad", "valence")}
    for name, vec in vectors.items():
        np.save(root / f"{name}.npy", vec)
    (root / "manifest.json").write_text(json.dumps({
        "dim": 16, "space": "attnres_source", "source_index": 7,
        "directions": {"happy": {"file": "happy.npy", "tier": "strong"}, "sad": {"file": "sad.npy"},
                       "valence": {"file": "valence.npy", "kind": "axis"}}}))
    (root / "calibration.json").write_text('{"default": {"happy": 1.5}}')
    out = tmp_path / "emotion.gguf"
    convert_gguf.main(["directions", str(root), "-o", str(out)])
    reader = GGUFReader(out)
    fields = reader.fields

    def strings(key):
        field = fields[key]
        return [bytes(field.parts[i]).decode() for i in field.data]

    assert strings("zonos3_directions.names") == ["happy", "sad"]
    assert strings("zonos3_directions.axes") == ["valence"]
    assert strings("zonos3_directions.space") == ["attnres_source"]
    assert int(fields["zonos3_directions.source_index"].parts[-1][0]) == 7
    assert json.loads(strings("zonos3_directions.calibration")[0]) == {"default": {"happy": 1.5}}
    assert json.loads(strings("zonos3_directions.tiers")[0]) == {"happy": "strong"}
    assert [t.name for t in reader.tensors] == ["dir.named.0", "dir.named.1", "dir.axis.0"]
    for tensor, name in zip(reader.tensors, ("happy", "sad", "valence")):
        np.testing.assert_array_equal(np.asarray(tensor.data), vectors[name])


def test_accents(tmp_path):
    from zonos3.models.accent_format import FORMAT_EXPERTS

    root = tmp_path / "accents"
    root.mkdir()
    g = torch.Generator().manual_seed(0)
    rank, dim, kv, hidden, experts, latent, moe_hidden = 2, 64, 16, 64, 4, 16, 64
    factors = {
        "layers.3.attention.wkv.weight": {"down": torch.randn(rank, dim, generator=g),
                                          "up": torch.randn(2 * kv, rank, generator=g), "shape": (2 * kv, dim)},
        "layers.0.feed_forward.w_in.weight": {"down": torch.randn(rank, dim, generator=g),
                                              "up": torch.randn(2 * hidden, rank, generator=g),
                                              "shape": (2 * hidden, dim)},
        "multi_output.weight": {"down": torch.randn(rank, dim, generator=g),
                                "up": torch.randn(340, rank, generator=g), "shape": (340, dim)},
        "layers.1.feed_forward.experts.w13": {
            "up": torch.randn(experts, 2 * moe_hidden, rank, generator=g),
            "down": torch.randn(experts, rank, latent, generator=g),
            "shape": (experts, 2 * moe_hidden, latent), "experts": experts},
        "layers.1.feed_forward.experts.w2": {
            "up": torch.randn(experts, latent, rank, generator=g),
            "down": torch.randn(experts, rank, moe_hidden, generator=g),
            "shape": (experts, latent, moe_hidden), "experts": experts},
    }
    bias = torch.randn(experts, generator=g)
    torch.save({"format": FORMAT_EXPERTS, "factors": factors,
                "exact": {"layers.1.feed_forward.router.balancing_biases": bias}}, root / "hin.pt")
    out = tmp_path / "accents.gguf"
    convert_gguf.main(["accents", str(root), "-o", str(out)])
    reader = GGUFReader(out)
    tensors = {t.name: np.asarray(t.data).reshape(tuple(int(n) for n in t.shape)[::-1]) for t in reader.tensors}
    wkv = factors["layers.3.attention.wkv.weight"]
    np.testing.assert_array_equal(tensors["acc.0.blk.3.attn_k.lora_down"], wkv["down"].numpy())
    np.testing.assert_array_equal(tensors["acc.0.blk.3.attn_k.lora_up"], wkv["up"][:kv].numpy())
    np.testing.assert_array_equal(tensors["acc.0.blk.3.attn_v.lora_up"], wkv["up"][kv:].numpy())
    w_in = factors["layers.0.feed_forward.w_in.weight"]
    np.testing.assert_array_equal(tensors["acc.0.blk.0.ffn_up.lora_up"], w_in["up"][:hidden].numpy())
    np.testing.assert_array_equal(tensors["acc.0.blk.0.ffn_gate.lora_up"], w_in["up"][hidden:].numpy())
    assert "acc.0.output.lora_up" in tensors
    w13 = factors["layers.1.feed_forward.experts.w13"]
    np.testing.assert_array_equal(tensors["acc.0.blk.1.ffn_gate_exps.lora_up"], w13["up"][:, 0::2].numpy())
    np.testing.assert_array_equal(tensors["acc.0.blk.1.ffn_up_exps.lora_up"], w13["up"][:, 1::2].numpy())
    np.testing.assert_array_equal(tensors["acc.0.blk.1.ffn_up_exps.lora_down"], w13["down"].numpy())
    w2 = factors["layers.1.feed_forward.experts.w2"]
    np.testing.assert_array_equal(tensors["acc.0.blk.1.ffn_down_exps.lora_up"], w2["up"].numpy())
    np.testing.assert_array_equal(tensors["acc.0.blk.1.router_bias"], bias.numpy())
    names = reader.fields["zonos3_accents.names"]
    assert [bytes(names.parts[i]).decode() for i in names.data] == ["hin"]
