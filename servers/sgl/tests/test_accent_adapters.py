"""Per-request accent adapters: each token gets exactly base + alpha * its adapter."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from convert_lingua_checkpoint import _output_key
from zonos3.models import accent_format
from zonos3.models.accent_format import EXPERT_WEIGHTS, FORMAT, FORMAT_EXPERTS, TARGETS
from zonos3_sgl.distributed import info as tp_info
from zonos3_sgl.layers import accent as accent_ops
from zonos3_sgl.layers.base import OPList
from zonos3_sgl.layers.linear import ChunkedLinear, LinearRowParallel
from zonos3_sgl.models import accent_adapters
from zonos3_sgl.models.accent_adapters import attach_accent_adapters
from zonos3_sgl.models.zonos3 import (
    Attention,
    FeedForward,
    FusedGroupedExperts,
    MoEFeedForward,
    MultiOutputHead,
    Router,
    SimpleLinear,
    Zonos3ForCausalLM,
    _RouterLinear,
)
from zonos3_sgl.scheduler.scheduler import TTSScheduler

HIDDEN, KV, INTER, EXPERTS, VOCAB = 6, 4, 5, 5, 7
# Expert dims of the test model's MoE (multiples of 4: torch._grouped_mm strides).
LATENT, EXPERT_INTER = 4, 4
EXPERT_SHAPES = {"w13": (EXPERTS, 2 * EXPERT_INTER, LATENT), "w2": (EXPERTS, LATENT, EXPERT_INTER)}

# lingua apps.zonos.lora.DEFAULT_TARGETS (bingua2-gdn375k-accent c6a6b9a).
LINGUA_DEFAULT_TARGETS = (
    "attention.wq",
    "attention.wkv",
    "attention.wo",
    "attention.gater",
    "attention.in_proj",
    "feed_forward.w_in",
    "feed_forward.w_out",
    "feed_forward.latent_down",
    "feed_forward.latent_up",
    "router.router_linear",
    "multi_output",
)

# Every module apps.zonos.lora adapts on the gdn375k model, as the training
# tree's apps.zonos.accent_adapter --lora exports them (por_lora64_s1000):
# module suffix -> layers.
GDN375K_LORA_MODULES = {
    "attention.in_proj": [0, 1, 2, 4, 5, 6, 8, 9, 10],
    "attention.wo": list(range(11)),
    "feed_forward.w_in": [0],
    "feed_forward.w_out": [0],
    "feed_forward.latent_down": list(range(1, 11)),
    "feed_forward.latent_up": list(range(1, 11)),
    "feed_forward.router.router_linear": list(range(1, 11)),
    "attention.gater": [3, 7],
    "attention.wkv": [3, 7],
    "attention.wq": [3, 7],
}

# checkpoints/gdn375k/params.json (convert_lingua_checkpoint of the 375k run).
GDN375K_PARAMS = {
    "n_layers": 11,
    "dim": 1024,
    "head_dim": 32,
    "n_heads": 16,
    "n_kv_heads": 4,
    "ffn_dim_multiplier": 2.0,
    "multiple_of": 64,
    "moe_ffn_dim_multiplier": 1.5,
    "moe_multiple_of": 64,
    "norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "max_seqlen": 32768,
    "n_codebooks": 10,
    "codebook_size": 4096,
    "codebook_sizes": [4096] * 10,
    "eoa_id": 4096,
    "audio_pad_id": 4097,
    "text_vocab": 8196,
    "speaker_enabled": False,
    "speaker_embedding_dim": 256,
    "attention_pattern": "gdn3_global1",
    "use_xsa": False,
    "residual_type": "full_attnres",
    "block_attnres_block_size": 4,
    "block_attnres_backend": "auto",
    "moe_router_topk": 2,
    "moe_n_experts": 64,
    "moe_alpha": 0.25,
    "moe_router_dim": 128,
    "moe_router_type": "linear",
    "moe_router_use_eda": False,
    "moe_balancing_strategy": "quantile",
    "moe_start_from_layer": 1,
    "moe_end_from_layer": 0,
    "loss_softcap": 15.0,
    "model_type": "zonos2",
    "dtype": "bfloat16",
    "norm_topk_prob": True,
    "moe_impl": "sonic",
}


@pytest.fixture(autouse=True)
def _single_gpu_tp(monkeypatch):
    monkeypatch.setattr(tp_info, "_TP_INFO", tp_info.DistributedInfo(0, 1))


def _owner(cls, **ops):
    """A served module of class ``cls`` holding only ``ops`` (no config needed)."""
    owner = cls.__new__(cls)
    owner.__dict__.update(ops)
    return owner


def _weighted(op, *shape):
    op.weight = torch.randn(*shape)
    return op


def _experts(sonic=True):
    return _owner(
        FusedGroupedExperts,
        num_experts=EXPERTS,
        _use_sonic_experts=sonic,
        gate_up_proj=torch.randn(EXPERT_SHAPES["w13"]),
        down_proj=torch.randn(EXPERT_SHAPES["w2"]),
    )


def _model(sonic=True):
    torch.manual_seed(0)
    attention = _owner(
        Attention,
        wq=_weighted(SimpleLinear(HIDDEN, HIDDEN), HIDDEN, HIDDEN),
        wkv=_weighted(ChunkedLinear(HIDDEN, 2 * KV, divisor=2), 2, KV, HIDDEN),
    )
    dense = _owner(
        FeedForward,
        w_in=_weighted(ChunkedLinear(HIDDEN, 2 * INTER, divisor=2), 2, INTER, HIDDEN),
        w_out=_weighted(LinearRowParallel(INTER, HIDDEN, has_bias=False), HIDDEN, INTER),
    )
    router = _owner(
        Router,
        router_linear=_weighted(_RouterLinear(HIDDEN, EXPERTS, has_bias=False), EXPERTS, HIDDEN),
        balancing_biases=torch.randn(EXPERTS),
    )
    moe = _owner(
        MoEFeedForward,
        router=router,
        latent_down=_weighted(SimpleLinear(HIDDEN, 3), 3, HIDDEN),
        experts=_experts(sonic),
    )
    layers = OPList(
        [
            SimpleNamespace(attention=attention, feed_forward=dense),
            SimpleNamespace(attention=attention, feed_forward=moe),
        ]
    )
    head = _weighted(MultiOutputHead(HIDDEN, VOCAB, min_rows=2), VOCAB, HIDDEN)
    return _owner(Zonos3ForCausalLM, layers=layers, multi_output=head)


TARGET_OPS = {
    "layers.0.attention.wq.weight": lambda m: m.layers.op_list[0].attention.wq,
    "layers.0.attention.wkv.weight": lambda m: m.layers.op_list[0].attention.wkv,
    "layers.0.feed_forward.w_in.weight": lambda m: m.layers.op_list[0].feed_forward.w_in,
    "layers.0.feed_forward.w_out.weight": lambda m: m.layers.op_list[0].feed_forward.w_out,
    "layers.1.feed_forward.latent_down.weight": lambda m: m.layers.op_list[1].feed_forward.latent_down,
    "layers.1.feed_forward.router.router_linear.weight": lambda m: m.layers.op_list[
        1
    ].feed_forward.router.router_linear,
    "multi_output.weight": lambda m: m.multi_output,
}


def _adapter(model, rank, seed, *, bias=False, exact=None, extra=None):
    generator = torch.Generator().manual_seed(seed)
    factors = {}
    for target, get in TARGET_OPS.items():
        weight = get(model).weight
        factors[target] = {
            "up": torch.randn(weight[..., 0].numel(), rank, generator=generator),
            "down": torch.randn(rank, weight.shape[-1], generator=generator),
            "shape": tuple(weight.shape),
        }
    adapter = {"format": FORMAT, "factors": factors | (extra or {}), "exact": dict(exact or {})}
    if bias:
        adapter["exact"]["layers.1.feed_forward.router.balancing_biases"] = torch.randn(EXPERTS, generator=generator)
    return adapter


def _expert_factors(rank, seed, weights=EXPERT_WEIGHTS):
    """/2 per-expert factors on layer 1's experts, as lingua accent_adapter exports them."""
    generator = torch.Generator().manual_seed(seed)
    factors = {}
    for weight in weights:
        num, out_features, in_features = EXPERT_SHAPES[weight]
        factors[f"layers.1.feed_forward.experts.{weight}"] = {
            "up": torch.randn(num, out_features, rank, generator=generator),
            "down": torch.randn(num, rank, in_features, generator=generator),
            "shape": EXPERT_SHAPES[weight],
            "experts": num,
        }
    return factors


def _expert_adapter(model, rank, expert_rank, seed, **kwargs):
    adapter = _adapter(model, rank, seed, extra=_expert_factors(expert_rank, seed + 100), **kwargs)
    adapter["format"] = FORMAT_EXPERTS
    return adapter


def _write(directory, adapters):
    for name, adapter in adapters.items():
        torch.save(adapter, directory / f"{name}.pt")


def _use_batch(monkeypatch, model, slots, scales):
    batch = SimpleNamespace(
        accent_slots=torch.tensor(slots, dtype=torch.int32),
        accent_scales=torch.tensor(scales, dtype=torch.float32),
    )
    batch.accent_mask = model._accents.mask(batch.accent_slots, batch.accent_scales, torch.float32)
    ctx = SimpleNamespace(batch=batch, _batch=batch)
    monkeypatch.setattr(accent_ops, "get_global_ctx", lambda: ctx)
    return batch


def test_targets_mirror_the_training_lora_contract():
    assert TARGETS == LINGUA_DEFAULT_TARGETS
    for module, layers in GDN375K_LORA_MODULES.items():
        assert accent_format.lora_target(f"layers.{layers[0]}.{module}") is not None
    assert accent_format.lora_target("multi_output") == "multi_output"
    assert accent_format.lora_target("layers.0.attention.conv") is None
    assert accent_format.lora_target("layers.1.feed_forward.experts.w13") is None
    # The release LM's prompt-side conditioning (fullenc_s2 speaker row, condition rows) stays frozen.
    assert accent_format.lora_target("speaker_projection") is None
    assert accent_format.lora_target("condition_embeddings.0") is None


def test_mixed_batch_serves_each_token_its_own_adapter_and_strength(tmp_path, monkeypatch):
    model = _model()
    base = {target: get(model).weight.clone() for target, get in TARGET_OPS.items()}
    biases = model.layers.op_list[1].feed_forward.router.balancing_biases.clone()
    adapters = {"cmn": _adapter(model, 3, 1, bias=True), "hin": _adapter(model, 2, 2)}
    _write(tmp_path, adapters)
    loaded = attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)
    assert loaded.names == ["cmn", "hin"] and loaded.slot("hin") == 2 and loaded.slot(None) == 0
    assert loaded.column_slots.tolist() == [1, 1, 1, 2, 2]
    with pytest.raises(ValueError):
        loaded.slot("deu")

    # Token rows: base, cmn at 1.0, hin at 0.5, cmn at -0.7, hin at 1.0.
    slots, scales = [0, 1, 2, 1, 2], [0.0, 1.0, 0.5, -0.7, 1.0]
    _use_batch(monkeypatch, model, slots, scales)
    for target, get in TARGET_OPS.items():
        op = get(model)
        weight = base[target].reshape(-1, base[target].shape[-1])
        x = torch.randn(len(slots), weight.shape[1])
        out = op.forward(x)
        for row, (slot, scale) in enumerate(zip(slots, scales)):
            merged = weight
            if slot:
                pair = adapters[loaded.names[slot - 1]]["factors"][target]
                merged = weight + scale * (pair["up"] @ pair["down"])
            expected = F.linear(x[row], merged)
            assert torch.allclose(out[row], expected, atol=1e-4), (target, row)

    router = model.layers.op_list[1].feed_forward.router
    bias = accent_ops.accent_router_bias(router, router.balancing_biases)
    shift = adapters["cmn"]["exact"]["layers.1.feed_forward.router.balancing_biases"]
    for row, (slot, scale) in enumerate(zip(slots, scales)):
        expected = biases + (scale * shift if slot == 1 else 0.0)
        assert torch.allclose(bias[row], expected)


def test_base_rows_are_bit_exact_and_padding_rows_stay_out_of_the_head(tmp_path, monkeypatch):
    model = _model()
    _write(tmp_path, {"cmn": _adapter(model, 4, 3)})
    head = model.multi_output
    x = torch.randn(1, HIDDEN)
    before = head.forward(x)  # one row, padded to min_rows inside the op
    attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)
    monkeypatch.setattr(accent_ops, "get_global_ctx", lambda: SimpleNamespace(_batch=None))
    assert torch.equal(head.forward(x), before)  # outside a forward: no mask, no shift
    _use_batch(monkeypatch, model, [0], [0.0])
    assert torch.equal(head.forward(x), before)
    _use_batch(monkeypatch, model, [1], [1.0])
    assert not torch.equal(head.forward(x), before)


def test_fixed_block_side_path_matches_the_plain_product(tmp_path):
    model = _model()
    _write(tmp_path, {"cmn": _adapter(model, 3, 9), "hin": _adapter(model, 2, 10)})
    loaded = attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)
    head = model.multi_output
    x = torch.randn(7, HIDDEN)
    mask = loaded.mask(torch.tensor([0, 1, 2, 1, 2, 2, 1]), torch.linspace(0.5, 1.5, 7), torch.float32)
    plain = accent_ops.accent_shift(head, x, mask)
    head._accent_block_rows = 4  # 7 rows -> two [4, K] GEMMs, the second padded
    blocked = accent_ops.accent_shift(head, x, mask)
    assert blocked.shape == plain.shape and torch.allclose(blocked, plain, atol=1e-6)
    assert torch.equal(accent_ops.accent_shift(head, x[:1], mask[:1]), blocked[:1])

    # The load-time probe leaves every side path batch-invariant (switching ops it must).
    switched = accent_adapters.ensure_batch_invariance(loaded)
    assert set(switched) <= set(loaded.ops) and len(loaded.ops) == len(TARGET_OPS)


def test_router_state_follows_the_converter(tmp_path):
    model = _model()
    router = "layers.1.feed_forward.router"
    for key in (
        f"{router}.balancing_biases",
        f"{router}.parametrizations.balancing_biases.original",
        f"{router}.ent_denom",
        f"{router}.normalized_entropy",
    ):
        served, _ = accent_format.served_exact_name(key)
        assert served == _output_key(key)

    legacy = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0])
    exact = {
        f"{router}.parametrizations.balancing_biases.original": legacy,
        f"{router}.normalized_entropy": torch.tensor(0.25),
    }
    _write(tmp_path, {"cmn": _adapter(model, 2, 4, exact=exact)})
    attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)
    table = model.layers.op_list[1].feed_forward.router._accent_bias
    assert torch.allclose(table[1], legacy - legacy.mean()) and not table[0].any()


@pytest.mark.parametrize(
    ("factor", "exact", "match"),
    [
        ("layers.1.feed_forward.experts.w13", None, "MoE expert"),
        ("layers.0.attention_norm", None, "outside the LoRA targets"),
        ("layers.0.attention.wo", None, "lacks"),
        (None, {"layers.0.attention_norm.weight": torch.zeros(HIDDEN)}, "attention_norm"),
        # ZONOS3 Mini release: the speaker projection (Linear 192 -> 1024) and the condition-row tables.
        ("speaker_projection", None, "outside the LoRA targets"),
        (None, {"speaker_projection.weight": torch.zeros(HIDDEN, 192)}, "speaker_projection"),
        (None, {"condition_embeddings.0.weight": torch.zeros(11, HIDDEN)}, "condition_embeddings"),
    ],
)
def test_unservable_adapters_are_rejected(tmp_path, factor, exact, match):
    model = _model()
    extra = None
    if factor is not None:
        extra = {f"{factor}.weight": {"up": torch.zeros(2, 1), "down": torch.zeros(1, HIDDEN), "shape": (2, HIDDEN)}}
    _write(tmp_path, {"bad": _adapter(model, 2, 5, exact=exact, extra=extra)})
    with pytest.raises(ValueError):
        attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)


def test_shape_mismatch_is_rejected(tmp_path):
    model = _model()
    adapter = _adapter(model, 2, 6)
    adapter["factors"]["layers.0.attention.wkv.weight"]["shape"] = (2 * KV, HIDDEN)
    _write(tmp_path, {"bad": adapter})
    with pytest.raises(ValueError):
        attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)


def test_v2_expert_factors_load_per_slot_grouped_by_expert_rank(tmp_path):
    model = _model()
    adapters = {
        "a_e4": _expert_adapter(model, 3, 4, 11),
        "b_r2": _adapter(model, 2, 12),  # /1: no expert factors
        "c_e8": _expert_adapter(model, 2, 8, 13),
        "d_e4": _expert_adapter(model, 3, 4, 14),
    }
    _write(tmp_path, adapters)
    loaded = attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)
    # Expert ranks stay out of the shared column space of the linear factors.
    assert loaded.column_slots.tolist() == [1] * 3 + [2] * 2 + [3] * 2 + [4] * 3
    experts = model.layers.op_list[1].feed_forward.experts
    assert loaded.expert_ops == {"layers.1.feed_forward.experts": experts}
    accents = experts._accent_experts
    e4, e8 = accents.groups
    assert (e4.rank, e8.rank) == (4, 8) and accents.num_experts == EXPERTS
    # Group local * E + e per rank; b_r2 (/1) and no accent are in neither.
    assert e4.first.tolist() == [-1, 0, -1, -1, EXPERTS]
    assert e8.first.tolist() == [-1, -1, -1, 0, -1]
    for group, local, name in ((e4, 0, "a_e4"), (e4, 1, "d_e4"), (e8, 0, "c_e8")):
        rows = slice(local * EXPERTS, (local + 1) * EXPERTS)
        for weight in EXPERT_WEIGHTS:
            pair = adapters[name]["factors"][f"layers.1.feed_forward.experts.{weight}"]
            down, up = group.factors[weight]
            assert torch.equal(down[rows], pair["down"].transpose(1, 2))
            assert torch.equal(up[rows], pair["up"].transpose(1, 2))
    linear = sum(t.numel() for op in loaded.ops.values() for t in op._accent)
    expert = sum(t.numel() for g in (e4, e8) for pair in g.factors.values() for t in pair)
    assert loaded.nbytes() == 4 * (linear + expert)


def test_v1_adapters_leave_the_experts_alone(tmp_path):
    model = _model()
    _write(tmp_path, {"cmn": _adapter(model, 3, 1), "hin": _adapter(model, 2, 2)})
    loaded = attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)
    assert loaded.expert_ops == {}
    assert "_accent_experts" not in model.layers.op_list[1].feed_forward.experts.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the expert side path is a Triton kernel")
def test_expert_side_path_is_the_dense_merge_per_routed_pair(tmp_path):
    """Every (token, expert) pair gets W_e + alpha * up_e @ down_e of its request's adapter (fp32)."""
    model = _model()
    adapters = {
        "a_e4": _expert_adapter(model, 3, 4, 21),
        "b_r2": _adapter(model, 2, 22),
        "c_e8": _expert_adapter(model, 2, 8, 23),
    }
    _write(tmp_path, adapters)
    cuda = torch.device("cuda")
    loaded = attach_accent_adapters(model, tmp_path, cuda, torch.float32)
    accents = model.layers.op_list[1].feed_forward.experts._accent_experts
    generator = torch.Generator().manual_seed(5)
    pairs = 60
    slots = torch.randint(0, 4, (pairs,), generator=generator, dtype=torch.int32)
    experts = torch.randint(0, EXPERTS, (pairs,), generator=generator, dtype=torch.int32)
    scales = torch.rand(pairs, generator=generator) * 2 - 0.5
    tokens = torch.arange(pairs, dtype=torch.int32)
    for weight in EXPERT_WEIGHTS:
        num, out_features, in_features = EXPERT_SHAPES[weight]
        base = torch.randn(num, out_features, in_features, generator=generator)
        x = torch.randn(pairs, in_features, generator=generator)
        shift = accent_ops.expert_pair_shift(
            accents,
            weight,
            x.to(cuda),
            tokens.to(cuda),
            experts.to(cuda),
            slots.to(cuda),
            scales.to(cuda),
        ).cpu()
        for p in range(pairs):
            slot, e, alpha = int(slots[p]), int(experts[p]), float(scales[p])
            name = loaded.names[slot - 1] if slot else None
            factors = adapters[name]["factors"] if name else {}
            pair = factors.get(f"layers.1.feed_forward.experts.{weight}")
            if pair is None:  # no accent, or a /1 adapter: an exact zero
                assert torch.equal(shift[p], torch.zeros(out_features)), (weight, p, slot)
                continue
            merged = base[e] + alpha * (pair["up"][e] @ pair["down"][e])
            served = base[e] @ x[p] + shift[p]
            assert torch.allclose(served, merged @ x[p], rtol=1e-4, atol=1e-4), (weight, p, slot)
    # The load-time probe covers the expert side paths too (it raises if they vary).
    assert set(accent_adapters.ensure_batch_invariance(loaded)) <= set(loaded.ops)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda a: a.update(format=FORMAT), "MoE expert"),
        (
            lambda a: a["factors"].__setitem__(
                "layers.1.feed_forward.experts.w1", a["factors"]["layers.1.feed_forward.experts.w2"]
            ),
            "servable expert factors",
        ),
        (
            lambda a: a["factors"]["layers.1.feed_forward.experts.w13"].__setitem__("up", torch.zeros(EXPERTS, 3, 4)),
            "do not shift",
        ),
        (lambda a: a["factors"]["layers.1.feed_forward.experts.w13"].__setitem__("experts", 4), "do not shift"),
        (lambda a: a["factors"].update(_expert_factors(2, 9, weights=("w2",))), "mix ranks"),
    ],
)
def test_unservable_expert_factors_are_rejected(tmp_path, edit, match):
    model = _model()
    adapter = _expert_adapter(model, 2, 4, 31)
    edit(adapter)
    _write(tmp_path, {"bad": adapter})
    with pytest.raises(ValueError):
        attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)


def test_expert_factors_need_the_exact_sonic_path(tmp_path):
    model = _model(sonic=False)
    _write(tmp_path, {"cmn": _expert_adapter(model, 2, 4, 41)})
    with pytest.raises(ValueError):
        attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)


def test_every_gdn375k_lora_module_maps_to_the_op_that_computes_it(tmp_path, monkeypatch):
    pytest.importorskip("fla", reason="FLA overlay not on the path")  # OriginalGDN imports it
    from zonos3.models.config import ModelConfig
    from zonos3.utils.hf import Zonos3Config
    from zonos3_sgl.layers import rotary
    from zonos3_sgl.models.gdn import OriginalGDN, _Weight
    from zonos3_sgl.models.zonos3 import SimpleLinearOProj

    monkeypatch.setattr(rotary, "_ROPE_DEVICE", torch.device("cpu"))
    monkeypatch.setenv("ZONOS3_USE_TRAINING_SONIC_EXPERTS", "1")  # expert factors need the exact path
    config = ModelConfig.from_zonos3_config(Zonos3Config(**GDN375K_PARAMS))
    with torch.device("meta"):
        model = Zonos3ForCausalLM(config)

    modules = [f"layers.{i}.{m}" for m, layers in GDN375K_LORA_MODULES.items() for i in layers]
    modules.append("multi_output")
    assert len(modules) == 59

    def weight_of(module):
        obj = model
        for part in module.split("."):
            obj = obj.op_list[int(part)] if isinstance(obj, OPList) else getattr(obj, part)
        return obj

    generator = torch.Generator().manual_seed(7)
    factors = {}
    for module in modules:
        shape = tuple(weight_of(module).weight.shape)
        out_features = torch.Size(shape[:-1]).numel()
        factors[f"{module}.weight"] = {
            "up": torch.randn(out_features, 2, generator=generator).bfloat16(),
            "down": torch.randn(2, shape[-1], generator=generator).bfloat16(),
            "shape": shape,
        }
    exact = {
        f"layers.{i}.feed_forward.router.balancing_biases": torch.randn(64, generator=generator) for i in range(1, 11)
    }
    # /2: rank-2 per-expert factors on every MoE layer (lingua SonicExperts w13 / w2).
    experts = {}
    for i in range(1, 11):
        op = model.layers.op_list[i].feed_forward.experts
        for weight, served_weight in (("w13", op.gate_up_proj), ("w2", op.down_proj)):
            num, out_features, in_features = served_weight.shape
            experts[f"layers.{i}.feed_forward.experts.{weight}"] = {
                "up": torch.randn(num, out_features, 2, generator=generator).bfloat16(),
                "down": torch.randn(num, 2, in_features, generator=generator).bfloat16(),
                "shape": tuple(served_weight.shape),
                "experts": num,
            }
    assert experts["layers.1.feed_forward.experts.w13"]["shape"] == (64, 3072, 256)
    assert experts["layers.1.feed_forward.experts.w2"]["shape"] == (64, 256, 1536)
    _write(tmp_path, {"por": {"format": FORMAT_EXPERTS, "factors": factors | experts, "exact": exact}})
    attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.bfloat16)

    served = {
        "attention.in_proj": (OriginalGDN, _Weight),
        "attention.wq": (Attention, SimpleLinear),
        "attention.wkv": (Attention, ChunkedLinear),
        "attention.gater": (Attention, SimpleLinear),
        "feed_forward.w_in": (FeedForward, ChunkedLinear),
        "feed_forward.w_out": (FeedForward, LinearRowParallel),
        "feed_forward.latent_down": (MoEFeedForward, SimpleLinear),
        "feed_forward.latent_up": (MoEFeedForward, SimpleLinear),
        "feed_forward.router.router_linear": (Router, _RouterLinear),
    }
    for module in modules:
        op = weight_of(module)
        down, up = op._accent
        pair = factors[f"{module}.weight"]
        assert torch.equal(down, pair["down"]) and torch.equal(up, pair["up"].T), module
        suffix = module.split(".", 2)[-1]
        if suffix == "attention.wo":
            layer = model.layers.op_list[int(module.split(".")[1])]
            gdn = isinstance(layer.attention, OriginalGDN)
            assert isinstance(op, _Weight if gdn else SimpleLinearOProj), module
        elif suffix in served:
            owner_cls, op_cls = served[suffix]
            layer = model.layers.op_list[int(module.split(".")[1])]
            owner = getattr(layer, suffix.split(".")[0])
            if suffix.startswith("feed_forward.router"):
                owner = owner.router
            assert isinstance(owner, owner_cls) and isinstance(op, op_cls), module
        else:
            assert isinstance(op, MultiOutputHead)
    routers = [layer.feed_forward.router for layer in model.layers.op_list if layer.is_moe]
    assert len(routers) == 10
    for i, router in enumerate(routers, start=1):
        key = f"layers.{i}.feed_forward.router.balancing_biases"
        assert torch.equal(router._accent_bias[1], exact[key]) and not router._accent_bias[0].any()
    for i in range(1, 11):
        op = model.layers.op_list[i].feed_forward.experts
        (group,) = op._accent_experts.groups
        assert group.rank == 2 and group.first.tolist() == [-1, 0]
        for weight in EXPERT_WEIGHTS:
            pair = experts[f"layers.{i}.feed_forward.experts.{weight}"]
            down, up = group.factors[weight]
            assert torch.equal(down, pair["down"].transpose(1, 2))
            assert torch.equal(up, pair["up"].transpose(1, 2))


class _FakeGDNKernels:
    """Row-local CPU stand-ins for the FLA kernels: out = (x W_v) * (x W_z), then wo."""

    @staticmethod
    def conv(qkv, weight, initial_state, output_final_state, activation, backend, cu_seqlens):
        return qkv, initial_state

    @staticmethod
    def prefill(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, **kwargs):
        return v, initial_state

    @staticmethod
    def norm(output, z, weight, bias, activation, eps):
        return output * z


def _gdn(monkeypatch, capacity):
    pytest.importorskip("fla", reason="FLA overlay not on the path")  # OriginalGDN imports it
    from zonos3_sgl.models import gdn as gdn_module

    gdn = gdn_module.OriginalGDN(SimpleNamespace(hidden_size=HIDDEN), layer_id=0)
    gdn.in_proj.weight = torch.randn(2056, HIDDEN) * 0.3
    gdn.wo.weight = torch.randn(HIDDEN, 512) * 0.1
    gdn.A_log = gdn.dt_bias = torch.zeros(4)
    gdn._conv_fn = _FakeGDNKernels.conv
    gdn._prefill_fn = _FakeGDNKernels.prefill
    gdn._norm_fn = _FakeGDNKernels.norm
    gdn.allocate_cache(capacity, torch.device("cpu"), torch.float32)
    return gdn_module, gdn


def test_gdn_mixer_shifts_replayed_tails_and_graph_windows_by_their_requests_accent(tmp_path, monkeypatch):
    torch.manual_seed(0)
    gdn_module, gdn = _gdn(monkeypatch, capacity=4)
    model = _owner(Zonos3ForCausalLM, layers=OPList([SimpleNamespace(attention=gdn)]))
    generator = torch.Generator().manual_seed(8)

    def adapter(rank):
        return {
            "format": FORMAT,
            "exact": {},
            "factors": {
                "layers.0.attention.in_proj.weight": {
                    "up": torch.randn(2056, rank, generator=generator) * 0.3,
                    "down": torch.randn(rank, HIDDEN, generator=generator) * 0.3,
                    "shape": (2056, HIDDEN),
                },
                "layers.0.attention.wo.weight": {
                    "up": torch.randn(HIDDEN, rank, generator=generator) * 0.3,
                    "down": torch.randn(rank, 512, generator=generator) * 0.3,
                    "shape": (HIDDEN, 512),
                },
            },
        }

    adapters = {"cmn": adapter(3), "hin": adapter(2)}
    _write(tmp_path, adapters)
    w_in, w_o = gdn.in_proj.weight.clone(), gdn.wo.weight.clone()
    loaded = attach_accent_adapters(model, tmp_path, torch.device("cpu"), torch.float32)

    def expected(x, slot, scale):
        wi, wo = w_in, w_o
        if slot:
            factors = adapters[loaded.names[slot - 1]]["factors"]
            pi, po = factors["layers.0.attention.in_proj.weight"], factors["layers.0.attention.wo.weight"]
            wi = wi + scale * (pi["up"] @ pi["down"])
            wo = wo + scale * (po["up"] @ po["down"])
        mixed = x @ wi.T
        return (mixed[:, 1024:1536] * mixed[:, 1536:2048]) @ wo.T

    # Request (table slot, accent slot, strength): base, hin at 0.5, cmn at 1.2.
    accents = [(0, 0, 0.0), (1, 2, 0.5), (2, 1, 1.2)]
    reqs = [SimpleNamespace(table_idx=t, uid=t, is_cfg_uncond=False, cached_len=0, device_len=0) for t, _, _ in accents]

    def step(lengths, *, decode, static=False):
        for req, n in zip(reqs, lengths):
            req.cached_len, req.device_len = req.device_len, req.device_len + n
            req.extend_len = n
        slots = [s for (_, s, _), n in zip(accents, lengths) for _ in range(n)]
        scales = [a for (_, _, a), n in zip(accents, lengths) for _ in range(n)]
        batch = _use_batch(monkeypatch, model, slots, scales)
        batch.reqs = batch.padded_reqs = reqs
        batch.is_decode, batch.padded_size = decode, len(reqs)
        monkeypatch.setattr(gdn_module, "get_global_ctx", lambda: SimpleNamespace(batch=batch))
        x = torch.randn(sum(lengths), HIDDEN)
        out = gdn.forward(x)
        rows = torch.tensor(lengths).cumsum(0) - torch.tensor(lengths)
        for (_, slot, scale), start, n in zip(accents, rows.tolist(), lengths):
            want = expected(x[start : start + n], slot, scale)
            assert torch.allclose(out[start : start + n], want, atol=1e-4), (slot, static)

    # Prefill: one request crosses a 64-row chunk (committed + tail), the others stay in their tails.
    step([70, 5, 63], decode=False)
    # Eager decode: each request replays its tail; request 2 completes its chunk.
    step([1, 1, 1], decode=True)
    # Graph decode: fixed 64-row windows per request.
    slots = torch.tensor([t for t, _, _ in accents], dtype=torch.long)
    gdn.enable_graph_decode(slots, gdn_module.OriginalGDN.static_tables(torch.device("cpu"), [3]))
    gdn.set_static_mode(True)
    for _ in range(3):
        step([1, 1, 1], decode=True, static=True)


def test_scheduler_aligns_accents_with_the_flattened_tokens():
    reqs = [
        SimpleNamespace(extend_len=3, accent_slot=2, accent_scale=0.7),
        SimpleNamespace(extend_len=1, accent_slot=0, accent_scale=0.0),
        SimpleNamespace(extend_len=2, accent_slot=1, accent_scale=1.0),
    ]
    batch = SimpleNamespace(padded_reqs=reqs)
    scheduler = SimpleNamespace(engine=SimpleNamespace(accent_adapters=object()), device="cpu")
    TTSScheduler._prepare_accents(scheduler, batch)
    assert batch.accent_slots.tolist() == [2, 2, 2, 0, 1, 1]
    assert torch.allclose(batch.accent_scales, torch.tensor([0.7, 0.7, 0.7, 0.0, 1.0, 1.0]))

    scheduler.engine.accent_adapters = None
    TTSScheduler._prepare_accents(scheduler, batch)
    assert batch.accent_slots is None and batch.accent_scales is None
