"""MLX LM vs the plain-PyTorch reference (zonos3.reference.lm.Zonos3LM) on a random tiny backbone converted with
tools/convert_gguf.py lm: prefill + decode logits, hidden deltas, speaker / condition rows, the alignment probe and an
accent adapter (the tiny model of servers/ggml/tests/parity/test_ggml_graph.py)."""

from __future__ import annotations

import json

import convert_gguf
import numpy as np
import pytest
import torch
from conftest import FIXTURES
from gguf import GGUFReader
from zonos3.models.accent_format import FORMAT_EXPERTS, grow_adapter, load_adapter
from zonos3.reference.accents import AccentState as RefAccentState
from zonos3.reference.lm import LMCache, ProbeSpec, Zonos3LM, Zonos3LMConfig
from zonos3_mlx.companions import AccentBank
from zonos3_mlx.models.lm import LMState, Zonos3MLX

CODEC = "tiny_codec"
TEXT_LEN = 14
HEADS = ((3, 1), (7, 2), (7, 0))
WINDOWS = {(3, 1): (2, 5), (7, 2): (6, 9)}


def tiny_params() -> dict:
    return {
        "attention_pattern": "gdn3_global1", "residual_type": "full_attnres", "use_xsa": False,
        "speaker_enabled": True, "condition_tokens_enabled": True, "moe_router_type": "linear",
        "moe_router_use_eda": False, "moe_balancing_strategy": "quantile", "moe_router_topk": 2,
        "moe_start_from_layer": 1, "moe_end_from_layer": 0,
        "n_layers": 8, "dim": 64, "head_dim": 16, "n_heads": 4, "n_kv_heads": 2,
        "n_codebooks": 10, "codebook_size": 4096, "eoa_id": 4096, "audio_pad_id": 4097,
        "text_vocab": 8196, "text_extra_tokens": 0,
        "text_tokenizer_spec": {"kind": "superbpe", "path": "tokenizer.json", "vocab_size": 8192,
                                "offset": 4, "bos": 2, "eos": 3},
        "moe_alpha": 0.5, "moe_n_experts": 4, "multiple_of": 32, "ffn_dim_multiplier": 1.0,
        "condition_count": 3, "condition_n_buckets": 4, "condition_bucket_counts": [4, 4, 6],
        "condition_names": ["loudness", "fmax", "speaking_rate"],
        "condition_edges": {"loudness": [-30.0, -20.0, -10.0], "fmax": [4000.0, 8000.0, 12000.0],
                            "speaking_rate": [10.0, 12.0, 14.0, 16.0, 18.0]},
        "condition_units": {"loudness": "LUFS", "fmax": "Hz", "speaking_rate": "utf-8 bytes per second"},
        "norm_eps": 1e-5, "rope_theta": 10000.0, "max_seqlen": 512, "loss_softcap": 15.0,
        "speaker_embedding_dim": 8, "speaker_reference_max_seconds": 60.0, "audio_codec": CODEC,
    }


def fixture_tokenizer() -> str:
    field = GGUFReader(FIXTURES / "dummy-zonos3.gguf").fields["zonos3.tokenizer.json"]
    return bytes(field.parts[field.data[0]]).decode()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("lm")
    lm_dir = root / "lm"
    lm_dir.mkdir()
    params = tiny_params()
    model = Zonos3LM(Zonos3LMConfig.from_params(params))
    generator = torch.Generator().manual_seed(0)
    state = {}
    for name, tensor in model.state_dict().items():
        value = torch.randn(tensor.shape, generator=generator) * 0.05
        if name.endswith("attention.temp"):
            value = 3.0 + value
        state[name] = value if name.endswith("balancing_biases") else value.to(torch.bfloat16)
    torch.save(state, lm_dir / "model.pth")
    (lm_dir / "params.json").write_text(json.dumps({"model": params}))
    (lm_dir / "tokenizer.json").write_text(fixture_tokenizer())
    silence = root / "silence.json"
    silence.write_text(json.dumps({"codec": CODEC, "frame": [11 * i for i in range(10)]}))
    gguf = root / "tiny.gguf"
    convert_gguf.main(["lm", str(lm_dir), "--outtype", "f32", "--silence-frame", str(silence), "-o", str(gguf)])
    return root, Zonos3LM.load(lm_dir), Zonos3MLX.load(gguf)


def prompt_rows(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.full((TEXT_LEN + 3, 11), 4097, dtype=np.int64)
    rows[:, 10] = 8196
    rows[0, 10] = 2
    rows[1:TEXT_LEN - 1, 10] = rng.integers(4, 8196, TEXT_LEN - 2)
    rows[TEXT_LEN - 1, 10] = 3
    rows[TEXT_LEN:, :10] = rng.integers(0, 4096, (3, 10))
    return rows


def decode_rows(n: int = 6, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = np.full((n, 11), 8196, dtype=np.int64)
    rows[:, :10] = rng.integers(0, 4096, (n, 10))
    return rows


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_both(ref, mlx, *, steps=6, probe=None, ref_accents=None, mlx_accents=None, **kw):
    """Teacher-forced (logits per row, readouts per decode row) of both models."""
    prompt, rows = prompt_rows(), decode_rows(steps)
    cache, state = LMCache(), LMState()
    ref_kw = dict(kw.items())
    r = ref.forward(torch.from_numpy(prompt), cache, accents=ref_accents, **ref_kw)
    m = mlx.forward(prompt, state, accents=mlx_accents, **kw)
    pairs = [(r.logits.numpy(), np.array(m.logits))]
    readouts = []
    for row in rows:
        kw_step = {"hidden_deltas": kw["hidden_deltas"]} if "hidden_deltas" in kw else {}
        r = ref.forward(torch.from_numpy(row[None]), cache, probe=probe, accents=ref_accents, **kw_step)
        m = mlx.forward(row[None], state, probe=probe, accents=mlx_accents, **kw_step)
        pairs.append((r.logits.numpy(), np.array(m.logits)))
        if probe is not None:
            readouts.append((r.probe[0], m.probe[0]))
    return pairs, readouts


def assert_logits(pairs, cos=0.99999):
    for i, (want, got) in enumerate(pairs):
        for row in range(len(want)):
            assert cosine(got[row], want[row]) >= cos, f"call {i} row {row}: cos {cosine(got[row], want[row])}"
            np.testing.assert_array_equal(got[row].argmax(-1), want[row].argmax(-1), err_msg=f"call {i} row {row}")


def test_prefill_and_decode(tiny):
    _, ref, mlx = tiny
    pairs, _ = run_both(ref, mlx)
    assert_logits(pairs)


def test_hidden_deltas(tiny):
    _, ref, mlx = tiny
    g = torch.Generator().manual_seed(11)
    deltas = {0: torch.randn(64, generator=g) * 0.5, 5: torch.randn(64, generator=g) * 0.5}
    pairs, _ = run_both(ref, mlx, hidden_deltas=deltas)
    plain, _ = run_both(ref, mlx)
    assert cosine(pairs[-1][0], plain[-1][0]) < 0.9999, "the deltas barely move the logits"
    assert_logits(pairs)


def test_speaker_and_condition_rows(tiny):
    _, ref, mlx = tiny
    g = torch.Generator().manual_seed(12)
    speaker = {TEXT_LEN: torch.randn(8, generator=g)}
    conditions = {TEXT_LEN + 1: 1, TEXT_LEN + 2: 3}
    pairs, _ = run_both(ref, mlx, speaker_rows=speaker, condition_rows=conditions)
    assert_logits(pairs)


@pytest.mark.parametrize("windows", [None, WINDOWS], ids=["readout", "masked"])
def test_probe_readout(tiny, windows):
    _, ref, mlx = tiny
    probe = ProbeSpec(HEADS, 0, TEXT_LEN, windows)
    pairs, readouts = run_both(ref, mlx, probe=probe)
    assert_logits(pairs)
    for step, ((ra, rc), (ma, mc)) in enumerate(readouts):
        assert ra == ma, f"step {step}: argmax {ma} != reference {ra}"
        assert abs(rc - mc) <= 1e-3, f"step {step}: centroid {mc} vs {rc}"


def write_adapter(lm: Zonos3LM, path, seed: int = 5, rank: int = 4) -> None:
    g = torch.Generator().manual_seed(seed)
    params = dict(lm.named_parameters())

    def rnd(*shape, scale=0.08):
        return torch.randn(*shape, generator=g) * scale

    factors, exact = {}, {}
    for module in ("layers.0.attention.in_proj", "layers.0.attention.wo", "layers.0.feed_forward.w_in",
                   "layers.0.feed_forward.w_out", "layers.3.attention.wq", "layers.3.attention.wkv",
                   "layers.3.attention.wo", "layers.3.attention.gater", "layers.5.feed_forward.latent_down",
                   "layers.5.feed_forward.latent_up", "layers.6.feed_forward.router.router_linear", "multi_output"):
        shape = tuple(params[module + ".weight"].shape)
        flat_out, flat_in = int(np.prod(shape[:-1])), shape[-1]
        factors[module + ".weight"] = {"down": rnd(rank, flat_in), "up": rnd(flat_out, rank), "shape": shape}
    for layer in (2, 7):
        experts = f"layers.{layer}.feed_forward.experts"
        for weight, param in (("w13", "gate_up_proj"), ("w2", "down_proj")):
            e, out, inp = params[f"{experts}.{param}"].shape
            factors[f"{experts}.{weight}"] = {"up": rnd(e, out, rank), "down": rnd(e, rank, inp),
                                              "shape": (e, out, inp), "experts": e}
    for layer in (1, 6):
        exact[f"layers.{layer}.feed_forward.router.balancing_biases"] = rnd(lm.config.n_experts, scale=0.6)
    torch.save({"format": FORMAT_EXPERTS, "factors": factors, "exact": exact}, path)


def test_accent_adapter(tiny):
    root, ref, mlx = tiny
    adapters = root / "adapters"
    adapters.mkdir(exist_ok=True)
    write_adapter(ref, adapters / "alpha.pt")
    write_adapter(ref, adapters / "beta.pt", seed=9, rank=2)
    gguf = root / "accents.gguf"
    convert_gguf.main(["accents", str(adapters), "-o", str(gguf)])
    bank = AccentBank.load(gguf)
    assert bank.names == ["alpha", "beta"]
    strength = 0.8
    ref_state = RefAccentState.for_model(ref, load_adapter(adapters / "beta.pt"), strength)
    pairs, _ = run_both(ref, mlx, ref_accents=ref_state, mlx_accents=bank.state("beta", strength))
    plain, _ = run_both(ref, mlx)
    assert cosine(pairs[0][0][-1], plain[0][0][-1]) < 0.99999, "the adapter barely moves the logits"
    for i, (want, got) in enumerate(pairs):
        for row in range(len(want)):
            assert cosine(got[row], want[row]) >= 0.9999, f"call {i} row {row}"


def test_edit_adapter_backbone_keeps_plain_logits_and_fits_old_head_factors(tiny):
    """`lm --edit-adapter`: a backbone with the /3 adapter's vocabulary; plain logits on the old ids are the ungrown
    backbone's, the /3 slot is the bank's edit adapter (not an accent), and an adapter trained before the vocabulary
    grew still matches the reference on the grown model."""
    root, ref, mlx = tiny
    adapters = root / "edit-adapters"
    adapters.mkdir(exist_ok=True)
    write_adapter(ref, adapters / "plain.pt", seed=7, rank=2)
    write_adapter(ref, root / "export.pt", seed=11)
    g = torch.Generator().manual_seed(4)
    dim = ref.config.dim
    rows = {f"multi_embedder.embedders.{i}.weight": torch.randn(1, dim, generator=g) * 0.05 for i in range(10)}
    rows |= {"multi_embedder.embedders.10.weight": torch.randn(3, dim, generator=g) * 0.05,
             "multi_output.weight": torch.randn(10, dim, generator=g) * 0.05}
    rows_before = dict.fromkeys(rows, 4098) | {"multi_embedder.embedders.10.weight": 8197}
    torch.save(grow_adapter(torch.load(root / "export.pt"), rows_before,
                            {"span_end_id": 4098, "text_extra_tokens": 3, "rows": rows}), adapters / "edit.pt")
    accents = root / "edit-accents.gguf"
    convert_gguf.main(["accents", str(adapters), "-o", str(accents)])
    grown_gguf = root / "grown.gguf"
    convert_gguf.main(["lm", str(root / "lm"), "--outtype", "f32", "--silence-frame", str(root / "silence.json"),
                       "--edit-adapter", str(adapters / "edit.pt"), "-o", str(grown_gguf)])
    grown = Zonos3MLX.load(grown_gguf)
    assert grown.config.span_end_id == 4098 and grown.config.text_extra_tokens == 3
    c = grown.config
    bank = AccentBank.load(accents, head_rows=c.n_codebooks * c.audio_vocab, n_codebooks=c.n_codebooks)
    assert bank.edit == "edit" and bank.accent_names == ["plain"]

    plain, _ = run_both(ref, mlx)
    on_grown, _ = run_both(ref, grown)
    for (_, want), (_, got) in zip(plain, on_grown):
        np.testing.assert_array_equal(got[..., :4098], want)

    ref_grown = Zonos3LM.load(root / "lm").grow(load_adapter(adapters / "edit.pt").grown)
    ref_state = RefAccentState.for_model(ref_grown, load_adapter(adapters / "plain.pt"), 0.8)
    pairs, _ = run_both(ref_grown, grown, ref_accents=ref_state, mlx_accents=bank.state("plain", 0.8))
    for i, (want, got) in enumerate(pairs):
        for row in range(len(want)):
            assert cosine(got[row], want[row]) >= 0.9999, f"call {i} row {row}"
