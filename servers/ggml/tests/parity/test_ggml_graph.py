"""Graph-feature parity of the ggml backbone against the plain-PyTorch reference (zonos3.reference.lm.Zonos3LM):

(a) the decode-step alignment probe (`zonos3-cli --probe-heads/--probe-window/--dump-probe`) against
    `Zonos3LM.forward(probe=ProbeSpec)` teacher-forced on the rows ggml generated: identical argmax per step, centroid
    within 1e-3, plus the logits (the masked heads' recomputed attention feeds the following layers);
(b) accent adapters (`--accents/--accent/--dump-logits`) against `zonos3.reference.accents.AccentState`: cos >= 0.9999
    on the prompt's last row and on every decode step.

The tiny backbone is a random Zonos3LM with the dimensions the ggml loader indexes into (10 x 4096 codebooks, the
8196-token SuperBPE vocabulary, 128-wide GDN heads) and global layers 3 and 7, converted with `tools/convert_gguf.py lm`.
Accent adapters go through `convert_gguf.py accents` as well.

    uv run --package zonos3-tools pytest servers/ggml/tests/parity/test_ggml_graph.py --ggml-build servers/ggml/build
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
GGML = REPO / "servers" / "ggml"
sys.path.insert(0, str(REPO / "tools"))

import convert_gguf  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from zonos3.models.accent_format import FORMAT_EXPERTS, grow_adapter, load_adapter  # noqa: E402
from zonos3.reference.accents import AccentState  # noqa: E402
from zonos3.reference.lm import LMCache, ProbeSpec, Zonos3LM, Zonos3LMConfig  # noqa: E402

CODEC = "tiny_codec"
TEXT_LEN = 14  # BOS + 12 tokens + EOS
HEADS = [(3, 1), (7, 2), (7, 0)]
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
        "condition_count": 3, "condition_n_buckets": 4,
        "condition_names": ["loudness", "fmax", "speaking_rate"],
        "condition_edges": {"loudness": [-30.0, -20.0, -10.0], "fmax": [4000.0, 8000.0, 12000.0],
                            "speaking_rate": [10.0, 14.0, 18.0]},
        "condition_units": {"loudness": "LUFS", "fmax": "Hz", "speaking_rate": "utf-8 bytes per second"},
        "norm_eps": 1e-5, "rope_theta": 10000.0, "max_seqlen": 512, "loss_softcap": 15.0,
        "speaker_embedding_dim": 8, "speaker_reference_max_seconds": 60.0, "audio_codec": CODEC,
    }


def fixture_tokenizer() -> str:
    """The pinned SuperBPE tokenizer embedded in tests/fixtures/dummy-zonos3.gguf."""
    field = GGUFReader(GGML / "tests" / "fixtures" / "dummy-zonos3.gguf").fields["zonos3.tokenizer.json"]
    return bytes(field.parts[field.data[0]]).decode()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, ggml_build):
    """(checkpoint dir, f32 GGUF, fp32 reference) of a random tiny backbone."""
    root = tmp_path_factory.mktemp("graph")
    lm_dir = root / "lm"
    lm_dir.mkdir()
    params = tiny_params()
    model = Zonos3LM(Zonos3LMConfig.from_params(params))
    generator = torch.Generator().manual_seed(0)
    state = {}
    for name, tensor in model.state_dict().items():
        value = torch.randn(tensor.shape, generator=generator) * 0.05
        if name.endswith("attention.temp"):
            value = 3.0 + value  # peaked attention: the readout's argmax is far from a tie
        state[name] = value if name.endswith("balancing_biases") else value.to(torch.bfloat16)
    torch.save(state, lm_dir / "model.pth")
    (lm_dir / "params.json").write_text(json.dumps({"model": params}))
    (lm_dir / "tokenizer.json").write_text(fixture_tokenizer())
    silence = root / "silence.json"
    silence.write_text(json.dumps({"codec": CODEC, "frame": [11 * i for i in range(10)]}))
    gguf = root / "tiny.gguf"
    convert_gguf.main(["lm", str(lm_dir), "--outtype", "f32", "--silence-frame", str(silence), "-o", str(gguf)])
    return lm_dir, gguf, Zonos3LM.load(lm_dir)


def prompt_rows(seed: int = 3) -> np.ndarray:
    """[BOS][12 text tokens][EOS][3 audio rows]: the text span is rows 0..13."""
    rng = np.random.default_rng(seed)
    rows = np.full((TEXT_LEN + 3, 11), 4097, dtype=np.int64)
    rows[:, 10] = 8196
    rows[0, 10] = 2
    rows[1:TEXT_LEN - 1, 10] = rng.integers(4, 8196, TEXT_LEN - 2)
    rows[TEXT_LEN - 1, 10] = 3
    rows[TEXT_LEN:, :10] = rng.integers(0, 4096, (3, 10))
    return rows


def run_cli(ggml_build: Path, gguf: Path, work: Path, *flags: str, max_frames: int = 10):
    prompt = work / "prompt.npy"
    np.save(prompt, prompt_rows().astype(np.float32))
    cmd = [str(ggml_build / "zonos3-cli"), str(gguf), "--cpu", "--generate", str(prompt), str(work / "codes.npy"),
           "--greedy", "--max", str(max_frames), "--dump-ids", str(work / "ids.npy"),
           "--dump-logits", str(work / "logits.npy"), *flags]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, f"{' '.join(cmd)}\n{done.stderr[-3000:]}"
    return np.load(work / "ids.npy").astype(np.int64), np.load(work / "logits.npy")


def reference_trace(lm: Zonos3LM, ids: np.ndarray, steps: int, *, probe: ProbeSpec | None = None, accents=None):
    """Teacher-forced reference: logits of the prompt's last row, then (logits, readout) of each decode row."""
    n0 = len(prompt_rows())
    cache = LMCache()
    out = lm.forward(torch.from_numpy(ids[:n0]), cache, accents=accents)
    logits, readouts = [out.logits[-1].float().numpy()], []
    for row in ids[n0:n0 + steps]:
        out = lm.forward(torch.from_numpy(row[None]), cache, probe=probe, accents=accents)
        logits.append(out.logits[-1].float().numpy())
        if probe is not None:
            readouts.append(out.probe[0])
    return logits, readouts


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------------------------------------------------------------------------------------- (a) probe


@pytest.mark.parametrize("masked", [False, True], ids=["readout", "masked"])
def test_probe_readout(tiny, ggml_build, tmp_path, masked):
    _, gguf, lm = tiny
    heads_json = tmp_path / "heads.json"
    heads_json.write_text(json.dumps({"frames_per_token": 3.0,
                                      "heads": [{"layer": l, "head": h, "radius": 4} for l, h in HEADS]}))
    flags = ["--probe-heads", str(heads_json), "--dump-probe", str(tmp_path / "probe.npy")]
    windows = WINDOWS if masked else None
    for (layer, head), (lo, hi) in (windows or {}).items():
        flags += ["--probe-window", f"{layer}:{head}:{lo}:{hi}"]
    ids, logits = run_cli(ggml_build, gguf, tmp_path, *flags)
    probe = np.load(tmp_path / "probe.npy")
    steps = len(probe)
    assert steps >= 3, "too few decode steps to compare"
    ref_logits, readouts = reference_trace(lm, ids, steps, probe=ProbeSpec(tuple(HEADS), 0, TEXT_LEN, windows))
    for step, ((argmax, centroid), (ref_argmax, ref_centroid)) in enumerate(zip(probe, readouts)):
        assert int(argmax) == ref_argmax, f"step {step}: argmax {int(argmax)} != reference {ref_argmax}"
        assert abs(float(centroid) - ref_centroid) <= 1e-3, f"step {step}: centroid {centroid} vs {ref_centroid}"
    for step, (got, want) in enumerate(zip(logits, ref_logits)):
        assert cosine(got, want) >= 0.9999, f"row {step}: logits diverge"


# ------------------------------------------------------------------------------------------------- (b) accents


def write_adapter(lm: Zonos3LM, path: Path, seed: int = 5, rank: int = 4) -> None:
    """A zonos-accent-lora/2 adapter over every linear kind, the experts of two layers and two routers."""
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


def test_accent_logits(tiny, ggml_build, tmp_path):
    _, gguf, lm = tiny
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    write_adapter(lm, adapters / "alpha.pt")
    write_adapter(lm, adapters / "beta.pt", seed=9, rank=2)
    accents = tmp_path / "accents.gguf"
    convert_gguf.main(["accents", str(adapters), "-o", str(accents)])
    strength = 0.8
    ids, logits = run_cli(ggml_build, gguf, tmp_path, "--accents", str(accents), "--accent", "beta",
                          "--accent-strength", str(strength), max_frames=6)
    state = AccentState.for_model(lm, load_adapter(adapters / "beta.pt"), strength)
    ref_logits, _ = reference_trace(lm, ids, len(logits) - 1, accents=state)
    plain_logits, _ = reference_trace(lm, ids, len(logits) - 1)
    assert cosine(ref_logits[0], plain_logits[0]) < 0.99999, "the adapter barely moves the logits"
    for step, (got, want) in enumerate(zip(logits, ref_logits)):
        assert cosine(got, want) >= 0.9999, f"row {step}: cos {cosine(got, want)}"


def test_edit_adapter_grows_the_backbone_and_is_no_accent(tiny, ggml_build, tmp_path):
    """`lm --edit-adapter`: a backbone with the /3 adapter's vocabulary whose plain logits on the old ids are the
    ungrown backbone's; the /3 slot of accents.gguf loads on it and is refused as an accent name."""
    lm_dir, gguf, lm = tiny
    adapters = tmp_path / "adapters"
    adapters.mkdir()
    write_adapter(lm, adapters / "plain.pt", seed=7, rank=2)
    write_adapter(lm, tmp_path / "export.pt", seed=11)
    g = torch.Generator().manual_seed(4)
    dim = lm.config.dim
    rows = {f"multi_embedder.embedders.{i}.weight": torch.randn(1, dim, generator=g) * 0.05 for i in range(10)}
    rows |= {"multi_embedder.embedders.10.weight": torch.randn(3, dim, generator=g) * 0.05,
             "multi_output.weight": torch.randn(10, dim, generator=g) * 0.05}
    rows_before = dict.fromkeys(rows, 4098) | {"multi_embedder.embedders.10.weight": 8197}
    edit = grow_adapter(torch.load(tmp_path / "export.pt"), rows_before,
                        {"span_end_id": 4098, "text_extra_tokens": 3, "rows": rows})
    torch.save(edit, adapters / "edit.pt")
    accents = tmp_path / "accents.gguf"
    convert_gguf.main(["accents", str(adapters), "-o", str(accents)])
    silence = tmp_path / "silence.json"
    silence.write_text(json.dumps({"codec": CODEC, "frame": [11 * i for i in range(10)]}))
    grown = tmp_path / "grown.gguf"
    convert_gguf.main(["lm", str(lm_dir), "--outtype", "f32", "--silence-frame", str(silence),
                       "--edit-adapter", str(adapters / "edit.pt"), "-o", str(grown)])
    assert GGUFReader(grown).fields["zonos3.span_end_id"].parts[-1][0] == 4098

    work_plain, work_grown = tmp_path / "plain", tmp_path / "grown"
    work_plain.mkdir(), work_grown.mkdir()
    ids, logits = run_cli(ggml_build, gguf, work_plain, max_frames=6)
    grown_ids, grown_logits = run_cli(ggml_build, grown, work_grown, "--accents", str(accents), "--accent", "plain",
                                      "--accent-strength", "0", max_frames=6)
    assert np.array_equal(ids, grown_ids)
    for got, want in zip(grown_logits, logits):
        got = got.reshape(10, 4099)[:, :4098]
        assert np.array_equal(got, want.reshape(10, 4098))

    refused = subprocess.run([str(ggml_build / "zonos3-cli"), str(grown), "--cpu", "--accents", str(accents),
                              "--accent", "edit", "--generate", str(work_grown / "prompt.npy"),
                              str(work_grown / "x.npy")], capture_output=True, text=True)
    assert refused.returncode != 0 and "speech-edit adapter is not an accent" in refused.stderr
    ungrown = subprocess.run([str(ggml_build / "zonos3-cli"), str(gguf), "--cpu", "--accents", str(accents),
                              "--accent", "plain", "--generate", str(work_plain / "prompt.npy"),
                              str(work_plain / "x.npy")], capture_output=True, text=True)
    assert ungrown.returncode != 0 and "lacks its vocabulary" in ungrown.stderr
