"""Graph-feature parity of the ggml backbone against the plain-PyTorch reference (zonos3.reference.lm.Zonos3LM):

(a) the decode-step alignment probe (`zonos3-cli --probe-heads/--probe-window/--dump-probe`) against
    `Zonos3LM.forward(probe=ProbeSpec)` teacher-forced on the rows ggml generated: identical argmax per step, centroid
    within 1e-3, plus the logits (the masked heads' recomputed attention feeds the following layers);
(b) accent adapters (`--accents/--accent/--dump-logits`) against `zonos3.reference.accents.AccentState`: cos >= 0.9999
    on the prompt's last row and on every decode step;
(c) emotion directions (`graph-features --emotion`) against zonos3.tts.emotion.apply_emotion / emotion_hidden_delta
    with the sgl server's calibrated weight fold: equal within 1e-6, raw and attnres_source sets.

The tiny backbone is a random Zonos3LM with the dimensions the ggml loader indexes into (10 x 4096 codebooks, the
8196-token SuperBPE vocabulary, 128-wide GDN heads) and global layers 3 and 7, converted with `tools/convert_gguf.py lm`.
Accent adapters and direction sets go through `convert_gguf.py accents|directions` as well.

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
from zonos3.models.accent_format import FORMAT_EXPERTS, load_adapter  # noqa: E402
from zonos3.reference.accents import AccentState  # noqa: E402
from zonos3.reference.lm import LMCache, ProbeSpec, Zonos3LM, Zonos3LMConfig  # noqa: E402
from zonos3.tts.emotion import EmotionCalibration, EmotionDirections, apply_emotion, emotion_hidden_delta  # noqa: E402

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
    subprocess.run(cmd, check=True, capture_output=True, text=True)
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


# ------------------------------------------------------------------------------------------------- (c) emotion


def make_directions(tmp_path: Path, space: str, dim: int, source_index: int = -1) -> tuple[Path, Path]:
    """(manifest dir, convert_gguf GGUF) of a random direction set; the dummy writer's GGUF must load identically."""
    manifest, dummy = tmp_path / f"{space}-dir", tmp_path / f"{space}-dummy.gguf"
    cmd = [sys.executable, str(GGML / "scripts" / "make-dummy-directions.py"), "-o", str(dummy),
           "--manifest-dir", str(manifest), "--space", space, "--dim", str(dim)]
    if source_index >= 0:
        cmd += ["--source-index", str(source_index)]
    subprocess.run(cmd, check=True, capture_output=True)
    converted = tmp_path / f"{space}.gguf"
    convert_gguf.main(["directions", str(manifest), "-o", str(converted)])
    return manifest, converted


def python_emotion(directions, calibration, request: dict) -> dict:
    """sgl _apply_request_emotion's math on one request (strict, optional calibrated fold)."""
    sliders = dict(request.get("sliders", []))
    valence, arousal = request.get("valence", 0.0), request.get("arousal", 0.0)
    speaker = request.get("speaker") or None
    if request.get("calibrated") and calibration is not None:
        sliders = {e: float(w) * calibration.pole_strength(speaker, e, float(w)) for e, w in sliders.items()}
        valence = float(valence) * calibration.pole_strength(speaker, "valence", float(valence))
        arousal = float(arousal) * calibration.pole_strength(speaker, "arousal", float(arousal))
    kwargs = dict(sliders=sliders, valence=valence, arousal=arousal, strength=request.get("strength", 1.0),
                  strict=request.get("strict", False))
    try:
        hidden = emotion_hidden_delta(directions, **kwargs)
        embedding = None
        if directions.space not in ("proj", "attnres_source") and request.get("base"):
            embedding = apply_emotion(torch.tensor(request["base"], dtype=torch.float32), directions=directions,
                                      preserve_norm=request.get("preserve_norm", True), **kwargs)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"embedding": None if embedding is None else embedding.tolist(),
            "hidden": None if hidden is None else hidden.tolist()}


REQUESTS = [
    {"sliders": [["happy", 0.7], ["angry", -0.4]], "valence": 0.3, "strength": 1.3},
    {"sliders": [["sad", 1.0]], "arousal": -0.6, "strength": 0.5, "preserve_norm": False},
    {"sliders": [["happy", 1.0], ["sad", 0.5]], "valence": -0.8, "arousal": 0.4, "calibrated": True,
     "speaker": "default_0123456789abcdef"},
    {"sliders": [["angry", 0.9]], "valence": 0.6, "calibrated": True, "speaker": "unknown_voice"},
    {"sliders": [["happy", 0.0]], "strength": 2.0},
    {"sliders": [["joy", 1.0]], "strict": True},
    {"sliders": [["joy", 1.0], ["sad", 0.2]], "strict": False},
]


@pytest.mark.parametrize("space,dim,source", [("raw", 192, -1), ("attnres_source", 64, 5)])
def test_emotion_delta(ggml_build, tmp_path, space, dim, source):
    manifest, converted = make_directions(tmp_path, space, dim, source)
    directions = EmotionDirections.load(manifest)
    calibration = EmotionCalibration.load(manifest)
    base = (np.sin(np.arange(dim) * 0.37) + 0.2).astype(np.float32).tolist()
    exe = ggml_build / "graph-features"
    for i, request in enumerate(REQUESTS):
        request = {**request, "base": base, "strict": request.get("strict", True)}
        path = tmp_path / f"request{i}.json"
        path.write_text(json.dumps(request))
        results = []
        for gguf in (converted, tmp_path / f"{space}-dummy.gguf"):
            out = subprocess.run([str(exe), "--emotion", str(gguf), "--request", str(path)],
                                 capture_output=True, text=True, check=True)
            results.append(json.loads(out.stdout))
        assert results[0] == results[1], "convert_gguf and make-dummy-directions GGUFs disagree"
        got, want = results[0], python_emotion(directions, calibration, request)
        if "error" in want:
            assert got.get("error") == want["error"], f"request {i}"
            continue
        assert "error" not in got, got.get("error")
        for key in ("embedding", "hidden"):
            assert (got[key] is None) == (want[key] is None), f"request {i}: {key}"
            if want[key] is not None:
                np.testing.assert_allclose(got[key], want[key], rtol=0, atol=1e-6, err_msg=f"request {i}: {key}")
