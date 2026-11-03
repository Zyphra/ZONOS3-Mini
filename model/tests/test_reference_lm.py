"""zonos3.reference.lm on a tiny random model (GPU-free): checkpoint names, strict loading, cache semantics."""

from __future__ import annotations

import json

import pytest
import torch
from zonos3.reference.lm import LMCache, ProbeSpec, Zonos3LM, Zonos3LMConfig

N_CB, CODEBOOK, TEXT_VOCAB = 10, 16, 20

TINY = {
    "n_layers": 2, "dim": 64, "head_dim": 16, "n_heads": 4, "n_kv_heads": 2, "norm_eps": 1e-5,
    "rope_theta": 10000.0, "max_seqlen": 256, "loss_softcap": 15.0, "ffn_dim_multiplier": 2.0, "multiple_of": 32,
    "moe_ffn_dim_multiplier": 1.0, "moe_multiple_of": 32, "moe_alpha": 0.5, "moe_n_experts": 4,
    "n_codebooks": N_CB, "codebook_size": CODEBOOK, "codebook_sizes": [CODEBOOK] * N_CB, "eoa_id": CODEBOOK,
    "audio_pad_id": CODEBOOK + 1, "text_vocab": TEXT_VOCAB, "speaker_embedding_dim": 8,
    "attention_pattern": "gdn1_global1",  # layer 0 GDN, layer 1 global attention (MoE)
    "residual_type": "full_attnres", "use_xsa": False, "speaker_enabled": True, "condition_tokens_enabled": True,
    "moe_router_type": "linear", "moe_router_use_eda": False, "moe_balancing_strategy": "quantile",
    "moe_router_topk": 2, "moe_start_from_layer": 1, "moe_end_from_layer": 0,
    "condition_count": 3, "condition_n_buckets": 4, "condition_names": ["loudness", "fmax", "speaking_rate"],
    "condition_edges": {"loudness": [-30.0, -20.0, -10.0], "fmax": [4e3, 8e3, 12e3], "speaking_rate": [10.0, 15.0, 20.0]},
}


def tiny_model(seed: int = 0, **overrides) -> Zonos3LM:
    torch.manual_seed(seed)
    model = Zonos3LM(Zonos3LMConfig.from_params({"model": {**TINY, **overrides}}))
    generator = torch.Generator().manual_seed(seed)
    for name, param in model.named_parameters():
        noise = torch.randn(param.shape, generator=generator)
        if name.endswith(("norm.weight", "rmsnorm_eda.weight", "out_weight")):
            param.data = 1.0 + 0.1 * noise
        elif name.endswith("balancing_biases"):
            param.data = 0.1 * noise
        else:
            param.data = 0.2 * noise
    return model.eval()


def prompt(rows: int = 12, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ids = torch.full((rows, N_CB + 1), CODEBOOK + 1, dtype=torch.long)
    ids[:, N_CB] = torch.randint(4, TEXT_VOCAB, (rows,), generator=g)
    ids[0, N_CB] = TEXT_VOCAB  # speaker row
    ids[1:4, N_CB] = TEXT_VOCAB  # condition rows
    ids[-4:, :N_CB] = torch.randint(0, CODEBOOK, (4, N_CB), generator=g)  # audio rows
    return ids


def conditioning():
    return {"speaker_rows": {0: torch.linspace(-1, 1, 8)}, "condition_rows": {1: 2, 2: 4, 3: 0}}


def test_strict_state_dict_round_trip_and_load(tmp_path):
    model = tiny_model()
    torch.save(model.state_dict(), tmp_path / "model.pth")
    (tmp_path / "params.json").write_text(json.dumps({"model": TINY}))
    loaded = Zonos3LM.load(tmp_path)
    for (name, a), (other, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert name == other and torch.equal(a, b)
    ids = prompt()
    assert torch.equal(model(ids, LMCache(), **conditioning()).logits, loaded(ids, LMCache(), **conditioning()).logits)
    state = model.state_dict()
    state["extra.weight"] = torch.zeros(1)
    torch.save(state, tmp_path / "model.pth")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        Zonos3LM.load(tmp_path)


def test_unsupported_architecture_is_refused():
    with pytest.raises(ValueError, match="moe_router_use_eda"):
        Zonos3LMConfig.from_params(dict(TINY, moe_router_use_eda=True))
    with pytest.raises(ValueError, match="speaking_rate_num_buckets"):
        Zonos3LMConfig.from_params(dict(TINY, speaking_rate_num_buckets=21))
    # extra text rows (edit markers / phoneme tokens) and the span-end token are part of the architecture
    config = Zonos3LMConfig.from_params(dict(TINY, text_extra_tokens=3, span_end_id=CODEBOOK + 2))
    assert config.text_rows == TEXT_VOCAB + 4 and config.audio_vocab == CODEBOOK + 3

def test_per_table_bucket_counts_size_each_condition_table():
    rate30 = dict(TINY, condition_bucket_counts=[4, 4, 6],
                  condition_edges={**TINY["condition_edges"], "speaking_rate": [8.0, 10.0, 12.0, 15.0, 20.0]})
    model = Zonos3LM(Zonos3LMConfig.from_params(rate30))
    assert [e.weight.shape[0] for e in model.condition_embeddings] == [5, 5, 7]
    # the edges of a table follow that table's count, not the shared condition_n_buckets
    with pytest.raises(ValueError, match=r"condition_edges\.speaking_rate"):
        Zonos3LMConfig.from_params(dict(rate30, condition_edges=TINY["condition_edges"]))



@pytest.mark.parametrize("k", [1, 3, 5])
def test_prefill_equals_prefill_plus_decode_steps(k):
    model = tiny_model()
    ids = prompt()
    full = model(ids, LMCache(), **conditioning()).logits
    cache = LMCache()
    parts = [model(ids[: len(ids) - k], cache, **conditioning()).logits]
    for row in range(len(ids) - k, len(ids)):
        parts.append(model(ids[row : row + 1], cache, **conditioning()).logits)
    assert cache.position == len(ids)
    torch.testing.assert_close(torch.cat(parts), full, rtol=1e-5, atol=1e-5)


def test_fork_isolates_caches():
    model = tiny_model()
    ids = prompt()
    cache = LMCache()
    model(ids[:-2], cache, **conditioning())
    fork = cache.fork()
    snapshot = cache.fork()
    a = model(ids[-2:-1], fork).logits
    model(ids[-1:], fork)
    assert cache.position == snapshot.position == len(ids) - 2 and fork.position == len(ids)
    for layer, (k, v) in cache.kv.items():
        assert torch.equal(k, snapshot.kv[layer][0]) and torch.equal(v, snapshot.kv[layer][1])
    for layer, state in cache.gdn_state.items():
        assert torch.equal(state, snapshot.gdn_state[layer])
        assert torch.equal(cache.conv_state[layer], snapshot.conv_state[layer])
    torch.testing.assert_close(model(ids[-2:-1], cache).logits, a, rtol=0, atol=0)


def test_hidden_deltas_and_conditioning_change_the_logits():
    model = tiny_model()
    ids = prompt()
    base = model(ids, LMCache(), **conditioning()).logits
    steered = model(ids, LMCache(), hidden_deltas={3: torch.ones(64)}, **conditioning()).logits
    plain = model(ids, LMCache()).logits
    assert not torch.allclose(base, steered) and not torch.allclose(base, plain)


def test_probe_readout_and_open_windows():
    model = tiny_model()
    ids = prompt()
    cache = LMCache()
    model(ids[:-1], cache)
    heads = ((1, 0), (1, 3))
    probe = ProbeSpec(heads, text_start=4, text_len=4)
    fork = cache.fork()
    out = model(ids[-1:], cache, probe=probe)
    (pos, centroid), = out.probe
    assert 0 <= pos < 4 and 0.0 <= centroid <= 3.0
    # a window spanning the whole text masks nothing
    opened = model(ids[-1:], fork, probe=ProbeSpec(heads, 4, 4, dict.fromkeys(heads, (0, 3))))
    torch.testing.assert_close(opened.logits, out.logits, rtol=1e-5, atol=1e-5)
    assert opened.probe[0][0] == pos


def test_capture_names_follow_the_golden_dump():
    model = tiny_model()
    capture = {}
    model(prompt(), LMCache(), capture=capture, **conditioning())
    for name in ("emb_sum", "emb_cond", "emb_norm", "attnres_attn_0", "gdn_state_0", "gdn_conv_state_0",
                 "attn_out_1", "router_logits_1", "ffn_out_1", "layer_x_1", "attnres_final", "out_norm", "logits"):
        assert name in capture
    assert capture["logits"].shape == (12, N_CB, CODEBOOK + 2)


def test_edit_fill_generation_stops_on_span_end_and_is_seeded():
    from zonos3.reference.edit import generate_edit_fill
    from zonos3.reference.synth import Sampling, prompt_builder

    # byte-text checkpoint with two edit marker rows and the span-end token
    model = tiny_model(text_vocab=448, text_extra_tokens=2, span_end_id=CODEBOOK + 2, max_seqlen=512)
    codes = torch.randint(0, CODEBOOK, (12, N_CB), generator=torch.Generator().manual_seed(3))
    text_rows = prompt_builder(model).build("hi there")
    runs = [generate_edit_fill(model, codes, [(4, 3)], text_rows, Sampling(min_frames=2), seed=7, max_tokens=24)
            for _ in range(2)]
    (fills, eos), (again, eos_again) = runs
    assert len(fills) == 1 and fills[0].shape[1] == N_CB and bool((fills[0] < CODEBOOK).all())
    assert torch.equal(fills[0], again[0]) and eos == eos_again
    assert eos[0] is None or eos[0] >= 2  # min_frames masks the span-end before frame 2


def test_laci_rollback_keeps_the_rows_before_the_failure(monkeypatch):
    import zonos3.reference.synth as synth
    from zonos3.tts import plan as P
    from zonos3.tts.laci import Detector, Failure, LaciConfig, LaciHeads
    from zonos3.tts.prompt import TTSPromptConfig, silence_rows_from_codes

    model = tiny_model()
    pad, text = CODEBOOK + 1, TEXT_VOCAB
    rows = torch.full((3 + 18, N_CB + 1), pad, dtype=torch.int32)
    rows[:, N_CB] = torch.tensor([text] * 3 + [2] + list(range(4, 20)) + [3], dtype=torch.int32)
    config = TTSPromptConfig(n_codebooks=N_CB, codebook_sizes=(CODEBOOK,) * N_CB, audio_pad_id=pad, text_vocab=448)
    prefill = silence_rows_from_codes(torch.arange(N_CB), 2, config)
    prefill[:, N_CB] = text
    ids = torch.cat((rows, prefill))
    attempts = []

    class OneSkip(synth.LaciSession):
        """The first attempt reports one skip at its 6th decode step (rolled back to the 3rd generated row) and nothing
        else; the regeneration runs the real detector with no retry left (its findings are only recorded)."""

        def on_step(self, probe, step, pos, m):
            if self.attempts:
                return super().on_step(probe, step, pos, m)
            if self.detector is None:
                self.text_len = probe.text_len
                self.detector = Detector(probe.text_len, self.heads.frames_per_token, self.cfg, self.n_prefill)
            if step == 6:
                self.pending_failure = Failure("skip", self.attempt_start + 6, self.n_prefill + 2, self.n_prefill + 2, 1, 2)
                probe.stop = True

        def finish_attempt(self):
            attempts.append([r.clone() for r in self.stream[self.n_prefill:]])
            return super().finish_attempt()

    monkeypatch.setattr(synth, "LaciSession", OneSkip)
    monkeypatch.setattr(synth, "LaciConfig", lambda: LaciConfig(max_retries=1))
    heads = LaciHeads(((1, 0), (1, 2)), {(1, 0): 3, (1, 2): 3}, 1.5)
    plan = P.PromptPlan(
        request=P.parse_request({"text": "x"}), text="x", language="en_us",
        cond=P.Branch(ids, condition_values=[4, 4, 4], condition_token_start=0), twin=None, cfg_kind=None,
        cfg_scale=1.0, reference_prefix_length=0, prefill_rows=prefill, context_codes=None, sources=(),
        hidden_delta=None, schedule=None, follow_tokens=None, follow_gate=None, accent=None, max_tokens=12, seed=3,
        laci="en_us",
    )
    assets = synth.ReferenceAssets(laci_heads={"en_us": heads})
    out, eos, record = synth._laci(model, plan, assets, synth.Sampling(), None, None, 3, 18)
    assert record["attempts"] == 1 and record["failures"][0]["kind"] == "skip"
    first = attempts[0]
    assert len(first) == 7  # the stop step's row ends the attempt
    assert all(torch.equal(a, b) for a, b in zip(out[:2], first[:2]))  # rows before the rollback row are kept
    failure = record["failures"][0]
    assert failure["retry_seed"] == 3 + 1000 and failure["rollback_row"] == 2 + 2 and 2 < len(out) <= 12
