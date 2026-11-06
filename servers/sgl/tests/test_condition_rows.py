"""Condition rows (loudness / fmax / speaking-rate buckets) of the cond3 finetunes in the runtime (2026-10-16):
prompt rows, request resolution, the scheduler's per-span gather and the offline prompt layout."""
import json
from types import SimpleNamespace

import pytest
import torch

api_server = pytest.importorskip("zonos3_sgl.server.api_server")
from zonos3.server.schemas import GenerateRequest, SpeechRequest  # noqa: E402
from zonos3.tts import plan  # noqa: E402
from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig  # noqa: E402
from zonos3_sgl.core import TTSBatch, TTSReq  # noqa: E402
from zonos3_sgl.message.tts import TTSSamplingParams  # noqa: E402
from zonos3_sgl.scheduler.scheduler import TTSScheduler, _speaking_rate_buckets  # noqa: E402

EDGES = {
    "loudness": [-29.002, -26.861, -25.366, -23.607, -21.916, -20.505, -19.38, -18.151, -16.297],
    "fmax": [7578.125, 8000.0, 8000.01, 11062.5, 12445.94, 15328.125, 15773.438, 16623.633, 18758.496],
    "speaking_rate": [10.117, 12.123, 13.286, 14.25, 15.14, 16.079, 17.094, 18.489, 21.208],
}
# the default LM's speaking-rate row (zonos3-rate30-20k-r2grpo, condition_bucket_counts [10, 10, 30])
RATE30_EDGES = [6.546, 8.182, 9.273, 10.31, 11.054, 11.667, 12.193, 12.656, 13.083, 13.476, 13.846, 14.196, 14.53,
                14.887, 15.193, 15.517, 15.865, 16.208, 16.579, 16.972, 17.411, 17.898, 18.434, 19.069, 19.818, 20.816,
                22.286, 24.722, 29.25]


def cfg(enabled=True, edges=EDGES, counts=(10, 10, 10)):
    return SimpleNamespace(model_config=SimpleNamespace(
        condition_tokens_enabled=enabled, condition_bucket_counts=counts if enabled else (), condition_count=3,
        condition_names=("loudness", "fmax", "speaking_rate"), condition_edges=edges,
        condition_units={"loudness": "LUFS", "fmax": "Hz", "speaking_rate": "utf-8 bytes per second"},
        speaker_reference_prefix_enabled=True, speaker_enabled=True, speaker_embedding_dim=256, audio_codec="opus_dual_3n_600k_sched600k_r4096_step600000"))


def cfg_rate30():
    return cfg(edges={**EDGES, "speaking_rate": RATE30_EDGES}, counts=(10, 10, 30))


def resolve(config, buckets, values):
    """zonos3.tts.plan.resolve_condition_buckets on the server's rules context."""
    return plan.resolve_condition_buckets(api_server._RulesContext(config), buckets, values)


def test_condition_rows_are_all_pad_rows_with_the_text_pad():
    builder = TTSPromptBuilder(TTSPromptConfig(n_codebooks=10, audio_pad_id=4097, text_vocab=8196))
    rows = builder.condition_rows(3)
    assert rows.shape == (3, 11) and rows.dtype == torch.int32
    assert bool((rows[:, :10] == 4097).all()) and bool((rows[:, 10] == 8196).all())
    assert torch.equal(rows[:1], builder.speaker_slot())          # the same all-pad row the reserved speaker slot uses


def test_capabilities_describe_the_rows_and_models_without_them_report_none():
    spec = api_server._condition_spec(cfg())
    assert spec["names"] == ["loudness", "fmax", "speaking_rate"]
    assert spec["bucket_counts"] == spec["unknown_ids"] == {"loudness": 10, "fmax": 10, "speaking_rate": 10}
    assert spec["edges"]["fmax"][1] == 8000.0 and spec["units"]["loudness"] == "LUFS"
    assert api_server._condition_spec(cfg(enabled=False)) is None
    rate30 = api_server._condition_spec(cfg_rate30())
    assert rate30["bucket_counts"] == rate30["unknown_ids"] == {"loudness": 10, "fmax": 10, "speaking_rate": 30}


def test_explicit_ids_raw_values_and_unknown_defaults():
    c = cfg()
    assert resolve(c, None, None) == [10, 10, 10]
    assert resolve(c, {"loudness": 3}, None) == [3, 10, 10]
    assert resolve(c, [0, None, 9], None) == [0, 10, 9]
    # raw values: bucket i = [edge_{i-1}, edge_i) like the training loader (bisect_right)
    assert resolve(c, None, {"loudness": -40.0}) == [0, 10, 10]
    assert resolve(c, None, {"loudness": -29.002}) == [1, 10, 10]
    assert resolve(c, None, {"fmax": 8000.0}) == [10, 2, 10]      # the 16 kHz-source atom bucket
    assert resolve(c, None, {"fmax": 7999.9}) == [10, 1, 10]
    assert resolve(c, None, {"speaking_rate": 99.0}) == [10, 10, 9]
    # ids win over raw values per name
    assert resolve(c, {"fmax": 5}, {"fmax": 100.0, "speaking_rate": 14.3}) == [10, 5, 4]


def test_bad_requests_are_refused():
    c = cfg()
    for bad in ({"volume": 1}, {"loudness": 11}, {"loudness": -1}):
        with pytest.raises(ValueError):
            resolve(c, bad, None)
    with pytest.raises(ValueError):
        resolve(c, [1, 2], None)
    with pytest.raises(ValueError):
        resolve(c, None, {"fmax": float("nan")})
    with pytest.raises(ValueError):
        resolve(cfg(enabled=False), {"loudness": 1}, None)
    assert resolve(cfg(enabled=False), None, None) is None
    with pytest.raises(ValueError):            # no edges for a name -> raw values cannot be mapped
        resolve(cfg(edges={}), None, {"loudness": -20.0})


def test_rate30_checkpoint_has_its_own_speaking_rate_count():
    c = cfg_rate30()
    assert resolve(c, None, None) == [10, 10, 30]
    assert resolve(c, {"speaking_rate": 29}, None) == [10, 10, 29]
    assert resolve(c, None, {"speaking_rate": 99.0}) == [10, 10, 29]
    with pytest.raises(ValueError):
        resolve(c, {"speaking_rate": 31}, None)
    # LACI's speaking-rate scale (duration targets) spans the row's 30 buckets; unknown is 30, not 10
    rates = _speaking_rate_buckets(c.model_config, (10, 10, 30))
    assert rates.index == 2 and rates.n_buckets == 30 and rates.row(10) == 10 and rates.row(30) == 30
    assert _speaking_rate_buckets(cfg(enabled=False).model_config, ()) is None


def make_req(values, start, cached_len, device_len):
    ids = torch.full((device_len, 11), 4097, dtype=torch.int32)
    ids[:, 10] = 8196
    req = TTSReq(input_ids=ids, table_idx=0, cached_len=cached_len, output_len=4, uid=1,
                 sampling_params=TTSSamplingParams(), cache_handle=None, n_codebooks=10, audio_pad_id=4097, stop_id=4096,
                 condition_values=values, condition_token_start=start)
    return req


def scheduler_stub(count=3, unknowns=(10, 10, 10)):
    s = TTSScheduler.__new__(TTSScheduler)
    s.condition_count = count
    s.condition_unknowns = unknowns if count else ()
    s.device = torch.device("cpu")
    return s


def test_scheduler_gathers_only_the_rows_inside_each_extend_span():
    s = scheduler_stub()
    # request A: rows at 5..7 inside a whole prefill of 20; request B: rows at 0..2 but the span starts at 2 (chunked
    # prefill cut inside the rows); request C: no rows (a model without them never gets here)
    a = make_req([3, 5, 7], 5, 0, 20)
    b = make_req([1, 2, 10], 0, 2, 12)
    batch = TTSBatch(reqs=[a, b], phase="prefill")
    s._prepare_condition_rows(batch)
    rows = batch.condition_rows
    assert len(rows) == 3
    assert rows[0] is not None and rows[0][0].tolist() == [3] and rows[0][1].tolist() == [5]          # B's loudness row is before its span
    assert rows[1][0].tolist() == [5] and rows[1][1].tolist() == [6]
    assert rows[2][0].tolist() == [7, 10] and rows[2][1].tolist() == [7, 20 + 0]                     # B's rate row at its packed offset
    # decode steps carry no rows
    a.cached_len, a.device_len = 20, 21
    batch = TTSBatch(reqs=[a], phase="decode")
    s._prepare_condition_rows(batch)
    assert batch.condition_rows is None
    # a model without condition rows never sets them
    batch = TTSBatch(reqs=[make_req([3, 5, 7], 5, 0, 20)], phase="prefill")
    scheduler_stub(count=0)._prepare_condition_rows(batch)
    assert batch.condition_rows is None



# ---------------------------------------------------------------- flag rows (the encode-start finetunes, 2026-10-17)
def cfg_flag():
    return SimpleNamespace(model_config=SimpleNamespace(
        condition_tokens_enabled=True, condition_bucket_counts=(10, 10, 10, 10), condition_count=4,
        condition_names=("loudness", "fmax", "speaking_rate", "encode_start"), condition_edges=EDGES,
        condition_units={"loudness": "LUFS", "fmax": "Hz", "speaking_rate": "utf-8 bytes per second", "encode_start": "flag"},
        condition_flags={"encode_start": {"values": [0, 1], "labels": {"0": "mid-file crop", "1": "genuine encode start"}, "max_frames": 0}},
        speaker_reference_prefix_enabled=True, speaker_enabled=True, speaker_embedding_dim=192, audio_codec="opus_dual_3n_600k_sched600k_r4096_step600000"))


def test_encode_start_flag_row_is_a_fourth_row_with_two_values():
    spec = api_server._condition_spec(cfg_flag())
    assert spec["names"] == ["loudness", "fmax", "speaking_rate", "encode_start"] and spec["count"] == 4
    assert spec["flags"] == {"encode_start": cfg_flag().model_config.condition_flags["encode_start"]}
    assert spec["layout"] == "[reference prefix | speaker row][loudness][fmax][speaking_rate][encode_start] BOS text EOS"
    assert spec["units"]["encode_start"] == "flag" and "encode_start" not in spec["edges"]
    assert resolve(cfg_flag(), {"encode_start": 1}, None) == [10, 10, 10, 1]
    assert resolve(cfg_flag(), {"encode_start": 0, "loudness": 3}, None) == [3, 10, 10, 0]
    assert resolve(cfg_flag(), None, {"encode_start": True}) == [10, 10, 10, 1]
    assert resolve(cfg_flag(), None, {"encode_start": 0}) == [10, 10, 10, 0]
    assert resolve(cfg_flag(), [None, None, None, 1], None) == [10, 10, 10, 1]
    assert resolve(cfg_flag(), None, None) == [10, 10, 10, 10]
    assert resolve(cfg_flag(), {"encode_start": 10}, None) == [10, 10, 10, 10]        # explicit unknown
    with pytest.raises(Exception):
        resolve(cfg_flag(), {"encode_start": 2}, None)
    # the three-row checkpoints are untouched: no flags, the old layout
    old = api_server._condition_spec(cfg())
    assert old["flags"] == {} and old["layout"].endswith("[speaking_rate] BOS text EOS") and old["count"] == 3


def test_request_fields_round_trip_through_the_pydantic_models():
    req = GenerateRequest(text="hi", condition_buckets={"loudness": 2, "fmax": None}, condition_values=[None, 8000.0, None])
    assert req.condition_buckets == {"loudness": 2, "fmax": None} and req.condition_values == [None, 8000.0, None]
    assert SpeechRequest(model="zonos3", input="hi").condition_buckets is None
    assert json.loads(json.dumps(api_server._condition_spec(cfg())))["layout"].startswith("[reference prefix")
