"""CPU tokenizer and scheduler coverage for ordinary text and general CFG."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from zonos3.tokenizer import textnorm
from zonos3.tts.prompt import BYTE_TEXT_VOCAB_SIZE, TTSPromptBuilder, TTSPromptConfig
from zonos3_sgl.distributed.info import DistributedInfo
from zonos3_sgl.message import TTSUserMsg
from zonos3_sgl.message.tts import TTSSamplingParams, TTSTokenizeMsg
from zonos3_sgl.scheduler.scheduler import TTSScheduler
from zonos3_sgl.tokenizer import server as tokenizer_server

C = 2  # codebooks
PAD = 1025
RATE, QUALITY, BACKGROUND, ACCURATE = 2, (2,), 2, 1
TEXT_VOCAB = BYTE_TEXT_VOCAB_SIZE + RATE + sum(QUALITY) + BACKGROUND + ACCURATE
CONDITIONS, UNKNOWNS = 3, (10, 10, 30)  # per-table unknown ids (condition_bucket_counts)
UNKNOWN = UNKNOWNS[1]
SILENCE_FRAMES = 3


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setattr("zonos3_sgl.distributed.info._TP_INFO", DistributedInfo(0, 1))


def _config():
    return TTSPromptConfig(
        n_codebooks=C,
        audio_pad_id=PAD,
        text_vocab=TEXT_VOCAB,
        speaking_rate_num_buckets=RATE,
        quality_bucket_counts=QUALITY,
        speaker_background_num_buckets=BACKGROUND,
        accurate_mode_num_buckets=ACCURATE,
    )


def _silence_rows(frames, config):
    rows = torch.full((frames, config.n_codebooks + 1), 7, dtype=torch.int32)
    rows[:, config.n_codebooks] = config.text_vocab
    return rows


class _Vocoder:
    """The tokenizer's view of a matched codec: silence prefill rows, nothing decoded."""

    silence_prefill_supported = True
    hop_length = 4096

    def __init__(self, **_):
        pass

    def warm_prefix_stream(self):
        pass

    def silence_prefill_frames_for(self, requested_frames):
        return SILENCE_FRAMES if requested_frames is None else int(requested_frames)

    def silence_prompt_rows(self, frames, config):
        return _silence_rows(frames, config)


def _message(**kwargs):
    fields = {
        "uid": 7,
        "text": "Hello there.",
        "language": "en_us",
        "sampling_params": TTSSamplingParams(max_tokens=20),
        "speaker_embedding": torch.tensor([1.0, 2.0]),
        "condition_buckets": [3, UNKNOWN, 5],
    }
    fields.update(kwargs)
    return TTSTokenizeMsg(**fields)


def _tokenize(monkeypatch, tmp_path, messages, *, normalize=False):
    """Run the real tokenize_worker loop on ``messages`` (with the real text normalizer when ``normalize``); return the
    TTSUserMsgs sent to the scheduler."""
    sent = []

    class Push:
        def __init__(self, addr, create, encoder):
            self.addr = addr

        def put(self, msg):
            if self.addr == "backend":
                sent.extend(msg.data if hasattr(msg, "data") else [msg])

    pending = list(messages)

    class Pull:
        def __init__(self, addr, create, decoder):
            pass

        def get(self):
            if not pending:
                raise KeyboardInterrupt
            return pending.pop(0)

        def empty(self):
            return not pending

    monkeypatch.setattr(tokenizer_server, "ZmqPushQueue", Push)
    monkeypatch.setattr(tokenizer_server, "ZmqPullQueue", Pull)
    monkeypatch.setattr("zonos3_sgl.tokenizer.vocoder.TTSVocoderManager", _Vocoder)
    monkeypatch.setattr(textnorm, "normalization_enabled", lambda: normalize)
    tokenizer_server.tokenize_worker(
        addr="tokenizer",
        create=True,
        backend_addr="backend",
        frontend_addr="frontend",
        local_bs=1,
        n_codebooks=C,
        audio_pad_id=PAD,
        audio_codec="matched",
        text_vocab=TEXT_VOCAB,
        speaking_rate_num_buckets=RATE,
        quality_bucket_counts=list(QUALITY),
        quality_buckets=[1],
        speaker_background_num_buckets=BACKGROUND,
        accurate_mode_num_buckets=ACCURATE,
        condition_count=CONDITIONS,
        condition_bucket_counts=UNKNOWNS,
        text_tokenizer_spec=None,
    )
    return sent


def _scheduler(max_seq_len=200, prefill_budget=1000, tables=4):
    class Table:
        def __init__(self):
            self.token_pool = torch.zeros((tables, max_seq_len, C + 1), dtype=torch.int32)
            self.free_indices = list(range(tables))

        @property
        def available_size(self):
            return len(self.free_indices)

        def allocate(self):
            return self.free_indices.pop(0)

    class Cache:
        def allocate_new_handle(self):
            return object()

    s = TTSScheduler.__new__(TTSScheduler)
    s.device = torch.device("cpu")
    s.engine = SimpleNamespace(max_seq_len=max_seq_len)
    s.table_manager = Table()
    s.token_pool = s.table_manager.token_pool
    s.cache_manager = Cache()
    s.waiting_reqs = []
    s.prefill_budget = prefill_budget
    s.n_codebooks, s.audio_pad_id, s.text_vocab = C, PAD, TEXT_VOCAB
    s.eoa_id, s.span_end_id, s.eoa_policy = 1024, None, "any"
    s.laci_heads, s.laci_requests = {}, {}
    s.speaking_rate_num_buckets, s.quality_bucket_counts = RATE, list(QUALITY)
    s.speaker_background_num_buckets, s.accurate_mode_num_buckets = BACKGROUND, ACCURATE
    s.condition_count, s.condition_unknowns = CONDITIONS, UNKNOWNS
    return s


def test_tokenizer_preserves_literal_angle_text(monkeypatch, tmp_path):
    text = "Hello <laugh/> there"
    (user,) = _tokenize(monkeypatch, tmp_path, [_message(text=text)])
    builder = TTSPromptBuilder(_config())
    expected = torch.cat((builder.speaker_slot(), builder.condition_rows(CONDITIONS),
                          builder.build(text, quality_buckets=[1]),
                          _silence_rows(SILENCE_FRAMES, builder.config)))
    assert torch.equal(user.input_ids, expected)


def _user(monkeypatch, tmp_path, **kwargs):
    (user,) = _tokenize(monkeypatch, tmp_path, [_message(**kwargs)])
    return user


def test_controls_cfg_mixes_raw_logits_and_synchronizes_tokens(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    user = _user(monkeypatch, tmp_path, steering_delta=torch.ones(4), sampling_params=TTSSamplingParams(controls_cfg_scale=1.5, max_tokens=20))
    s = _scheduler(max_seq_len=len(user.input_ids) + 2 + 6)
    s._process_one_msg(user)
    batch = s._schedule_prefill()
    assert batch is not None and len(batch.reqs) == 2
    conditional, twin = batch.reqs
    assert conditional.cfg_twin is twin and twin.is_cfg_uncond and not conditional.is_cfg_uncond
    assert conditional.cfg_scale == 1.5 and twin.rng is None
    assert torch.equal(conditional.speaker_embedding, twin.speaker_embedding)
    assert conditional.speaker_token_position == twin.speaker_token_position == 0
    assert conditional.condition_values == twin.condition_values == [3, UNKNOWN, 5]
    assert conditional.condition_token_start == twin.condition_token_start
    # the scheduler adds the same background / accurate-mode markers to both branches
    added = len(conditional.input_ids) - len(user.input_ids)
    assert added == 2 and len(twin.input_ids) - len(user.input_ids) == added
    assert conditional.condition_token_start == 1 + added
    assert torch.equal(conditional.input_ids[: 1 + added + CONDITIONS], twin.input_ids[: 1 + added + CONDITIONS])
    assert torch.equal(conditional.input_ids[-SILENCE_FRAMES:], twin.input_ids[-SILENCE_FRAMES:])
    assert conditional.output_len == twin.output_len == 6
    for req in batch.reqs:
        assert torch.equal(s.token_pool[req.table_idx, : req.device_len], req.input_ids)

    logits = torch.tensor([[[4.0, 1.0]], [[1.0, 3.0]]])  # rows: conditional, twin
    guided = s._apply_cfg(logits, batch)
    assert torch.equal(guided[0], torch.tensor([[5.5, 0.0]]))  # twin + 1.5 * (conditional - twin)
    assert torch.equal(guided[1], logits[1])
    generated = torch.tensor([[11, 12, 13], [99, 99, 99]])
    s._sync_cfg_twin_tokens(generated, batch)
    assert torch.equal(generated[0], generated[1])

def test_scale_one_takes_the_unguided_path(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    user = _user(monkeypatch, tmp_path)
    s = _scheduler()
    s._process_one_msg(user)
    batch = s._schedule_prefill()
    assert len(batch.reqs) == 1 and batch.reqs[0].cfg_twin is None and batch.reqs[0].cfg_scale == 1.0
    assert len(s.table_manager.free_indices) == 3
    logits = torch.randn(1, C, 5, dtype=torch.bfloat16)
    assert s._apply_cfg(logits, batch) is logits

def test_controls_cfg_needs_free_slots_for_both_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    s = _scheduler(tables=1)
    s._process_one_msg(_user(monkeypatch, tmp_path, steering_delta=torch.ones(4),
                                 sampling_params=TTSSamplingParams(controls_cfg_scale=1.5, max_tokens=20)))
    assert s._schedule_prefill() is None and len(s.waiting_reqs) == 1


def test_speaker_cfg_twin_carries_the_same_steering(monkeypatch, tmp_path):
    """Speaker CFG retains steering delta, span schedule and accent on both branches."""
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    delta, schedule = torch.ones(4), torch.tensor([[0.0, 5.0, 3.0, 1.0, 1.0, 1.0, 1.0]])
    user = _user(monkeypatch, tmp_path, steering_delta=delta, steer_schedule=schedule,
                 sampling_params=TTSSamplingParams(speaker_cfg_scale=1.5, max_tokens=20))
    s = _scheduler()
    s._process_one_msg(user)
    tagged, plain = s._schedule_prefill().reqs
    assert plain.is_cfg_uncond and torch.equal(plain.steering_delta, delta)
    assert plain.steer_schedule is tagged.steer_schedule is not None
    assert plain.accent_slot == tagged.accent_slot and plain.accent_scale == tagged.accent_scale

def test_controls_cfg_twin_drops_the_steering_only(monkeypatch, tmp_path):
    """controls_cfg_scale != 1 with a hidden steering delta and a speaker embedding: the twin is the same prompt and
    speaker without the steering delta / schedule, guided at the controls scale."""
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    delta, schedule = torch.ones(4), torch.tensor([[0.0, 5.0, 3.0, 1.0, 1.0, 1.0, 1.0]])
    params = TTSSamplingParams(controls_cfg_scale=1.5, max_tokens=20)
    user = _user(
        monkeypatch,
        tmp_path,
        text="Hello there.",
        sampling_params=params,
        steering_delta=delta,
        steer_schedule=schedule,
    )
    s = _scheduler()
    s._process_one_msg(user)
    steered, twin = s._schedule_prefill().reqs
    assert steered.cfg_twin is twin and twin.is_cfg_uncond and steered.cfg_scale == 1.5
    assert torch.equal(steered.input_ids, twin.input_ids)
    assert torch.equal(steered.speaker_embedding, twin.speaker_embedding)
    assert torch.equal(steered.steering_delta, delta) and steered.steer_schedule is not None
    assert twin.steering_delta is None and twin.steer_schedule is None

