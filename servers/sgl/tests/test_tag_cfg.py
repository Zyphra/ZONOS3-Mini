"""Tag CFG: nonverbal markup helpers, markup-safe normalization, the tokenizer's plain-text twin
and the scheduler's tagged-vs-plain pair. CPU only; the real NeMo grammar cases skip without pynini."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from zonos3.tokenizer import textnorm
from zonos3.tokenizer.textnorm import TTSTextNormalizer
from zonos3.tts import plan
from zonos3.tts.markup import (
    _NONVERBAL_MARKUP,
    NONVERBAL_MARKUP_TAGS,
    has_nonverbal_markup,
    split_nonverbal_markup,
    strip_nonverbal_markup,
)
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
TAGGED = "Hello <laugh/> there <shouting> friend! </shouting>"


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


class _UpperNormalizer:
    """Stand-in grammar: upper-cases, spells digits, and refuses any markup character it is handed."""

    def normalize(self, text, verbose=False, punct_pre_process=False, punct_post_process=False):
        assert "<" not in text and ">" not in text, f"markup reached the grammar: {text!r}"
        return text.upper().replace("5", "FIVE")


def _message(scale=2.0, **kwargs):
    fields = {
        "uid": 7,
        "text": TAGGED,
        "language": "en_us",
        "sampling_params": TTSSamplingParams(tag_cfg_scale=scale, max_tokens=20),
        "speaker_embedding": torch.tensor([1.0, 2.0]),
        "condition_buckets": [3, UNKNOWN, 5],
    }
    fields.update(kwargs)
    return TTSTokenizeMsg(**fields)


def _tokenize(monkeypatch, tmp_path, messages, *, normalizer=None):
    """Run the real tokenize_worker loop on ``messages``; return the TTSUserMsgs sent to the scheduler."""
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

    class Normalizer:
        def __init__(self):
            self.inner = TTSTextNormalizer()
            if normalizer is not None:
                self.inner.get = lambda lang: normalizer

        def warmup(self, languages=None):
            pass

        def normalize(self, text, language):
            return self.inner.normalize(text, language)

    monkeypatch.setattr(tokenizer_server, "ZmqPushQueue", Push)
    monkeypatch.setattr(tokenizer_server, "ZmqPullQueue", Pull)
    monkeypatch.setattr("zonos3_sgl.tokenizer.vocoder.TTSVocoderManager", _Vocoder)
    monkeypatch.setattr(textnorm, "TTSTextNormalizer", Normalizer)
    monkeypatch.setattr(textnorm, "normalization_enabled", lambda: normalizer is not None)
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


# ------------------------------------------------------------------------------------------------ markup helpers
def test_markup_grammar_is_the_training_recipe():
    assert _NONVERBAL_MARKUP.pattern == r"</?(?:laugh|breath|sigh|cough|panting|shouting|crying)\s*/?>"
    assert NONVERBAL_MARKUP_TAGS == ("laugh", "breath", "sigh", "cough", "panting", "shouting", "crying")


def test_strip_only_recognized_markup_preserves_spoken_text():
    text = "Hallo  <laugh/> schöne  Welt <shouting>  heute </shouting> <unknown> + 7"
    assert has_nonverbal_markup(text)
    assert strip_nonverbal_markup(text) == "Hallo schöne Welt heute <unknown> + 7"
    assert has_nonverbal_markup("<breath/> <panting/> <cough/> <sigh/> <crying>hello</crying>")
    assert has_nonverbal_markup("<laugh />")
    for text in ["<unknown> hi", "<giggle/> hi", "<Laugh/> hi", "a < b > c"]:
        assert not has_nonverbal_markup(text)
        assert strip_nonverbal_markup(text) == text
    assert strip_nonverbal_markup("two  spaces <unknown>") == "two  spaces <unknown>"
    assert strip_nonverbal_markup(TAGGED) == "Hello there friend!"


def test_split_alternates_text_and_markers_and_round_trips():
    pieces = split_nonverbal_markup(TAGGED)
    assert pieces[1::2] == ["<laugh/>", "<shouting>", "</shouting>"]
    assert pieces[0::2] == ["Hello ", " there ", " friend! ", ""]
    assert "".join(pieces) == TAGGED
    assert split_nonverbal_markup("plain") == ["plain"]


# ------------------------------------------------------------------------------------------------ normalization
def test_normalizer_never_hands_markup_to_a_grammar_and_keeps_the_spacing():
    tn = TTSTextNormalizer()
    tn.get = lambda lang: _UpperNormalizer()
    assert tn.normalize("So I said <laugh/> no way, $5.", "en_us") == "SO I SAID <laugh/> NO WAY, $FIVE."
    assert tn.normalize("<shouting>Get out!</shouting>  ok", "en_us") == "<shouting>GET OUT!</shouting> OK"
    assert tn.normalize("我说<laugh/>不行5", "cmn") == "我说<laugh/>不行FIVE"
    assert tn.normalize("a <laugh/> <sigh/> b\n<cough/>", "ko") == "A <laugh/> <sigh/> B <cough/>"
    # untagged text takes the whole-text path; unsupported languages come back as typed
    assert tn.normalize("no markup 5", "en_us") == "NO MARKUP FIVE"
    assert tn.normalize("keep  <laugh/>  as typed", "xx") == "keep  <laugh/>  as typed"


@pytest.fixture(scope="module")
def tn():
    pytest.importorskip("pynini", reason="NeMo grammars need pynini")
    return TTSTextNormalizer()


@pytest.mark.parametrize(
    "language,text,expected",
    [
        (
            "en_us",
            "<shouting>Get out!</shouting> It costs $5 <sigh/> today.",
            "<shouting>Get out!</shouting> It costs five dollars <sigh/> today.",
        ),
        ("cmn", "<shouting>出去！</shouting>他花了5元<sigh/>。", "<shouting>出去！</shouting>他花了五元<sigh/>。"),
        ("cmn", "我有3个 <laugh/> 苹果", "我有三个 <laugh/> 苹果"),
        ("ja", "<shouting>出て行け！</shouting>5円です<sigh/>。", "<shouting>出て行け！</shouting>五円です<sigh/>。"),
        # whole-text normalization raised on the ASCII space next to the marker and left "3日" unread
        ("ja", "今日は3日 <breath/> です", "今日は三日 <breath/> です"),
        ("ko", "<shouting>나가!</shouting> 5원이에요 <sigh/>.", "<shouting>나가!</shouting> 오 원이에요 <sigh/>."),
    ],
)
def test_real_grammars_keep_markup(tn, language, text, expected):
    assert tn.normalize(text, language) == expected


# ------------------------------------------------------------------------------------------------ tokenizer twin
def test_tokenizer_builds_the_plain_twin_from_the_same_normalized_text(monkeypatch, tmp_path):
    builder = TTSPromptBuilder(_config())
    msg = _message(text="So I said <laugh/> 5 times.")
    (user,) = _tokenize(monkeypatch, tmp_path, [msg], normalizer=_UpperNormalizer())
    assert isinstance(user, TTSUserMsg) and user.plain_input_ids is not None
    tagged, plain = user.input_ids, user.plain_input_ids

    def layout(text):
        # today's layout: speaker slot, condition rows, text rows, codec-silence prefill
        return torch.cat(
            (
                builder.speaker_slot(),
                builder.condition_rows(CONDITIONS),
                builder.build(text, quality_buckets=[1]),
                _silence_rows(SILENCE_FRAMES, builder.config),
            )
        )

    assert torch.equal(tagged, layout("SO I SAID <laugh/> FIVE TIMES."))
    assert torch.equal(plain, layout("SO I SAID FIVE TIMES."))
    # shared conditioning: speaker slot, condition rows (same values / start) and the prefill rows
    assert user.speaker_token_position == 0 and user.condition_token_start == 1
    assert user.condition_values == [3, UNKNOWN, 5]
    assert user.prefill_rows is not None and len(user.prefill_rows) == SILENCE_FRAMES
    assert len(tagged) > len(plain)


def test_tokenizer_scale_one_sends_no_twin_and_today_layout(monkeypatch, tmp_path):
    builder = TTSPromptBuilder(_config())
    users = _tokenize(monkeypatch, tmp_path, [_message(scale=1.0), _message(scale=1.0, text="Hello there.")])
    assert [u.plain_input_ids for u in users] == [None, None]
    expected = torch.cat(
        (
            builder.speaker_slot(),
            builder.condition_rows(CONDITIONS),
            builder.build("Hello there.", quality_buckets=[1]),
            _silence_rows(SILENCE_FRAMES, builder.config),
        )
    )
    assert torch.equal(users[1].input_ids, expected)


def test_tokenizer_twin_on_the_raw_reference_path(monkeypatch, tmp_path):
    codes = torch.arange(12, dtype=torch.int32).reshape(6, C)
    msg = _message(speaker_embedding=None, speaker_reference_codes=codes, silence_prefill_frames=0)
    (user,) = _tokenize(monkeypatch, tmp_path, [msg])
    prefix = user.speaker_reference_prefix_length
    assert prefix > 0 and user.condition_token_start == prefix
    assert torch.equal(user.input_ids[: prefix + CONDITIONS], user.plain_input_ids[: prefix + CONDITIONS])
    builder = TTSPromptBuilder(_config())
    plain_rows = builder.build(strip_nonverbal_markup(TAGGED), quality_buckets=[1])
    assert torch.equal(user.plain_input_ids[prefix + CONDITIONS :], plain_rows)


def test_plain_twin_refuses_a_prompt_not_ending_in_its_text_rows():
    rows = torch.ones((4, C + 1), dtype=torch.int32)
    with pytest.raises(ValueError):
        plan.PromptRows(rows, 0, -1, -1).with_text(rows[:2] + 1, rows[:1])


# ------------------------------------------------------------------------------------------------ scheduler pair
def _user(monkeypatch, tmp_path, scale=2.0, **kwargs):
    (user,) = _tokenize(monkeypatch, tmp_path, [_message(scale=scale, **kwargs)])
    return user


def test_prefill_runs_a_plain_twin_with_identical_conditioning(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    user = _user(monkeypatch, tmp_path, steering_delta=None)
    s = _scheduler(max_seq_len=len(user.input_ids) + 2 + 6)
    s._process_one_msg(user)
    batch = s._schedule_prefill()
    assert batch is not None and len(batch.reqs) == 2
    tagged, plain = batch.reqs
    assert tagged.cfg_twin is plain and plain.is_cfg_uncond and not tagged.is_cfg_uncond
    assert tagged.cfg_scale == 2.0 and plain.rng is None
    assert torch.equal(tagged.speaker_embedding, plain.speaker_embedding)
    assert tagged.speaker_token_position == plain.speaker_token_position == 0
    assert tagged.condition_values == plain.condition_values == [3, UNKNOWN, 5]
    assert tagged.condition_token_start == plain.condition_token_start
    # the scheduler adds the same background / accurate-mode markers to both branches
    added = len(tagged.input_ids) - len(user.input_ids)
    assert added == 2 and len(plain.input_ids) - len(user.plain_input_ids) == added
    assert tagged.condition_token_start == 1 + added
    assert torch.equal(tagged.input_ids[: 1 + added + CONDITIONS], plain.input_ids[: 1 + added + CONDITIONS])
    assert torch.equal(tagged.input_ids[-SILENCE_FRAMES:], plain.input_ids[-SILENCE_FRAMES:])
    assert tagged.output_len == plain.output_len == 6
    for req in batch.reqs:
        assert torch.equal(s.token_pool[req.table_idx, : req.device_len], req.input_ids)

    logits = torch.tensor([[[4.0, 1.0]], [[1.0, 3.0]]])  # rows: tagged, plain
    guided = s._apply_cfg(logits, batch)
    assert torch.equal(guided[0], torch.tensor([[7.0, -1.0]]))  # plain + 2 * (tagged - plain)
    assert torch.equal(guided[1], logits[1])
    generated = torch.tensor([[11, 12, 13], [99, 99, 99]])
    s._sync_cfg_twin_tokens(generated, batch)
    assert torch.equal(generated[0], generated[1])


def test_prefill_accounts_for_a_longer_plain_twin_and_drops_oversize(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    user = _user(monkeypatch, tmp_path)
    body = user.plain_input_ids[1:-SILENCE_FRAMES]
    user.plain_input_ids = torch.cat(
        [user.plain_input_ids[:-SILENCE_FRAMES], body, body, user.plain_input_ids[-SILENCE_FRAMES:]]
    )
    assert len(user.plain_input_ids) > len(user.input_ids)
    s = _scheduler(max_seq_len=len(user.plain_input_ids) + 2 + 4)
    s._process_one_msg(user)
    batch = s._schedule_prefill()
    assert all(req.output_len == 4 for req in batch.reqs)
    too_long = _scheduler(max_seq_len=len(user.plain_input_ids))
    too_long._process_one_msg(user)
    assert not too_long.waiting_reqs and too_long._schedule_prefill() is None


def test_scale_one_takes_the_unguided_path(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    user = _user(monkeypatch, tmp_path, scale=1.0)
    s = _scheduler()
    s._process_one_msg(user)
    batch = s._schedule_prefill()
    assert len(batch.reqs) == 1 and batch.reqs[0].cfg_twin is None and batch.reqs[0].cfg_scale == 1.0
    assert len(s.table_manager.free_indices) == 3
    logits = torch.randn(1, C, 5, dtype=torch.bfloat16)
    assert s._apply_cfg(logits, batch) is logits


def test_tag_cfg_needs_free_slots_for_both_branches(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    s = _scheduler(tables=1)
    s._process_one_msg(_user(monkeypatch, tmp_path))
    assert s._schedule_prefill() is None and len(s.waiting_reqs) == 1


@pytest.mark.parametrize("field", ["speaker_cfg_scale", "controls_cfg_scale"])
def test_scheduler_refuses_stacked_guidance(monkeypatch, tmp_path, field):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    user = _user(monkeypatch, tmp_path)
    user.sampling_params = replace(user.sampling_params, **{field: 1.5})
    s = _scheduler()
    s._process_one_msg(user)
    with pytest.raises(ValueError):
        s._schedule_prefill()


def test_scheduler_refuses_a_guided_request_without_its_twin(monkeypatch, tmp_path):
    user = _user(monkeypatch, tmp_path)
    user.plain_input_ids = None
    s = _scheduler()
    s._process_one_msg(user)
    with pytest.raises(ValueError):
        s._schedule_prefill()


# ------------------------------------------------------------------------------------------------ interactions
# Markup together with the rest of this runtime: pronunciation groups, steering and edit text handling.
G = "\ue000həˈloʊ\ue001"


def test_normalizer_keeps_markup_and_pronunciation_groups_together():
    tn = TTSTextNormalizer()
    tn.get = lambda lang: _UpperNormalizer()
    assert tn.normalize(f"say {G} <laugh/> 5 times", "en_us") == f"SAY {G} <laugh/> FIVE TIMES"
    assert tn.normalize(f"<shouting>{G} 5</shouting>", "en_us") == f"<shouting>{G} FIVE</shouting>"
    # no markup: the group path as before (whitespace around groups kept exactly)
    assert tn.normalize(f"say  {G}  5", "en_us") == f"SAY  {G}  FIVE"


def test_plain_twin_text_keeps_pronunciation_groups():
    # (as on the source branch, a removed marker leaves a space: "</shouting>!" -> " !")
    assert strip_nonverbal_markup(f"<laugh/>{G} and <shouting>{G}</shouting>!") == f"{G} and {G} !"


def test_tag_cfg_twin_carries_the_same_steering(monkeypatch, tmp_path):
    """The twin differs in its text rows only: steering delta, span schedule and accent steer both branches."""
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
    delta, schedule = torch.ones(4), torch.tensor([[0.0, 5.0, 3.0, 1.0, 1.0, 1.0, 1.0]])
    user = _user(monkeypatch, tmp_path, steering_delta=delta, steer_schedule=schedule)
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
        scale=1.0,
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


def test_rate_schedule_gives_markers_no_time(tmp_path):
    import numpy as np
    from zonos3.tts.steering import load_bundle, resolve_controls

    d = tmp_path / "style"
    d.mkdir()
    np.save(d / "whisper.npy", np.array([1.0, 0.0], dtype=np.float32))
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "dim": 2,
                "space": "attnres_source",
                "source_index": 19,
                "directions": {"whisper": {"file": "whisper.npy", "kind": "named"}},
            }
        )
    )
    (tmp_path / "bundle.json").write_text(json.dumps({"name": "t", "sets": {"style": "style"}, "ui": {"controls": []}}))
    bundle = load_bundle(tmp_path / "bundle.json")

    def placed(text, start, end):
        r = resolve_controls(
            bundle,
            None,
            [{"start": start, "end": end, "controls": {"style.whisper": 1.0}}],
            text=text,
            speaker_embedding=None,
            sources=(19,),
            hidden_size=2,
            frames_per_char=2.0,
        )
        return r.schedule[0, :2].tolist()

    assert placed("ab cd", 3, 5) == [6.0, 10.0]
    text = "ab <laugh/>cd"
    assert placed(text, text.index("cd"), len(text)) == [6.0, 10.0]  # the marker's 8 characters take no frames
    text = "ab <shouting>cd</shouting>"
    assert placed(text, text.index("<shouting>"), len(text)) == [6.0, 10.0]  # span tags around the words take none


def test_span_follow_tokens_skip_marker_edges():
    from zonos3.tts.span_follow import span_token_ranges

    text = "a <laugh/> b"  # byte ids: BOS 0, a 1, ' ' 2, <laugh/> 3..10, ' ' 11, b 12, EOS 13
    assert span_token_ranges(text, text, [(0, len(text))], None) == [(1, 13)]  # markers inside a span stay in it
    assert span_token_ranges(text, text, [(2, 10)], None) == [(3, 11)]  # a span over a marker only keeps it
    text = "<shouting>hi</shouting> ok"
    assert span_token_ranges(text, text, [(0, 23)], None) == [(11, 13)]  # span tags do not open / close it
    # normalization collapsed the spaces around a marker: offsets still map through
    original, final = "a  <laugh/>  b", "a <laugh/> b"
    assert span_token_ranges(original, final, [(original.index("b"), len(original))], None) == [(12, 13)]


def test_edit_target_markup_is_outside_the_word_diff():
    from zonos3.tts.edit_align import canonical_markup, target_tokens

    target = canonical_markup(f"Say <laugh /> hi{G} <shouting>now</shouting>")
    assert target == f"Say <laugh/> hi{G} <shouting>now</shouting>"
    # [target tokens] + one token list per whitespace word: a marker word has none, a group is one token
    assert target_tokens(target) == [["SAY", "HI", G, "NOW"], ["SAY"], [], ["HI", G], ["NOW"]]
    assert target_tokens("plain words, too") == [["PLAIN", "WORDS", "TOO"], ["PLAIN"], ["WORDS"], ["TOO"]]
