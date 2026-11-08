from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from zonos3.tts.sampler import sample_tts
from zonos3_sgl.engine.sample import TTSSampler
from zonos3_sgl.message.tts import TTSSamplingParams

# Edit-checkpoint layout: codes 0..3, EOA 4, pad 5, span-end 6 (V = codebook_size + 3).
CODEBOOK_SIZE, EOA, PAD, SPAN_END = 4, 4, 5, 6
CODEBOOK_SIZES = (4, 2)


def _sample(logits, stop_ids, temperature):
    batch = logits.shape[0]
    return sample_tts(
        logits=logits,
        temperatures=torch.full((batch,), temperature),
        top_ks=torch.full((batch,), logits.shape[-1]),
        top_ps=torch.ones(batch),
        min_ps=torch.zeros(batch),
        codebook_size=CODEBOOK_SIZE,
        audio_pad_id=PAD,
        stop_token_ids=torch.tensor(stop_ids),
        codebook_sizes=CODEBOOK_SIZES,
        text_vocab=99,
    ).tolist()


def test_greedy_rows_only_take_their_own_end_token():
    utterance = [0.0, 0.0, 0.0, 0.0, 3.0, 9.0, 6.0]  # pad > span-end > EOA > codes
    span = [0.0, 0.0, 0.0, 0.0, 6.0, 9.0, 3.0]  # pad > EOA > span-end > codes
    logits = torch.tensor([[utterance, utterance], [span, span]])

    assert _sample(logits, [EOA, SPAN_END], 0.0) == [[EOA, EOA, 99], [SPAN_END, SPAN_END, 99]]


def test_greedy_keeps_per_codebook_ordinary_limits():
    row = [0.0, 1.0, 2.0, 3.0, -9.0, 9.0, -9.0]
    logits = torch.tensor([[row, row], [row, row]])

    assert _sample(logits, [EOA, SPAN_END], 0.0) == [[3, 1, 99], [3, 1, 99]]


def test_stochastic_rows_never_sample_pad_or_the_other_end_token():
    torch.manual_seed(0)
    logits = torch.zeros((2, 2, CODEBOOK_SIZE + 3))
    seen = [[set(), set()], [set(), set()]]
    for _ in range(500):
        for row, tokens in enumerate(_sample(logits, [EOA, SPAN_END], 1.0)):
            for codebook in range(2):
                seen[row][codebook].add(tokens[codebook])

    assert seen[0] == [{0, 1, 2, 3, EOA}, {0, 1, EOA}]
    assert seen[1] == [{0, 1, 2, 3, SPAN_END}, {0, 1, SPAN_END}]


def test_models_without_span_end_keep_eoa_and_mask_pad():
    torch.manual_seed(0)
    logits = torch.zeros((1, 2, CODEBOOK_SIZE + 2))
    seen = [set(), set()]
    for _ in range(500):
        tokens = _sample(logits, [EOA], 1.0)[0]
        seen[0].add(tokens[0])
        seen[1].add(tokens[1])

    assert seen == [{0, 1, 2, 3, EOA}, {0, 1, EOA}]


def test_prepare_builds_stop_ids_and_rejects_unknown_end_tokens():
    def req(stop_id):
        return SimpleNamespace(
            sampling_params=TTSSamplingParams(repetition_penalty=1.0),
            total_generated=0,
            history_rows=0,
            device_len=1,
            table_idx=0,
            input_ids=torch.zeros((1, 3), dtype=torch.int32),
            rng=None,
            stop_id=stop_id,
        )

    def sampler(end_token_ids):
        return TTSSampler(
            device=torch.device("cpu"),
            n_codebooks=2,
            codebook_size=CODEBOOK_SIZE,
            text_vocab=99,
            audio_pad_id=PAD,
            end_token_ids=end_token_ids,
            codebook_sizes=CODEBOOK_SIZES,
        )

    args = sampler((EOA, SPAN_END)).prepare(SimpleNamespace(reqs=[req(EOA), req(SPAN_END)]))
    assert args.stop_token_ids.tolist() == [EOA, SPAN_END]

    with pytest.raises(ValueError):
        sampler((EOA,)).prepare(SimpleNamespace(reqs=[req(SPAN_END)]))


def _seeded(logits, stop_ids, seeds):
    batch = logits.shape[0]
    return sample_tts(
        logits=logits,
        temperatures=torch.ones(batch),
        top_ks=torch.full((batch,), 3),
        top_ps=torch.full((batch,), 0.9),
        min_ps=torch.zeros(batch),
        codebook_size=CODEBOOK_SIZE,
        audio_pad_id=PAD,
        stop_token_ids=torch.tensor(stop_ids),
        codebook_sizes=CODEBOOK_SIZES,
        text_vocab=99,
        generators=[torch.Generator().manual_seed(seed) for seed in seeds],
    ).tolist()


def test_utterances_draw_the_same_samples_as_a_checkpoint_without_span_end():
    """multinomial draws one random number per column: an utterance on an edit checkpoint must
    sample over the base vocabulary so a seed gives the audio of the stripped checkpoint."""
    torch.manual_seed(1)
    for _ in range(20):
        edit = torch.randn(1, 2, CODEBOOK_SIZE + 3)
        base = edit[..., : CODEBOOK_SIZE + 2]
        assert _seeded(edit, [EOA], [7]) == _seeded(base, [EOA], [7])
        # Batched with an edit span (per-row generators): each row keeps its own draws.
        pair = torch.cat([edit, torch.randn(1, 2, CODEBOOK_SIZE + 3)])
        assert _seeded(pair, [EOA, SPAN_END], [7, 8])[0] == _seeded(base, [EOA], [7])[0]


@pytest.mark.parametrize("min_frames", [1, 4, 7])
def test_min_frames_hold_the_aligned_stop_frame_back(min_frames):
    """The stop token is masked per codebook until s >= min_frames + c, so a request that wants
    to stop at once ends exactly at aligned frame min_frames (check_eos shifts by the codebook)."""
    from zonos3_sgl.core import TTSReq

    n = 2
    req = TTSReq(
        input_ids=torch.full((2, n + 1), PAD, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=64,
        uid=0,
        sampling_params=TTSSamplingParams(min_frames=min_frames),
        cache_handle=None,
        n_codebooks=n,
        audio_pad_id=PAD,
        stop_id=SPAN_END,
    )
    sampler = TTSSampler(
        device=torch.device("cpu"),
        n_codebooks=n,
        codebook_size=CODEBOOK_SIZE,
        text_vocab=99,
        audio_pad_id=PAD,
        end_token_ids=(EOA, SPAN_END),
        codebook_sizes=CODEBOOK_SIZES,
    )
    stop_first = torch.tensor([[[1.0, 0.0, 0.0, 0.0, -9.0, -9.0, 9.0]] * n])
    for _ in range(min_frames + n + 2):
        args = sampler.prepare(SimpleNamespace(reqs=[req]))
        tokens = sample_tts(
            stop_first,
            torch.zeros(1),
            torch.full((1,), 7),
            torch.ones(1),
            torch.zeros(1),
            codebook_size=CODEBOOK_SIZE,
            audio_pad_id=PAD,
            stop_token_ids=args.stop_token_ids,
            codebook_sizes=CODEBOOK_SIZES,
            text_vocab=99,
            stop_blocked=args.stop_blocked,
        )[0].tolist()
        req.total_generated += 1
        if req.check_eos(tokens[:n]):
            break
    assert req.eos_frame == min_frames


def test_min_frames_off_builds_no_mask():
    req = SimpleNamespace(
        sampling_params=TTSSamplingParams(repetition_penalty=1.0),
        total_generated=0,
        history_rows=0,
        device_len=1,
        table_idx=0,
        input_ids=torch.zeros((1, 3), dtype=torch.int32),
        rng=None,
        stop_id=EOA,
    )
    sampler = TTSSampler(
        device=torch.device("cpu"),
        n_codebooks=2,
        codebook_size=CODEBOOK_SIZE,
        text_vocab=99,
        audio_pad_id=PAD,
        end_token_ids=(EOA,),
        codebook_sizes=CODEBOOK_SIZES,
    )
    assert sampler.prepare(SimpleNamespace(reqs=[req])).stop_blocked is None
