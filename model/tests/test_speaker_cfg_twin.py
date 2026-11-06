"""Speaker CFG twin on both speaker paths (2026-10-16): the raw-reference twin drops the reference prefix, the
embedding twin drops the reserved speaker row; both get all-unknown condition rows shifted back by the removed rows."""

from types import SimpleNamespace

import pytest
import torch
from zonos3.speaker_cfg import SpeakerCfgTwin, is_reserved_speaker_slot, speaker_cfg_twin, unconditional_prompt

C, PAD, TEXT_VOCAB = 10, 4097, 8196
UNKNOWNS = [10, 10, 30]  # per-table unknown ids of the default LM (condition_bucket_counts [10, 10, 30])


def frame(text_token):
    return [PAD] * C + [text_token]


def slot():
    return frame(TEXT_VOCAB)


def prompt(*, reference_frames=0, speaker=False, condition=3, text=(5, 6, 7, 8)):
    rows = []
    if speaker:
        rows.append(slot())
    for i in range(reference_frames):
        rows.append([100 + i] * C + [TEXT_VOCAB])
    rows += [slot()] * condition  # condition rows are the same all-pad row as the speaker slot
    rows += [frame(t) for t in text]
    return torch.tensor(rows, dtype=torch.int32)


def msg(ids, *, prefix=0, embedding=False, position=-1):
    return SimpleNamespace(
        input_ids=ids,
        speaker_reference_prefix_length=prefix,
        speaker_embedding=torch.ones(192) if embedding else None,
        speaker_token_position=position,
    )


def test_reference_twin_drops_the_prefix_and_shifts_the_rows_back():
    ids = prompt(reference_frames=7)
    m = msg(ids, prefix=7)
    assert torch.equal(unconditional_prompt(m, n_codebooks=C, audio_pad_id=PAD), ids[7:])
    twin = speaker_cfg_twin(
        m,
        ids,
        n_codebooks=C,
        audio_pad_id=PAD,
        text_vocab=TEXT_VOCAB,
        speaker_token_position=-1,
        condition_values=[3, 5, 7],
        condition_token_start=7,
        condition_unknowns=UNKNOWNS,
    )
    assert isinstance(twin, SpeakerCfgTwin) and twin.removed_rows == 7
    assert (
        torch.equal(twin.input_ids, ids[7:])
        and twin.condition_values == [10, 10, 30]
        and twin.condition_token_start == 0
    )
    with pytest.raises(ValueError):
        speaker_cfg_twin(
            m,
            ids,
            n_codebooks=C,
            audio_pad_id=PAD,
            text_vocab=TEXT_VOCAB,
            speaker_token_position=-1,
            condition_values=[3, 5, 7],
            condition_token_start=7,
            condition_unknowns=[10],
        )
    # models without condition rows keep None / -1
    twin = speaker_cfg_twin(
        m,
        ids,
        n_codebooks=C,
        audio_pad_id=PAD,
        text_vocab=TEXT_VOCAB,
        speaker_token_position=-1,
        condition_values=None,
        condition_token_start=-1,
        condition_unknowns=UNKNOWNS,
    )
    assert twin.condition_values is None and twin.condition_token_start == -1


def test_embedding_twin_drops_the_reserved_speaker_row():
    ids = prompt(speaker=True)
    m = msg(ids, embedding=True, position=0)
    assert is_reserved_speaker_slot(ids[0], n_codebooks=C, audio_pad_id=PAD, text_vocab=TEXT_VOCAB)
    twin = speaker_cfg_twin(
        m,
        ids,
        n_codebooks=C,
        audio_pad_id=PAD,
        text_vocab=TEXT_VOCAB,
        speaker_token_position=0,
        condition_values=[0, 9, 4],
        condition_token_start=1,
        condition_unknowns=UNKNOWNS,
    )
    assert twin.removed_rows == 1 and torch.equal(twin.input_ids, ids[1:])
    assert twin.condition_values == UNKNOWNS and twin.condition_token_start == 0
    # the evaluation generator's twin: condition rows first, then the text, nothing else
    assert torch.equal(twin.input_ids, prompt(speaker=False))


def test_embedding_twin_with_the_scheduler_added_slot_offline_path():
    # offline TTSLLM prompts carry no slot; the scheduler prepends it and passes the final prompt + position 0
    bare = prompt(speaker=False)
    m = msg(bare, embedding=True, position=-1)
    final = torch.cat((torch.tensor([slot()], dtype=torch.int32), bare))
    twin = speaker_cfg_twin(
        m,
        final,
        n_codebooks=C,
        audio_pad_id=PAD,
        text_vocab=TEXT_VOCAB,
        speaker_token_position=0,
        condition_values=list(UNKNOWNS),
        condition_token_start=1,
        condition_unknowns=UNKNOWNS,
    )
    assert torch.equal(twin.input_ids, bare) and twin.removed_rows == 1 and twin.condition_token_start == 0


def test_refusals():
    ids = prompt(speaker=True)
    with pytest.raises(ValueError):
        unconditional_prompt(
            msg(ids, embedding=True, position=-1), n_codebooks=C, audio_pad_id=PAD, text_vocab=TEXT_VOCAB
        )
    with pytest.raises(ValueError):  # position 0 but a text row there (no condition rows)
        unconditional_prompt(
            msg(prompt(speaker=False, condition=0), embedding=True, position=0),
            n_codebooks=C,
            audio_pad_id=PAD,
            text_vocab=TEXT_VOCAB,
        )
    with pytest.raises(ValueError):
        unconditional_prompt(
            msg(ids, prefix=3, embedding=True, position=0), n_codebooks=C, audio_pad_id=PAD, text_vocab=TEXT_VOCAB
        )
    with pytest.raises(ValueError):
        unconditional_prompt(msg(prompt(), prefix=0), n_codebooks=C, audio_pad_id=PAD)
    no_text = torch.tensor([[7] * C + [TEXT_VOCAB]] * 3, dtype=torch.int32)  # audio rows only: no text anywhere
    with pytest.raises(ValueError):
        unconditional_prompt(
            msg(no_text, embedding=True, position=0), n_codebooks=C, audio_pad_id=PAD, text_vocab=TEXT_VOCAB
        )


def test_trailing_prefilled_audio_rows_stay_in_both_branches():
    """A codec-silence prefill (2026-10-17) appends sheared audio rows after EOS; the twin keeps them on both paths."""
    opening = torch.tensor([[7] * C + [TEXT_VOCAB], [7] * C + [TEXT_VOCAB]], dtype=torch.int32)
    ids = torch.cat((prompt(speaker=True), opening))
    twin = unconditional_prompt(
        msg(ids, embedding=True, position=0), n_codebooks=C, audio_pad_id=PAD, text_vocab=TEXT_VOCAB
    )
    assert torch.equal(twin, ids[1:]) and torch.equal(twin[-2:], opening)
    ids = torch.cat((prompt(reference_frames=4), opening))
    twin = unconditional_prompt(msg(ids, prefix=4), n_codebooks=C, audio_pad_id=PAD)
    assert torch.equal(twin, ids[4:]) and torch.equal(twin[-2:], opening)
