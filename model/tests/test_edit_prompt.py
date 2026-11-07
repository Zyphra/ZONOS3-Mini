import numpy as np
import pytest
import torch
from zonos3.tts.edit import edit_prompt_tokens, extract_fill, splice_edit
from zonos3.tts.edit_align import AlignedWord, diff_spans, new_word_confidence, seconds_to_spans
from zonos3.tts.prompt import TTSPromptConfig, shear


@pytest.mark.parametrize("spans", [((0, 10),), ((50, 10),), ((5, 10), (25, 8), (50, 10)), ((5, 10), (20, 8))])
def test_prompt_matches_independent_delayed_layout(spans):
    k, eoa, pad, span_end, text_pad = 3, 8, 9, 10, 448
    cfg = TTSPromptConfig(n_codebooks=k, codebook_sizes=(8,) * k, audio_pad_id=pad, text_vocab=text_pad)
    codes = np.arange(60 * k).reshape(60, k) % 8
    text = np.array([[pad, pad, pad, 2], [pad, pad, pad, 192], [pad, pad, pad, 3]])

    def block(frames, end):
        source = np.full((len(frames) + int(end is not None) + k - 1, k), pad)
        source[: len(frames)] = frames
        if end is not None:
            source[len(frames)] = end
        delayed = np.full_like(source, pad)
        for c in range(k):
            delayed[c:, c] = source[: len(source) - c, c]
        return np.column_stack((delayed, np.full(len(source), text_pad)))

    def marker(j):
        return np.array([[pad] * k + [text_pad + 1 + j]])

    # Training layout: context blocks without end row, final context block ends
    # with EOA, every infill block ends with the dedicated span-end token.
    context, cursor = [text], 0
    for j, (start, length) in enumerate(spans):
        if start > cursor:
            context.append(block(codes[cursor:start], None))
        context.append(marker(j))
        cursor = start + length
    context.append(block(codes[cursor:], eoa))
    fills = []
    for j, (start, length) in enumerate(spans):
        expected = context + [marker(j)]
        actual = edit_prompt_tokens(text, codes, spans, fills, cfg, span_end_id=span_end)
        np.testing.assert_array_equal(actual.numpy(), np.concatenate(expected))
        fills.append(codes[start : start + length])
        context.extend((marker(j), block(fills[-1], span_end)))
    np.testing.assert_array_equal(splice_edit(codes, spans, fills).numpy(), codes)


def test_prompt_rejects_span_end_other_than_pad_plus_one():
    cfg = TTSPromptConfig(n_codebooks=3, codebook_sizes=(8,) * 3, audio_pad_id=9, text_vocab=448)
    codes = np.zeros((20, 3), dtype=np.int64)
    text = np.array([[9, 9, 9, 2], [9, 9, 9, 3]])
    with pytest.raises(ValueError):
        edit_prompt_tokens(text, codes, ((5, 4),), [], cfg, span_end_id=8)


def test_splice_different_fill_lengths_preserves_context():
    codes = torch.arange(20).reshape(10, 2)
    actual = splice_edit(codes, [(2, 2), (7, 2)], [torch.full((1, 2), 99), torch.full((3, 2), 88)])
    expected = torch.cat((codes[:2], torch.full((1, 2), 99), codes[4:7], torch.full((3, 2), 88), codes[9:]))
    assert torch.equal(actual, expected)


def _generated_rows(frames, end, tail):
    # Rows as the model emits them: delay-stacked frames, the end token, then delay tail rows.
    k, pad = frames.shape[1], 9
    source = torch.full((len(frames) + int(end is not None) + tail, k), pad)
    source[: len(frames)] = frames
    if end is not None:
        source[len(frames)] = end
    return shear(source, pad)


def test_extract_fill_cuts_at_span_end_and_flags_stop():
    frames = torch.arange(15).reshape(5, 3) % 8
    rows = _generated_rows(frames, 10, 2)
    fill, stopped = extract_fill(rows, 5, audio_pad_id=9, codebook_sizes=(8,) * 3)
    assert torch.equal(fill, frames) and stopped


def test_extract_fill_without_end_drops_incomplete_delay_tail():
    frames = torch.arange(15).reshape(5, 3) % 8
    fill, stopped = extract_fill(_generated_rows(frames, None, 0), None, audio_pad_id=9, codebook_sizes=(8,) * 3)
    assert torch.equal(fill, frames[:3]) and not stopped


def test_extract_fill_truncates_before_non_codec_token():
    frames = torch.arange(15).reshape(5, 3) % 8
    frames[3, 1] = 8  # an end-of-utterance token inside a span fill
    fill, stopped = extract_fill(_generated_rows(frames, 10, 2), 5, audio_pad_id=9, codebook_sizes=(8,) * 3)
    assert torch.equal(fill, frames[:3]) and not stopped


def test_seconds_to_spans_rounds_outward_and_clips_to_audio():
    assert seconds_to_spans([[0.1, 0.5], [1.0, 9.9]], total_frames=100, frame_rate=15.0) == [(1, 7), (15, 85)]


def words():
    return [
        AlignedWord(t, i * 2, i * 2 + 1, (i + 1) / 10, [t]) for i, t in enumerate(["A", "B", "C", "D", "E", "F", "G"])
    ]


@pytest.mark.parametrize(
    "target,indices",
    [
        (["X", "B", "C", "D", "E", "F", "G"], [0]),
        (["A", "C", "D", "E", "F", "G"], []),
        (["X", "A", "B", "C", "D", "E", "F", "G"], [0]),
        (["A", "B", "X", "C", "D", "E", "F", "G"], [2]),
        (["A", "B", "C", "D", "E", "F", "G", "X"], [7]),
    ],
)
def test_diff_substitution_deletion_insertions(target, indices):
    result = diff_spans(words(), target, total_frames=210)
    assert result.new_target_tokens == indices
    assert len(result.spans) == 1
    start, length = result.spans[0]
    assert 0 <= start < start + length <= 210


def test_no_change_and_excess_regions():
    with pytest.raises(ValueError):
        diff_spans(words(), ["A", "B", "C", "D", "E", "F", "G"], total_frames=210)
    with pytest.raises(ValueError):
        diff_spans(words(), ["X", "B", "X", "D", "X", "F", "X"], total_frames=210)


def test_expanded_word_confidence_and_near_merge():
    w = [AlignedWord("a-b", 0, 1, 0.9, ["A", "B"]), AlignedWord("c", 1.1, 2, 0.3, ["C"])]
    assert new_word_confidence(w, [0, 1]) == 0.9
    assert new_word_confidence(w, []) is None
    assert new_word_confidence(w, [1, 2]) == pytest.approx(0.6)
    result = diff_spans(w, ["X", "B", "Y"], total_frames=30)
    assert len(result.spans) == 1
    assert result.new_target_tokens == [0, 2]


def test_seconds_to_spans_round_trips_frame_boundaries():
    assert seconds_to_spans([[17 / 15, 32 / 15], [70 / 15, 78 / 15]], total_frames=114) == [(17, 15), (70, 8)]
