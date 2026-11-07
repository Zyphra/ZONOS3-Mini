"""Reference speech-edit infill: each masked span generated until the span-end token (zonos3_sgl TTSLLM.edit).

The prompt is the training edit layout of zonos3.tts.edit (text, the unmasked context with infill markers, the fills
generated so far, the next marker), behind the condition rows and the speaker row like any prompt. The caller splices
the fills into the clip with ``zonos3.tts.edit.splice_edit``.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from zonos3.condition_buckets import check_ids
from zonos3.reference.lm import Zonos3LM
from zonos3.reference.synth import Sampling, generate_rows, prompt_builder
from zonos3.tts.edit import edit_marker_rows, edit_prompt_tokens, extract_fill, validate_edit_spans
from zonos3.tts.plan import Branch


@torch.inference_mode()
def generate_edit_fill(
    lm: Zonos3LM,
    codes,
    spans,
    text_rows,
    sampling: Sampling,
    seed: int,
    *,
    speaker_embedding: torch.Tensor | None = None,
    condition_values: list[int] | None = None,
    min_frames: list[int] | None = None,
    max_tokens: int = 1024,
    accents=None,
) -> tuple[list[torch.Tensor], list[int | None]]:
    """Fill every span ``(start, length)`` of ``codes`` [frames, n_codebooks] for the target ``text_rows`` (the prompt
    builder's rows of the edited transcript). Returns the fills ([frames, n_codebooks] codec codes each) and each fill's
    aligned span-end frame (None when it hit ``max_tokens``). Every span generates with ``seed``; ``min_frames`` (one per
    span) masks the span-end token until the fill has that many frames."""
    c = lm.config
    if c.span_end_id is None:
        raise ValueError(
            "Speech editing needs a checkpoint with a span-end audio token (span_end_id); "
            "this checkpoint only has the end-of-utterance token"
        )
    builder = prompt_builder(lm)
    max_spans = edit_marker_rows(c.text_extra_tokens, builder.config.text_tokenizer_spec)
    if max_spans <= 0:
        raise ValueError("Speech editing needs edit marker rows; this checkpoint's extra text rows are phoneme tokens")
    codes = torch.as_tensor(codes, dtype=torch.int64, device="cpu")
    validate_edit_spans(spans, len(codes), max_spans)
    if codes.ndim != 2 or codes.shape[1] != c.n_codebooks or bool(((codes < 0) | (codes >= c.codebook_size)).any()):
        raise ValueError("Invalid input codec frames")
    values = list(c.condition_bucket_counts) if condition_values is None else condition_values
    values = check_ids(values, c.condition_bucket_counts, "condition_values")
    lead = [builder.condition_rows(c.condition_count)]
    if speaker_embedding is not None:
        lead.insert(0, builder.speaker_slot())
    start = 1 if speaker_embedding is not None else 0
    fills: list[torch.Tensor] = []
    eos_frames: list[int | None] = []
    for region in range(len(spans)):
        prompt = edit_prompt_tokens(text_rows, codes, spans, fills, builder.config, span_end_id=c.span_end_id)
        branch = Branch(
            torch.cat(lead + [prompt]).to(torch.int32),
            speaker_embedding,
            0 if speaker_embedding is not None else -1,
            values,
            start,
            steered=False,
        )
        region_sampling = replace(sampling, min_frames=min_frames[region] if min_frames else sampling.min_frames)
        attempt = generate_rows(
            lm,
            branch,
            sampling=region_sampling,
            seed=seed,
            max_tokens=max_tokens,
            stop_id=c.span_end_id,
            accents=accents,
        )
        # the finishing row is not part of the generated span (it only completes the delayed codebooks' flush)
        rows = (
            torch.stack(attempt.rows[:-1])[:, : c.n_codebooks]
            if len(attempt.rows) > 1
            else torch.zeros((0, c.n_codebooks), dtype=torch.int32)
        )
        fill, _ = extract_fill(rows, attempt.eos_frame, audio_pad_id=c.audio_pad_id, codebook_sizes=c.codebook_sizes)
        fills.append(fill)
        eos_frames.append(attempt.eos_frame)
    return fills, eos_frames
