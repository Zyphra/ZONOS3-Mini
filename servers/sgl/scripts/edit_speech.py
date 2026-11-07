#!/usr/bin/env python3
"""Edit speech with automatic word-diff regions or explicit second spans.

Clips longer than --max-context-seconds are edited on word-snapped windows around the regions
(--context-seconds of unedited audio per side), planned over a FuseAlign alignment, as /v2/edit does.
Qwen3-ASR (.venv-asr, tools/setup_asr_env.sh) transcribes the original when --original-text is omitted; the words'
edit tokens are zonos3.tts.textmatch's.
"""

import argparse
import json
import math
from pathlib import Path

import soundfile as sf
from zonos3.tokenizer.dual_reference import encode_dual_reference_file
from zonos3.tts import textmatch
from zonos3.tts.edit_align import (
    AsrWorker,
    FuseAlignWorker,
    diff_spans,
    edit_aligned_words,
    edit_language,
    expected_fill_seconds,
    new_word_confidence,
    plan_edit_windows,
    seconds_to_spans,
    target_tokens,
    transcribe_audio,
)
from zonos3_sgl.tts.llm import TTSLLM, TTSSamplingParams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--target-text", required=True)
    p.add_argument("--original-text")
    p.add_argument("--span", nargs=2, type=float, action="append")
    p.add_argument("--margin", type=float, default=0.08)
    p.add_argument("--max-span-seconds", type=float, default=30)
    p.add_argument("--max-context-seconds", type=float, default=16)
    p.add_argument("--context-seconds", type=float, default=4)
    p.add_argument(
        "--min-fill-ratio",
        type=float,
        default=0.7,
        help="span-end masked until a fill reaches this share of its expected length",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--language", default="en_us")
    p.add_argument("--no-text-normalization", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    asr, aligner = AsrWorker(), FuseAlignWorker()
    lang = edit_language(a.language, asr.languages())
    duration = sf.info(a.audio).duration
    windowed = duration > a.max_context_seconds
    target_words = a.target_text.split()
    original, new_indices, words, normalized = a.original_text, [], None, None
    if not a.span or windowed or (a.min_fill_ratio > 0 and original):
        if duration > 150:
            p.error("speech editing supports clips up to 150 s")
        normalized = target_tokens(a.target_text)  # pronunciation groups are not normalized
        utterances = None
        if original is None:
            transcript = transcribe_audio(asr, aligner, a.audio, lang[1])
            original, utterances = transcript.text, transcript.utterances
            print("Qwen3-ASR original transcript: " + original)
        words = edit_aligned_words(asr, aligner, a.audio, original, lang, utterances)
    if a.span:
        spans = seconds_to_spans(a.span, math.ceil(duration * 15))
    else:
        diff = diff_spans(words, normalized[0], total_frames=math.ceil(duration * 15), margin_seconds=a.margin)
        spans, new_indices = diff.spans, diff.new_target_tokens
    llm = TTSLLM(
        a.model_path,
        attention_backend="fa",
        cache_type="naive",
        max_running_req=2,
        cuda_graph_max_bs=2,
        memory_ratio=0.5,
    )
    codes = encode_dual_reference_file(a.audio, max_seconds=600)
    windows = None
    if windowed:
        windows = plan_edit_windows(
            spans,
            words,
            target_words,
            normalized[1:],
            total_frames=len(codes),
            context_seconds=a.context_seconds,
            target_text=a.target_text,
        )
    expected = (
        [length / 15 for _, length in spans]
        if words is None
        else expected_fill_seconds(spans, words, target_words, normalized[1:])
    )
    min_frames = [min(int(a.min_fill_ratio * seconds * 15), int(a.max_span_seconds * 15)) for seconds in expected]
    result = llm.edit(
        codes,
        a.target_text,
        spans,
        TTSSamplingParams(seed=a.seed, max_tokens=int(a.max_span_seconds * 15) + llm.n_codebooks + 1),
        language=a.language,
        text_normalization=not a.no_text_normalization,
        windows=windows,
        min_frames=min_frames,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.out, result["audio"], 44100, subtype="FLOAT")
    report = {
        "original_text": original,
        "spans_seconds": [[start / 15, (start + length) / 15] for start, length in spans],
        "fill_seconds": [len(fill) / 15 for fill in result["fills"]],
        "stopped": result["stopped"],
        "expected_seconds": expected,
    }
    if windows is not None:
        report["contexts"] = [[w.start / 15, w.end / 15] for w in windows]
    if a.verify:
        heard = transcribe_audio(asr, aligner, a.out, lang[1])
        scored = textmatch.verify(a.target_text, heard.text)
        report.update(asr_text=heard.text, cer=scored["cer"], wer=scored["word_errors"] / max(1, scored["ref_words"]))
        if new_indices:
            report.update(
                new_word_confidence_edited=new_word_confidence(
                    edit_aligned_words(asr, aligner, a.out, a.target_text, lang, heard.utterances), new_indices
                ),
                new_word_confidence_original=new_word_confidence(
                    edit_aligned_words(asr, aligner, a.audio, a.target_text, lang), new_indices
                ),
            )
    asr.close()
    aligner.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
