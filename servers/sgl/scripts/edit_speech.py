#!/usr/bin/env python3
"""Edit speech with automatic word-diff regions or explicit second spans.

Clips longer than --max-context-seconds are edited on word-snapped windows around the regions
(--context-seconds of unedited audio per side), planned over a FuseAlign alignment, as /v2/edit does.
MMS (ZSA_ROOT) transcribes the original when --original-text is omitted and normalizes the words to zsa tokens.
"""
import argparse
import json
import math
from pathlib import Path
import soundfile as sf
from zonos3.tts.align_text import align_tokens
from zonos3.tts.edit_align import FuseAlignWorker, run_mms, expect_ok, aligned_words, diff_spans, expected_fill_seconds, mms_language, new_word_confidence, plan_edit_windows, seconds_to_spans, target_normalize_request
from zonos3_sgl.tts.llm import TTSLLM, TTSSamplingParams
from zonos3.tokenizer.dual_reference import encode_dual_reference_file


def words_of(aligner, audio, text, lang, normalize=()):
    """FuseAlign words of ``text`` in ``audio`` with their zsa tokens, and the zsa tokens of the ``normalize`` texts."""
    tokens = align_tokens(text, lang)
    normalized = expect_ok(run_mms([dict(id='words', op='normalize', normalize=list(normalize) + tokens)], language=lang)[0])['normalized']
    aligned = aligner.align([dict(audio=audio, tokens=tokens, language=lang)])[0] if tokens else []
    return aligned_words(aligned, normalized[len(normalize):]), normalized[:len(normalize)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--audio', required=True)
    p.add_argument('--target-text', required=True)
    p.add_argument('--original-text')
    p.add_argument('--span', nargs=2, type=float, action='append')
    p.add_argument('--margin', type=float, default=0.08)
    p.add_argument('--max-span-seconds', type=float, default=30)
    p.add_argument('--max-context-seconds', type=float, default=16)
    p.add_argument('--context-seconds', type=float, default=4)
    p.add_argument('--min-fill-ratio', type=float, default=0.7, help='span-end masked until a fill reaches this share of its expected length')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--language', default='en_us')
    p.add_argument('--no-text-normalization', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    lang = mms_language(a.language)
    aligner = FuseAlignWorker()
    duration = sf.info(a.audio).duration
    windowed = duration > a.max_context_seconds
    target_words = a.target_text.split()
    original, new_indices, words, normalized = a.original_text, [], None, None
    if not a.span or windowed or (a.min_fill_ratio > 0 and original):
        if duration > 150:
            p.error('speech editing supports clips up to 150 s')
        target_texts, assemble = target_normalize_request(a.target_text)   # pronunciation groups are not normalized
        if original is None:
            original = expect_ok(run_mms([dict(id='orig', op='transcribe', audio=a.audio)], language=lang)[0])['text']
            print('MMS original transcript: ' + original)
        words, normalized = words_of(aligner, a.audio, original, lang, target_texts)
        normalized = assemble(normalized)
    if a.span:
        spans = seconds_to_spans(a.span, math.ceil(duration * 15))
    else:
        diff = diff_spans(words, normalized[0], total_frames=math.ceil(duration * 15), margin_seconds=a.margin)
        spans, new_indices = diff.spans, diff.new_target_tokens
    llm = TTSLLM(a.model_path, attention_backend='fa', cache_type='naive', max_running_req=2, cuda_graph_max_bs=2, memory_ratio=0.5)
    codes = encode_dual_reference_file(a.audio, max_seconds=600)
    windows = None
    if windowed:
        windows = plan_edit_windows(spans, words, target_words, normalized[1:], total_frames=len(codes),
                                    context_seconds=a.context_seconds, target_text=a.target_text)
    expected = ([length / 15 for _, length in spans] if words is None else
                expected_fill_seconds(spans, words, target_words, normalized[1:]))
    min_frames = [min(int(a.min_fill_ratio * seconds * 15), int(a.max_span_seconds * 15)) for seconds in expected]
    result = llm.edit(codes, a.target_text, spans, TTSSamplingParams(seed=a.seed, max_tokens=int(a.max_span_seconds * 15) + llm.n_codebooks + 1), language=a.language, text_normalization=not a.no_text_normalization, windows=windows, min_frames=min_frames)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.out, result['audio'], 44100, subtype='FLOAT')
    report = dict(original_text=original, spans_seconds=[[start / 15, (start + length) / 15] for start, length in spans], fill_seconds=[len(fill) / 15 for fill in result['fills']], stopped=result['stopped'], expected_seconds=expected)
    if windows is not None:
        report['contexts'] = [[w.start / 15, w.end / 15] for w in windows]
    if a.verify:
        asr = expect_ok(run_mms([dict(id='asr', op='transcribe', audio=a.out, reference=a.target_text)], device='cpu', language=lang)[0])
        report.update(asr_text=asr['text'], cer=asr['cer'], wer=asr['word_errors'] / max(1, asr['ref_words']))
        if new_indices:
            report.update(new_word_confidence_edited=new_word_confidence(words_of(aligner, a.out, a.target_text, lang)[0], new_indices),
                          new_word_confidence_original=new_word_confidence(words_of(aligner, a.audio, a.target_text, lang)[0], new_indices))
    aligner.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
