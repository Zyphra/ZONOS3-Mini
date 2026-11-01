#!/usr/bin/env python3
"""Edit speech with automatic MMS word-diff regions or explicit second spans.

Clips longer than --max-context-seconds are edited on word-snapped windows around the regions
(--context-seconds of unedited audio per side), planned over an MMS alignment, as /tts/edit does.
"""
import argparse
import json
import math
from pathlib import Path
import soundfile as sf
from zonos3.tts.edit_align import run_mms, expect_ok, aligned_words, diff_spans, expected_fill_seconds, new_word_confidence, plan_edit_windows, seconds_to_spans, target_normalize_request
from zonos3.tts.llm import TTSLLM, TTSSamplingParams
from zonos3.tokenizer.dual_reference import encode_dual_reference_file


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
    duration = sf.info(a.audio).duration
    windowed = duration > a.max_context_seconds
    target_words = a.target_text.split()
    original, new_indices, response = a.original_text, [], None
    if not a.span or windowed or (a.min_fill_ratio > 0 and original):
        if duration > 150:
            p.error('MMS alignment supports clips up to 150 s')
        target_texts, assemble = target_normalize_request(a.target_text)   # pronunciation groups are not normalized
        request = dict(id='orig', op='align' if original else 'transcribe', audio=a.audio, normalize=target_texts)
        if original:
            request['text'] = original
        response = expect_ok(run_mms([request])[0])
        response['normalized'] = assemble(response['normalized'])
        if original is None:
            original = response['text']
            print('MMS original transcript: ' + original)
    if a.span:
        spans = seconds_to_spans(a.span, math.ceil(duration * 15))
    else:
        diff = diff_spans(aligned_words(response), response['normalized'][0], total_frames=math.ceil(duration * 15), margin_seconds=a.margin)
        spans, new_indices = diff.spans, diff.new_target_tokens
    llm = TTSLLM(a.model_path, attention_backend='fa', cache_type='naive', max_running_req=2, cuda_graph_max_bs=2, memory_ratio=0.5)
    codes = encode_dual_reference_file(a.audio, max_seconds=600)
    windows = None
    if windowed:
        windows = plan_edit_windows(spans, aligned_words(response), target_words, response['normalized'][1:], total_frames=len(codes),
                                    context_seconds=a.context_seconds, target_text=a.target_text)
    expected = ([length / 15 for _, length in spans] if response is None else
                expected_fill_seconds(spans, aligned_words(response), target_words, response['normalized'][1:]))
    min_frames = [min(int(a.min_fill_ratio * seconds * 15), int(a.max_span_seconds * 15)) for seconds in expected]
    result = llm.edit(codes, a.target_text, spans, TTSSamplingParams(seed=a.seed, max_tokens=int(a.max_span_seconds * 15) + llm.n_codebooks + 1), language=a.language, text_normalization=not a.no_text_normalization, windows=windows, min_frames=min_frames)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.out, result['audio'], 44100, subtype='FLOAT')
    report = dict(original_text=original, spans_seconds=[[start / 15, (start + length) / 15] for start, length in spans], fill_seconds=[len(fill) / 15 for fill in result['fills']], stopped=result['stopped'], expected_seconds=expected)
    if windows is not None:
        report['contexts'] = [[w.start / 15, w.end / 15] for w in windows]
    if a.verify:
        requests = [dict(id='asr', op='transcribe', audio=a.out, reference=a.target_text), dict(id='edited', op='align', audio=a.out, text=a.target_text), dict(id='original', op='align', audio=a.audio, text=a.target_text)]
        responses = [expect_ok(r) for r in run_mms(requests, device='cpu')]
        asr = responses[0]
        report.update(asr_text=asr['text'], cer=asr['cer'], wer=asr['word_errors'] / max(1, asr['ref_words']))
        if new_indices:
            report.update(new_word_confidence_edited=new_word_confidence(aligned_words(responses[1]), new_indices), new_word_confidence_original=new_word_confidence(aligned_words(responses[2]), new_indices))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
