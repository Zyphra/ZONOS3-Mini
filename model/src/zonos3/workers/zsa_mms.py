#!/usr/bin/env python3
"""MMS alignment and greedy CTC recognition in the zsa environment.

Reads one JSON request per stdin line and answers each with one JSON line on
stdout, in order, until stdin closes; the model loads once per process, with the
MMS-1B-all adapter of ``--lang`` (a Whisper-style code, zsa's ``WHISPER_TO_MMS``).
"""
import argparse
import json
import os
import sys
import torch
import soundfile
from z_speech_alignment.mms import WHISPER_TO_MMS, load_mms
from z_speech_alignment.textmatch import normalize, char_error_rate


def word_errors(reference, hypothesis):
    prev = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        row = [i]
        for j, hyp in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (ref != hyp)))
        prev = row
    return prev[-1]


def answer(aligner, request):
    if request['op'] == 'normalize':  # zsa tokens of request['normalize'] only, no audio
        return {'id': request['id'], 'normalized': [normalize(s).split() for s in request['normalize']]}
    response = {'id': request['id'], 'duration': soundfile.info(request['audio']).duration}
    if request['op'] == 'align':
        words = aligner.align_words(request['audio'], request['text'])
    elif request['op'] == 'transcribe':
        text, words = aligner.transcribe(request['audio'])
        response['text'] = text
        if 'reference' in request:
            ref = normalize(request['reference']).split()
            response.update(cer=char_error_rate(request['reference'], text), word_errors=word_errors(ref, normalize(text).split()), ref_words=len(ref))
    else:
        raise ValueError('Unknown MMS operation')
    response['words'] = [dict(word=w.word, start=w.start, end=w.end, confidence=w.confidence, tokens=normalize(w.word).split()) for w in words]
    if 'normalize' in request:
        response['normalized'] = [normalize(s).split() for s in request['normalize']]
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--lang', default='en', help='Whisper-style language code (en, fr, zh, ...)')
    args = parser.parse_args()
    # Keep stdout for responses only: library prints go to stderr.
    protocol = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    if args.lang not in WHISPER_TO_MMS:
        sys.exit(f'no MMS-1B-all adapter for language {args.lang!r}')
    aligner = load_mms(device=args.device, lang=WHISPER_TO_MMS[args.lang])
    for line in sys.stdin:
        request = json.loads(line)
        try:
            response = answer(aligner, request)
        except Exception as exc:
            response = {'id': request['id'], 'error': repr(exc)}
        protocol.write(json.dumps(response) + '\n')


if __name__ == '__main__':
    main()
