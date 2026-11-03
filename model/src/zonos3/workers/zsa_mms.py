#!/usr/bin/env python3
"""MMS greedy CTC recognition and zsa token normalization in the zsa environment (speech editing).

Reads one JSON request per stdin line and answers each with one JSON line on
stdout, in order, until stdin closes; the model loads once per process, with the
MMS-1B-all adapter of ``--lang`` (a Whisper-style code, zsa's ``WHISPER_TO_MMS``).
Forced alignment is FuseAlign's (zonos3/workers/fusealign.py).

Ops (any request may add ``normalize: [str]`` -> ``normalized``: the zsa tokens of each string):
  normalize {normalize}                    -> {"normalized": [[token]]}
  transcribe {audio, reference?}           -> {"duration", "text", "tokens": zsa tokens of the text,
                                               and with a reference "cer", "word_errors", "ref_words"}
Errors answer {"id", "error": repr}.
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
    response = {'id': request['id']}
    if request['op'] == 'transcribe':
        text, _ = aligner.transcribe(request['audio'])
        response.update(duration=soundfile.info(request['audio']).duration, text=text, tokens=normalize(text).split())
        if 'reference' in request:
            ref = normalize(request['reference']).split()
            response.update(cer=char_error_rate(request['reference'], text), word_errors=word_errors(ref, response['tokens']), ref_words=len(ref))
    elif request['op'] != 'normalize':
        raise ValueError('Unknown MMS operation')
    if 'normalize' in request:
        response['normalized'] = [normalize(s).split() for s in request['normalize']]
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
