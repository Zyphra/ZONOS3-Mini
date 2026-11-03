#!/usr/bin/env python3
"""FuseAlign forced alignment (third_party/fusealign-multilingual, package ``fusealign_hub``) in a persistent process.

Runs with the workspace interpreter and ``PYTHONPATH=<model/src>``. Reads one JSON request per stdin line and answers
each with one JSON line on stdout, in order, until stdin closes: ``{"id", "status": "ok", ...}`` or ``{"id", "status":
"error", "error", "message", "invalid"}`` ("invalid": a ValueError, i.e. the input is at fault). The model loads on
the first ``align``, so ``ping`` answers at once.

Ops:
  ping                                          -> {"model": checkpoint id (Zyphra/fusealign/<file>, or the local path)}
  align {items: [{audio, tokens, language}]}    -> {"results": [{"words": [{word, start, end, presence, placed}]}]}

``audio`` is the path of any sound file soundfile reads (channels are averaged), ``tokens`` the words to place
(zonos3.tts.align_text.align_tokens; the worker never segments) and ``language`` the Whisper language of a recording
over 30 s (long-form anchoring; null auto-detects). Each item gets one word per token, in order, with times in
seconds; an unplaced word keeps FuseAlign's fallback times.
"""
import argparse
import json
import os
import sys
from typing import Callable, Iterable, TextIO


def read_mono(path: str):
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def align(aligner, items: list) -> list[dict]:
    """Per item ``{"words": [...]}``; items sharing a sample rate and language share batched forwards."""
    if not isinstance(items, list):
        raise ValueError("align: items must be a list")
    clips = []
    for k, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("audio"), str):
            raise ValueError(f"align: items[{k}] needs an audio path")
        tokens, language = item.get("tokens"), item.get("language")
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError(f"align: items[{k}].tokens must be a list of strings")
        if language is not None and not isinstance(language, str):
            raise ValueError(f"align: items[{k}].language must be a string or null")
        clips.append((*read_mono(item["audio"]), tokens, language))
    groups: dict[tuple[int, str | None], list[int]] = {}
    for k, (_, sr, _, language) in enumerate(clips):
        groups.setdefault((sr, language), []).append(k)
    results: list[dict] = [{"words": []} for _ in clips]
    for (sr, language), ks in groups.items():
        out = aligner.align_batch([clips[k][0] for k in ks], sr, [clips[k][2] for k in ks], language=language)
        for k, words in zip(ks, out):
            results[k] = {"words": [{"word": w.word, "start": float(w.start), "end": float(w.end),
                                     "presence": float(w.presence), "placed": bool(w.placed)} for w in words]}
    return results


def serve(lines: Iterable[str], out: TextIO, model_id: str, load: Callable[[], object]) -> None:
    """Answer the requests of ``lines`` on ``out``; ``load()`` builds the aligner on the first ``align``."""
    aligner = None
    for line in lines:
        if not line.strip():
            continue
        rid = None
        try:
            request = json.loads(line)
            rid = request.get("id")
            op = request.get("op")
            if op == "ping":
                response = {"model": model_id}
            elif op == "align":
                if aligner is None:
                    aligner = load()
                response = {"results": align(aligner, request.get("items"))}
            else:
                raise ValueError(f"unknown op {op!r}")
            response.update(id=rid, status="ok")
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the worker
            response = {"id": rid, "status": "error", "error": repr(exc), "message": str(exc),
                        "invalid": isinstance(exc, ValueError)}
        out.write(json.dumps(response, ensure_ascii=False) + "\n")
        out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=os.environ.get("ZONOS3_FUSEALIGN_DEVICE"),
                        help="torch device (default: $ZONOS3_FUSEALIGN_DEVICE, else cuda:0 when available, else cpu)")
    parser.add_argument("--checkpoint", default=None,
                        help="a local fusealign-hub/1 safetensors file (default: the Zyphra/fusealign checkpoint)")
    args = parser.parse_args()
    # Keep stdout for responses only: library prints go to stderr.
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import torch
    from fusealign_hub import load_aligner
    from fusealign_hub.aligner import CHECKPOINT_FILE, HF_REPO

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model_id = args.checkpoint or f"{HF_REPO}/{CHECKPOINT_FILE}"
    serve(sys.stdin, protocol, model_id, lambda: load_aligner(args.checkpoint, device=device))


if __name__ == "__main__":
    main()
