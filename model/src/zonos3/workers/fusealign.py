#!/usr/bin/env python3
"""FuseAlign forced alignment (third_party/fusealign-multilingual, package ``fusealign_hub``) in a persistent process.

Runs with the workspace interpreter and ``PYTHONPATH=<model/src>``. Reads one JSON request per stdin line and answers
each with one JSON line on stdout, in order, until stdin closes: ``{"id", "status": "ok", ...}`` or ``{"id", "status":
"error", "error", "message", "invalid"}`` ("invalid": a ValueError, i.e. the input is at fault). The model loads on
the first ``align``, so ``ping``, ``speech`` and ``segments`` (which need only Silero VAD) answer without it.

Ops:
  ping                                          -> {"model": checkpoint id (Zyphra/fusealign/<file>, or the local path)}
  speech {items: [{audio}]}                     -> {"results": [{"regions": [[start, end], ...]}]}
  segments {items: [{audio}]}                   -> {"results": [{"chunks": [[start, end], ...]}]}
  align {items: [{audio, tokens, utterances?}]} -> {"results": [{"words": [{word, start, end, presence, placed}]}]}

``audio`` is the path of any sound file soundfile reads (channels are averaged).

``speech`` is where speech is in a recording: the Silero VAD speech regions (``fusealign_hub.vad.speech_regions`` of
the 16 kHz mono audio, in seconds, in time order). No speech gives no regions.

``segments`` cuts a recording into the pieces speech recognition transcribes for long-form anchors: its ``speech``
regions packed greedily, in order, into
chunks spanning (first region start .. last region end) at most ``CHUNK_SECONDS``; a region longer than that is split
into the fewest equal parts of at most ``CHUNK_SECONDS``, each a chunk of its own. No speech gives no chunks.

``align`` places ``tokens`` (the words, zonos3.tts.align_text.align_tokens; the worker never segments): one word per
token, in order, with times in seconds; an unplaced word keeps FuseAlign's fallback times. ``utterances``
(``[{text, start, end}]``, speech recognition of the recording's ``segments`` chunks) are the approximate word times a
recording over 30 s or a transcript over ``fusealign_hub`` MAX_TEXT_BYTES is routed by (``guide_words``); without
them such an item falls back to FuseAlign's VAD character-time map. FuseAlign's built-in Whisper anchoring never runs.
"""
import argparse
import json
import math
import os
import sys
from typing import Callable, Iterable, TextIO

CHUNK_SECONDS = 25.0
SAMPLE_RATE = 16000


def read_mono(path: str):
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), int(sr)


def pack_regions(regions: list[tuple[float, float]], limit: float = CHUNK_SECONDS) -> list[list[float]]:
    """Speech regions ``(start, end)`` in time order -> chunks ``[start, end]`` (see ``segments`` above)."""
    chunks: list[list[float]] = []
    current: list[float] | None = None
    for start, end in regions:
        start, end = float(start), float(end)
        if end - start > limit:
            if current is not None:
                chunks.append(current)
                current = None
            n = math.ceil((end - start) / limit)
            step = (end - start) / n
            chunks.extend([start + i * step, end if i == n - 1 else start + (i + 1) * step] for i in range(n))
        elif current is not None and end - current[0] <= limit:
            current[1] = end
        else:
            if current is not None:
                chunks.append(current)
            current = [start, end]
    if current is not None:
        chunks.append(current)
    return chunks


def _items(op: str, items, fields: set[str]) -> list[dict]:
    if not isinstance(items, list):
        raise ValueError(f"{op}: items must be a list")
    for k, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("audio"), str):
            raise ValueError(f"{op}: items[{k}] needs an audio path")
        unknown = set(item) - fields
        if unknown:
            raise ValueError(f"{op}: items[{k}] has unknown fields {sorted(unknown)}")
    return items


def speech(items, speech_regions: Callable, op: str = "speech") -> list[dict]:
    """Per item ``{"regions": [[start, end], ...]}``; ``speech_regions(wav16k)`` is ``fusealign_hub.vad.speech_regions``
    (``[{start, end}]`` seconds)."""
    import torch
    import torchaudio

    results = []
    for item in _items(op, items, {"audio"}):
        data, sr = read_mono(item["audio"])
        wav = torch.from_numpy(data)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        results.append({"regions": [[float(r["start"]), float(r["end"])] for r in speech_regions(wav)]})
    return results


def segments(items, speech_regions: Callable) -> list[dict]:
    """Per item ``{"chunks": [[start, end], ...]}``: its ``speech`` regions packed (``pack_regions``)."""
    return [{"chunks": pack_regions(r["regions"])} for r in speech(items, speech_regions, op="segments")]


def align(aligner, items, guide_words: Callable) -> list[dict]:
    """Per item ``{"words": [...]}``; items sharing a sample rate share batched forwards. ``guide_words`` is
    ``fusealign_hub.longform.guide_words``."""
    clips = []
    for k, item in enumerate(_items("align", items, {"audio", "tokens", "utterances"})):
        tokens, utterances = item.get("tokens"), item.get("utterances")
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError(f"align: items[{k}].tokens must be a list of strings")
        if utterances is not None and not (isinstance(utterances, list) and all(
                isinstance(u, dict) and isinstance(u.get("text"), str)
                and all(isinstance(u.get(key), (int, float)) and not isinstance(u.get(key), bool)
                        for key in ("start", "end")) for u in utterances)):
            raise ValueError(f"align: items[{k}].utterances must be a list of {{text, start, end}}")
        # Never None: FuseAlign runs Whisper for a long-form item without approximate word times.
        anchors = guide_words(utterances) if utterances is not None else []
        clips.append((*read_mono(item["audio"]), tokens, anchors))
    groups: dict[int, list[int]] = {}
    for k, (_, sr, _, _) in enumerate(clips):
        groups.setdefault(sr, []).append(k)
    results: list[dict] = [{"words": []} for _ in clips]
    for sr, ks in groups.items():
        out = aligner.align_batch([clips[k][0] for k in ks], sr, [clips[k][2] for k in ks],
                                  asr_words=[clips[k][3] for k in ks], language=None)
        for k, words in zip(ks, out):
            results[k] = {"words": [{"word": w.word, "start": float(w.start), "end": float(w.end),
                                     "presence": float(w.presence), "placed": bool(w.placed)} for w in words]}
    return results


def no_whisper(*_args, **_kwargs):
    raise RuntimeError("FuseAlign's Whisper anchoring is disabled: long-form anchors come from Qwen3-ASR utterances")


def serve(lines: Iterable[str], out: TextIO, model_id: str, load: Callable[[], object],
          speech_regions: Callable, guide_words: Callable) -> None:
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
            elif op == "speech":
                response = {"results": speech(request.get("items"), speech_regions)}
            elif op == "segments":
                response = {"results": segments(request.get("items"), speech_regions)}
            elif op == "align":
                if aligner is None:
                    aligner = load()
                    aligner.whisper_words = no_whisper  # a guard: anchors are always supplied
                response = {"results": align(aligner, request.get("items"), guide_words)}
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
    from fusealign_hub.longform import guide_words
    from fusealign_hub.vad import speech_regions

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model_id = args.checkpoint or f"{HF_REPO}/{CHECKPOINT_FILE}"
    serve(sys.stdin, protocol, model_id, lambda: load_aligner(args.checkpoint, device=device),
          speech_regions, guide_words)


if __name__ == "__main__":
    main()
