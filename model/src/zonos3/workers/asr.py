#!/usr/bin/env python3
"""Qwen3-ASR speech recognition (speech editing and plugins) in a persistent process.

Runs in .venv-asr (tools/setup_asr_env.sh; qwen-asr pins its own transformers) with ``PYTHONPATH=<model/src>``. Reads
one JSON request per stdin line (``{"id", "op", ...}``) and answers each with one JSON line on stdout, in order, until
stdin closes: ``{"id", "status": "ok", ...}`` or ``{"id", "status": "error", "error", "message", "invalid"}``
("invalid": a ValueError, i.e. the input is at fault). The model loads on the first ``transcribe``, so ``ping`` answers
at once.

Ops:
  ping                                          -> {"model": the ASR model id, "languages": [Qwen3-ASR language names]}
  transcribe {items: [{audio, start?, end?,     -> {"results": [{"text", "language"}]}  (one per item, in order)
                       language}]}

``audio`` is the path of any sound file soundfile reads (channels are averaged); ``start`` / ``end`` (seconds, default
the file's start / end, clipped to it) select the part to transcribe; ``language`` is a Qwen3-ASR language name of
``ping``'s list (``"English"``, ``"Japanese"``, ...) or null to detect it. ``text`` is the stripped transcript and
``language`` the language Qwen3-ASR reports (the forced one, or the detected ones joined with ","; "" for an empty
selection). The items of one request share batched forwards.
"""

import argparse
import json
import os
import sys
from typing import Callable, Iterable, TextIO

MODEL = "Qwen/Qwen3-ASR-1.7B"


def read_part(path: str, start, end):
    """Mono float32 samples of ``[start, end)`` seconds of a sound file, and its sample rate."""
    import soundfile as sf

    with sf.SoundFile(path) as f:
        sr, frames = int(f.samplerate), f.frames
        first = 0 if start is None else min(frames, max(0, round(float(start) * sr)))
        last = frames if end is None else min(frames, max(first, round(float(end) * sr)))
        f.seek(first)
        data = f.read(last - first, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def transcribe(model, languages: list[str], items: list) -> list[dict]:
    """Per item ``{"text", "language"}``; ``model.transcribe`` is qwen_asr's ``Qwen3ASRModel.transcribe``."""
    if not isinstance(items, list):
        raise ValueError("transcribe: items must be a list")
    parts, langs = [], []
    for k, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("audio"), str):
            raise ValueError(f"transcribe: items[{k}] needs an audio path")
        unknown = set(item) - {"audio", "start", "end", "language"}
        if unknown:
            raise ValueError(f"transcribe: items[{k}] has unknown fields {sorted(unknown)}")
        for key in ("start", "end"):
            if item.get(key) is not None and (isinstance(item[key], bool) or not isinstance(item[key], (int, float))):
                raise ValueError(f"transcribe: items[{k}].{key} must be a number of seconds or null")
        language = item.get("language")
        if language is not None and language not in languages:
            raise ValueError(
                f"transcribe: items[{k}].language {language!r} is not a Qwen3-ASR language "
                f"(supported: {', '.join(languages)})"
            )
        parts.append(read_part(item["audio"], item.get("start"), item.get("end")))
        langs.append(language)
    results = [{"text": "", "language": ""} for _ in items]
    voiced = [k for k, (wav, _) in enumerate(parts) if wav.size]
    if voiced:
        out = model.transcribe(audio=[parts[k] for k in voiced], language=[langs[k] for k in voiced])
        for k, r in zip(voiced, out, strict=True):
            results[k] = {"text": (r.text or "").strip(), "language": r.language or ""}
    return results


def serve(lines: Iterable[str], out: TextIO, model_id: str, languages: list[str], load: Callable[[], object]) -> None:
    """Answer the requests of ``lines`` on ``out``; ``load()`` builds the model on the first ``transcribe``."""
    model = None
    for line in lines:
        if not line.strip():
            continue
        rid = None
        try:
            request = json.loads(line)
            rid = request.get("id")
            op = request.get("op")
            if op == "ping":
                response = {"model": model_id, "languages": list(languages)}
            elif op == "transcribe":
                if model is None:
                    model = load()
                response = {"results": transcribe(model, languages, request.get("items"))}
            else:
                raise ValueError(f"unknown op {op!r}")
            response.update(id=rid, status="ok")
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the worker
            response = {
                "id": rid,
                "status": "error",
                "error": repr(exc),
                "message": str(exc),
                "invalid": isinstance(exc, ValueError),
            }
        out.write(json.dumps(response, ensure_ascii=False) + "\n")
        out.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--device",
        default=os.environ.get("ZONOS3_ASR_DEVICE"),
        help="torch device (default: $ZONOS3_ASR_DEVICE, else cuda:0 when available, else cpu)",
    )
    parser.add_argument("--model", default=MODEL, help=f"Qwen3-ASR checkpoint (default: {MODEL})")
    args = parser.parse_args()
    # Keep stdout for responses only: library prints go to stderr.
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import torch
    from qwen_asr.inference.utils import SUPPORTED_LANGUAGES

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def load():
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel.from_pretrained(
            args.model,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            device_map=device,
            max_inference_batch_size=32,
            max_new_tokens=1024,
        )

    with torch.no_grad():
        serve(sys.stdin, protocol, args.model, SUPPORTED_LANGUAGES, load)


if __name__ == "__main__":
    main()
