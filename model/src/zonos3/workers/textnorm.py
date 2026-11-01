#!/usr/bin/env python3
"""TTS text normalization for the ggml server (zonos3.tokenizer.textnorm in a persistent process).

Reads one JSON request per stdin line, ``{"id": int, "text": str, "language": str}`` (a server language code), and
answers each with one JSON line on stdout, in order, until stdin closes: ``{"id": int, "text": str}`` or
``{"id": int, "error": str}``. The NeMo grammars load once per language and stay loaded.
"""
import json
import os
import sys


def main() -> None:
    # Keep stdout for responses only: grammar compilation and library prints go to stderr.
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    from zonos3.tokenizer.textnorm import TTSTextNormalizer

    normalizer = TTSTextNormalizer()
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            response = {"id": request["id"], "text": normalizer.normalize(request["text"], request["language"])}
        except Exception as exc:  # noqa: BLE001 - reported to the server, which answers 400
            response = {"id": request.get("id"), "error": f"{type(exc).__name__}: {exc}"}
        protocol.write(json.dumps(response, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
