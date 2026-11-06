#!/usr/bin/env python3
"""TTS text normalization for the ggml and MLX servers (zonos3.tokenizer.textnorm in a persistent process).

Reads one JSON request per stdin line, ``{"id": int, "text": str, "language": str}`` (a server language code), and
answers each with one JSON line on stdout, in order, until stdin closes: ``{"id": int, "text": str}`` or
``{"id": int, "error": str}``. The NeMo grammars load once per language and stay loaded. ``TextNormalizerClient`` is
the Python side of the protocol (the MLX server's; the ggml server's is textnorm.cpp).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path


class TextNormalizerClient:
    """JSON-lines client of this script in its own interpreter: one request per line, answered in order; restarted
    once after the worker died."""

    def __init__(self, python: str, python_src: Path):
        self.python, self.python_src = str(python), Path(python_src)
        self.script = self.python_src / "zonos3" / "workers" / "textnorm.py"
        if not self.script.is_file():
            raise FileNotFoundError(f"{self.script} does not exist (--python-src)")
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._next_id = 0

    def _start(self) -> subprocess.Popen:
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        env["PYTHONPATH"] = str(self.python_src)
        return subprocess.Popen([self.python, str(self.script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=None, env=env, text=True, encoding="utf-8", bufsize=1)

    def _call(self, request: dict) -> dict:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = self._start()
        self._proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise BrokenPipeError("text normalization worker exited")
        return json.loads(line)

    def normalize(self, text: str, language: str) -> str:
        with self._lock:
            self._next_id += 1
            request = {"id": self._next_id, "text": text, "language": language}
            for attempt in range(2):
                try:
                    response = self._call(request)
                    break
                except (OSError, ValueError) as exc:
                    if self._proc is not None:
                        self._proc.kill()
                    self._proc = None
                    if attempt:
                        raise RuntimeError(f"text normalization failed: {exc}") from exc
        if "error" in response:
            raise ValueError(f"text normalization failed: {response['error']}")
        if not isinstance(response.get("text"), str):
            raise RuntimeError("text normalization failed: the worker sent no text")
        return response["text"]

    def close(self) -> None:
        with self._lock:
            if self._proc is not None:
                self._proc.kill()
                self._proc = None


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
