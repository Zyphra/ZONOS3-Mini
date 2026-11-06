#!/usr/bin/env bash
# One-time (re-runnable): .venv-asr, the interpreter of the speech-recognition worker (zonos3/workers/asr.py: Qwen3-ASR,
# used by speech editing and plugins), with its weights. qwen-asr pins its own transformers, hence the separate venv.
# Forced alignment is FuseAlign's (fusealign-hub, installed in the workspace by `uv sync`), not this venv's.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-asr"
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" torch==2.9.1 qwen-asr==0.0.6
HF_HUB_OFFLINE=0 "$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-ASR-1.7B"))
PY
