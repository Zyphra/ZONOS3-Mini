#!/usr/bin/env bash
# One-time (re-runnable): venv of the dubbing alignment worker (Qwen3-ASR + SimAlign) and its weights, plus AV-sync
# (lip-sync step, https://github.com/Ryu1845/AV-sync) at a pinned commit with its models. Forced alignment is FuseAlign's
# (fusealign-hub, installed in the workspace by `uv sync`), not this venv's.
set -euo pipefail
DUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$DUB/.venv-dub-align"
AVSYNC="$DUB/.avsync"
AVSYNC_COMMIT=4edeac8b21add4bcdec98a72a94b3924c0cd53cd
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" torch==2.9.1 qwen-asr==0.0.6 simalign==0.4
HF_HUB_OFFLINE=0 "$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-ASR-1.7B"))
print(snapshot_download("google-bert/bert-base-multilingual-cased",
                        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]))
PY
# AV-sync needs only torch (the worker venv's); its models go to its data/ (download_model.sh)
[ -d "$AVSYNC/.git" ] || git clone https://github.com/Ryu1845/AV-sync.git "$AVSYNC"
git -C "$AVSYNC" fetch --quiet origin
git -C "$AVSYNC" checkout --quiet "$AVSYNC_COMMIT"
[ -f "$AVSYNC/data/syncnet_v2.model" ] && [ -f "$AVSYNC/data/sfd_face.pth" ] || (cd "$AVSYNC" && sh download_model.sh)
