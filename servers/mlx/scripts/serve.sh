#!/bin/bash
# MLX ZONOS3 server on 127.0.0.1:1919 (Apple Silicon). Extra flags are appended.
# Serves the GGUF files of the gguf/ folder of hf://Zyphra/ZONOS3-Mini (the ggml server's files) from MODELS_DIR, with the voices in
# assets/default_voices/. Companions (zac, spk-encoder, zac-encoder, ssl-teacher, steer, accents) are picked up
# next to the backbone. Overrides:
#   MODELS_DIR                  directory holding the GGUFs (default servers/ggml/gguf)
#   SERVE_MODEL                 backbone GGUF (default $MODELS_DIR/backbone-ptq.gguf)
#   SERVE_VOICES                default-voice dir (--tts-default-voices-dir)
#   PHONEME_LEXICON             gold pronunciation lexicon (--tts-phoneme-lexicon: POST /v2/phonemize)
#   ZONOS3_STUDIO_PROJECTS      shared studio project store (--studio-projects-dir); empty keeps projects in the browser
#   WORKER_PYTHON               interpreter of the helper workers (--worker-python): the FuseAlign forced aligner
#                               (needs fusealign-hub; default this venv's interpreter; ZONOS3_FUSEALIGN_DEVICE, default
#                               cpu on macOS)
#   LACI_HEADS                  "lang=path ..." LACI alignment heads (--laci-heads)
#   LACI_ALL=1                  LACI heads for every server language from LACI_HEADS_DIR (default assets/laci_heads/): the
#                               default *_heads.json, overridden by a *_heads_<speech code>.json (one per non-English language)
#   PLUGINS                     "name ..." server plugins (--plugins; e.g. dub, from Zyphra/ZONOS3-dub)
# Speech recognition (speech editing and plugins) is Qwen3-ASR in .venv-asr (tools/setup_asr_env.sh); ZONOS3_ASR_PYTHON
# and ZONOS3_ASR_DEVICE (default mps on macOS) override its interpreter and its device.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
serve_package=zonos3-mlx
cd "$repo_root"
models="${MODELS_DIR:-$repo_root/servers/ggml/gguf}"
model="${SERVE_MODEL:-$models/backbone-ptq.gguf}"
if [[ ! -f "$model" ]]; then
    echo "missing $model: hf download Zyphra/ZONOS3-Mini --include 'gguf/*' --local-dir servers/ggml" >&2
    exit 1
fi
source "$repo_root/servers/serve-common.sh"
exec uv run --locked --package "$serve_package" --group phonemize python -m zonos3_mlx "$model" \
    --host 127.0.0.1 --port 1919 \
    --tts-default-voices-dir "${SERVE_VOICES:-$repo_root/assets/default_voices}" \
    ${PHONEME_LEXICON:+--tts-phoneme-lexicon "$PHONEME_LEXICON"} \
    ${WORKER_PYTHON:+--worker-python "$WORKER_PYTHON"} \
    ${studio_projects:+--studio-projects-dir "$studio_projects"} \
    ${laci_args[@]+"${laci_args[@]}"} \
    ${plugin_args[@]+"${plugin_args[@]}"} \
    "$@"
