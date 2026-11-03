#!/bin/bash
# MLX ZONOS3 server on 127.0.0.1:1919 (Apple Silicon). Extra flags are appended.
# Serves the GGUF files of hf://Zyphra/ZONOS3_Mini_GGUF (the ggml server's) from MODELS_DIR, with the voices in
# default_voices/. Companions (zac, spk-encoder, zac-encoder, ssl-teacher, steer, accents) are picked up
# next to the backbone. Overrides:
#   MODELS_DIR                  directory holding the GGUFs (default servers/ggml/models)
#   SERVE_MODEL                 backbone GGUF (default $MODELS_DIR/zonos3-ptq.gguf)
#   SERVE_VOICES                default-voice dir (--tts-default-voices-dir)
#   PHONEME_LEXICON             gold pronunciation lexicon (--tts-phoneme-lexicon: POST /v2/phonemize)
#   ZONOS3_STUDIO_PROJECTS      shared studio project store (--studio-projects-dir); empty keeps projects in the browser
#   WORKER_PYTHON               interpreter of the helper workers (--worker-python): text normalization (needs pynini;
#                               off without it) and the FuseAlign forced aligner (needs fusealign-hub; default this
#                               venv's interpreter; ZONOS3_FUSEALIGN_DEVICE, default cpu on macOS)
#   LACI_HEADS                  "lang=path ..." LACI alignment heads (--laci-heads)
#   DUB=1                       serve the studio's Dub screen: LACI heads for every dub language from LACI_HEADS_DIR
#                               (default laci_heads/) and the word aligner / AV-sync of tools/setup_asr_env.sh
# Speech recognition (speech editing and dubbing) is Qwen3-ASR in .venv-asr (tools/setup_asr_env.sh); ZONOS3_ASR_PYTHON,
# ZONOS3_ASR_DEVICE (default mps on macOS) and ZONOS3_DUB_AVSYNC_HOME override its interpreter, its device and the
# AV-sync checkout.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
models="${MODELS_DIR:-$repo_root/servers/ggml/models}"
model="${SERVE_MODEL:-$models/zonos3-ptq.gguf}"
if [[ ! -f "$model" ]]; then
    echo "missing $model: hf download Zyphra/ZONOS3_Mini_GGUF --local-dir $models" >&2
    exit 1
fi
laci_args=()
if [[ -n "${LACI_HEADS:-}" ]]; then
    read -r -a laci_specs <<<"$LACI_HEADS"
    laci_args=(--laci-heads "${laci_specs[@]}")
elif [[ "${DUB:-0}" == 1 ]]; then
    laci_dir="${LACI_HEADS_DIR:-$repo_root/laci_heads}"
    laci_en="$laci_dir/zonos3_mini_mect60k_heads.json"
    laci_args=(--laci-heads "cmn=$laci_dir/zonos3_mini_mect60k_heads_zh.json")
    for lang in en_us fr_fr de es it pt_br ja ko; do laci_args+=("$lang=$laci_en"); done
fi
studio_projects="${ZONOS3_STUDIO_PROJECTS:-}"
exec uv run --locked --package zonos3-mlx --group phonemize python -m zonos3_mlx "$model" \
    --host 127.0.0.1 --port 1919 \
    --tts-default-voices-dir "${SERVE_VOICES:-$repo_root/default_voices}" \
    ${PHONEME_LEXICON:+--tts-phoneme-lexicon "$PHONEME_LEXICON"} \
    ${WORKER_PYTHON:+--worker-python "$WORKER_PYTHON"} \
    ${studio_projects:+--studio-projects-dir "$studio_projects"} \
    ${laci_args[@]+"${laci_args[@]}"} \
    "$@"
