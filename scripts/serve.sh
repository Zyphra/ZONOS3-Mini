#!/bin/bash
# Exact-mode ZONOS3 server on 127.0.0.1:1919 (sources scripts/env.sh). Extra flags are appended.
# Defaults serve the ZONOS3 Mini release LM (hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/mect_60k) with the voices in
# default_voices/ and the emotion directions in steering/emotion_directions_mect60k_axes. Overrides:
#   SERVE_MODEL                 checkpoint dir or hf:// location (--model-path)
#   SERVE_VOICES                default-voice dir (--tts-default-voices-dir)
#   EMOTION_SERVE_DIRECTIONS    emotion direction set (--tts-emotion-directions-dir)
#   ACCENT_SERVE_DIRECTIONS     accent direction set (--accent-directions: hidden-space or speaker-embedding steering)
#   STEER_BUNDLE                steering bundle (--steer-bundle), e.g. steering/studio_bundle_mect60k.json for /studio
#   PHONEME_LEXICON             gold pronunciation lexicon (--tts-phoneme-lexicon: POST /tts/phonemize)
#   ZONOS3_STUDIO_PROJECTS      shared studio project store (--studio-projects-dir); empty keeps projects in the browser
#   LACI_HEADS                  "lang=path ..." LACI alignment heads (--laci-heads)
#   DUB=1                       serve the studio's Dub screen (/dub/*): LACI heads for every dub language from
#                               LACI_HEADS_DIR (default laci_heads/: the zh file for Chinese, the English file for the
#                               rest) and the alignment worker / AV-sync of scripts/setup_dub_align_env.sh
#                               (ZONOS3_DUB_ALIGN_PYTHON / ZONOS3_DUB_AVSYNC_HOME / ZONOS3_DUB_ALIGN_DEVICE override).
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_root/scripts/env.sh"

cd "$repo_root"
if [[ -n "${STEER_BUNDLE:-}" ]]; then
    # The studio embeds every library voice and every upload: run the speaker encoder on the GPU.
    export ZONOS3_SPEAKER_EMBEDDER_DEVICE="${ZONOS3_SPEAKER_EMBEDDER_DEVICE:-cuda}"
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
if [[ "${DUB:-0}" == 1 ]]; then
    export ZONOS3_DUB_ALIGN_PYTHON="${ZONOS3_DUB_ALIGN_PYTHON:-$repo_root/.venv-dub-align/bin/python}"
    export ZONOS3_DUB_AVSYNC_HOME="${ZONOS3_DUB_AVSYNC_HOME:-$repo_root/.avsync}"
    export ZONOS3_DUB_ALIGN_DEVICE="${ZONOS3_DUB_ALIGN_DEVICE:-cuda}"
fi
studio_projects="${ZONOS3_STUDIO_PROJECTS:-}"
exec uv run --locked --group phonemize python -m zonos3 \
    ${SERVE_MODEL:+--model-path "$SERVE_MODEL"} \
    --attention-backend fa --cache-type naive \
    --host 127.0.0.1 --port 1919 \
    --max-running-requests 2 --cuda-graph-max-bs 2 \
    --max-prefill-length 256 --memory-ratio 0.5 \
    --tts-default-voices-dir "${SERVE_VOICES:-$repo_root/default_voices}" \
    --tts-emotion-directions-dir "${EMOTION_SERVE_DIRECTIONS:-$repo_root/steering/emotion_directions_mect60k_axes}" \
    ${ACCENT_SERVE_DIRECTIONS:+--accent-directions "$ACCENT_SERVE_DIRECTIONS"} \
    ${STEER_BUNDLE:+--steer-bundle "$STEER_BUNDLE"} \
    ${PHONEME_LEXICON:+--tts-phoneme-lexicon "$PHONEME_LEXICON"} \
    ${studio_projects:+--studio-projects-dir "$studio_projects"} \
    ${laci_args[@]+"${laci_args[@]}"} \
    "$@"
