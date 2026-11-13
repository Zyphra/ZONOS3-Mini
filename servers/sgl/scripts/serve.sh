#!/bin/bash
# Exact-mode ZONOS3 server on 127.0.0.1:1919 (sources servers/sgl/scripts/env.sh). Extra flags are appended.
# Defaults serve the ZONOS3 Mini release LM (hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini) with the voices in
# assets/default_voices/ and the steering bundle assets/steering/studio_bundle_mect60k_x10.json (the studio's controls). Overrides:
#   SERVE_MODEL                 checkpoint dir or hf:// location (--model-path)
#   SERVE_TEXT_ROWS             phoneme / start / pause text rows (--phoneme-rows, zonos-text-extra/1); default the
#                               release rows with the release LM (SERVE_MODEL unset), none otherwise; empty loads none
#   SERVE_VOICES                default-voice dir (--tts-default-voices-dir)
#   STEER_BUNDLE                steering bundle (--steer-bundle); empty serves no controls
#   PHONEME_LEXICON             gold pronunciation lexicon (--tts-phoneme-lexicon: POST /v2/phonemize)
#   ZONOS3_STUDIO_PROJECTS      shared studio project store (--studio-projects-dir); empty keeps projects in the browser
#   LACI_HEADS                  "lang=path ..." LACI alignment heads (--laci-heads)
#   LACI_ALL=1                  LACI heads for every server language from LACI_HEADS_DIR (default assets/laci_heads/): the
#                               default *_heads.json, overridden by a *_heads_<speech code>.json (the zh file for Chinese)
#   PLUGINS                     "name ..." server plugins (--plugins), e.g. dub (Zyphra/ZONOS3-dub)
# Speech editing on the release LM: --accent-adapters <dir with the edit adapter> (docs/guide.md); its rows go before
# the release text rows (pronunciation groups, the start marker, pause markers).
# Speech recognition (speech editing and plugins) is Qwen3-ASR in .venv-asr (tools/setup_asr_env.sh); ZONOS3_ASR_PYTHON
# and ZONOS3_ASR_DEVICE override its interpreter and its device.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
serve_package=zonos3-sgl
source "$repo_root/servers/sgl/scripts/env.sh"

cd "$repo_root"
steer_bundle="${STEER_BUNDLE-$repo_root/assets/steering/studio_bundle_mect60k_x10.json}"
if [[ -z "${SERVE_MODEL:-}" ]]; then
    text_rows="${SERVE_TEXT_ROWS-hf://Zyphra/ZONOS3-Mini/lm/zonos3_mini/text_rows/pause_tokens_s5000.pt}"
else
    text_rows="${SERVE_TEXT_ROWS:-}"
fi
if [[ -n "$steer_bundle" ]]; then
    # The studio embeds every library voice and every upload: run the speaker encoder on the GPU.
    export ZONOS3_SPEAKER_EMBEDDER_DEVICE="${ZONOS3_SPEAKER_EMBEDDER_DEVICE:-cuda}"
fi
source "$repo_root/servers/serve-common.sh"
exec uv run --locked --package "$serve_package" --group phonemize python -m zonos3_sgl \
    ${SERVE_MODEL:+--model-path "$SERVE_MODEL"} \
    --attention-backend fa --cache-type naive \
    --host 127.0.0.1 --port 1919 \
    --max-running-requests 2 --cuda-graph-max-bs 2 \
    --max-prefill-length 256 --memory-ratio 0.5 \
    --tts-default-voices-dir "${SERVE_VOICES:-$repo_root/assets/default_voices}" \
    ${steer_bundle:+--steer-bundle "$steer_bundle"} \
    ${PHONEME_LEXICON:+--tts-phoneme-lexicon "$PHONEME_LEXICON"} \
    ${studio_projects:+--studio-projects-dir "$studio_projects"} \
    ${laci_args[@]+"${laci_args[@]}"} \
    ${plugin_args[@]+"${plugin_args[@]}"} \
    ${text_rows:+--phoneme-rows "$text_rows"} \
    "$@"
