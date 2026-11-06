# Shared flags of servers/sgl/scripts/serve.sh and servers/mlx/scripts/serve.sh, sourced with repo_root (the checkout)
# and serve_package (the server's uv package) set. Sets laci_args, plugin_args and studio_projects from:
#   LACI_HEADS                  "lang=path ..." LACI alignment heads (--laci-heads)
#   LACI_ALL=1                  LACI heads for every server language from LACI_HEADS_DIR (default laci_heads/): the
#                               default *_heads.json, overridden by a *_heads_<speech code>.json (python -m zonos3.tts.laci)
#   PLUGINS                     "name ..." server plugins (--plugins)
#   ZONOS3_STUDIO_PROJECTS      shared studio project store (--studio-projects-dir)

laci_args=()
if [[ -n "${LACI_HEADS:-}" ]]; then
    read -r -a laci_specs <<<"$LACI_HEADS"
    laci_args=(--laci-heads "${laci_specs[@]}")
elif [[ "${LACI_ALL:-0}" == 1 ]]; then
    laci_dir="${LACI_HEADS_DIR:-$repo_root/laci_heads}"
    specs=$(uv run --locked --package "$serve_package" python -m zonos3.tts.laci --all "$laci_dir") || exit 1
    read -r -a laci_specs <<<"$specs"
    laci_args=(--laci-heads "${laci_specs[@]}")
fi
plugin_args=()
if [[ -n "${PLUGINS:-}" ]]; then
    read -r -a plugin_names <<<"$PLUGINS"
    plugin_args=(--plugins "${plugin_names[@]}")
fi
studio_projects="${ZONOS3_STUDIO_PROJECTS:-}"
