#!/usr/bin/env bash
# Upload public/models to the R2 bucket served by the Worker at /models/*.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-}
case "$mode" in --local|--remote) ;; *) echo "usage: $0 --local|--remote" >&2; exit 2 ;; esac
BUCKET=zonos3-studio-models
[ -d public/models ] || { echo "public/models missing; run pnpm export first" >&2; exit 1; }
find public/models -type f ! -name '.*' | sort | while read -r f; do
  key=${f#public/models/}
  case "$key" in
    *.json) ct=application/json ;;
    *.mp3) ct=audio/mpeg ;;
    *.gguf) ct=application/octet-stream ;;
    *) ct=application/octet-stream ;;
  esac
  echo "$key"
  pnpm exec wrangler r2 object put "$BUCKET/$key" --file "$f" --content-type "$ct" "$mode"
done
