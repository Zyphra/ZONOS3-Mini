#!/usr/bin/env bash
# Exercise local-GGUF launch, optional reference encoders, ffmpeg fetch, and teardown.
#
# Uses SIGTERM (not SIGINT) to test cleanup: background non-interactive shells
# have SIGINT ignored on entry, so its trap can never fire here — interactively
# (the real double-click case) Ctrl-C works through the same trap.
#
# Usage:  bash tests/launcher-smoke.sh [server-bin]
#   binary defaults to auto-discovery under build*/.  Run from anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LAUNCHER="scripts/start-zonos3.sh"
MODEL="tests/fixtures/dummy-zonos3.gguf"
DAC="tests/fixtures/dummy-dac.gguf"

find_bin() {
  if [ -n "${1:-}" ]; then echo "$1"; return; fi
  find build* \( -name zonos3-server -o -name zonos3-server.exe \) -type f 2>/dev/null | head -1
}
SRV="$(find_bin "${1:-}")"
[ -x "$SRV" ]     || { echo "FAIL: zonos3-server not found (pass it as \$1)"; exit 1; }
[ -f "$MODEL" ]   || { echo "FAIL: missing $MODEL"; exit 1; }
[ -f "$DAC" ]     || { echo "FAIL: missing $DAC"; exit 1; }
command -v curl    >/dev/null 2>&1 || { echo "SKIP: curl not available"; exit 0; }
command -v python3 >/dev/null 2>&1 || { echo "SKIP: python3 not available"; exit 0; }

WORK="$(mktemp -d)"
LNCH_PID=""
HTTP_PID=""
cleanup() {
  local rc=$?
  [ -n "$LNCH_PID" ] && { kill "$LNCH_PID" 2>/dev/null || true; wait "$LNCH_PID" 2>/dev/null || true; }
  [ -n "$HTTP_PID" ] && { kill "$HTTP_PID" 2>/dev/null || true; wait "$HTTP_PID" 2>/dev/null || true; }
  rm -rf "$WORK"
  exit "$rc"
}
trap cleanup EXIT
pass() { echo "  ok: $1"; }
fail() { echo "FAIL: $1"; [ -f "$WORK/launch.log" ] && { echo "--- launcher log ---"; cat "$WORK/launch.log"; }; exit 1; }
is_riff() { [ "$(head -c4 "$1" 2>/dev/null)" = "RIFF" ]; }

PORT="${ZONOS3_TEST_PORT:-19188}"
BASE="http://127.0.0.1:$PORT"

# Start the launcher against local models and wait for /v2/health.
run_launcher() {
  ZONOS3_MODEL_DIR="$WORK/models" ZONOS3_SERVER_BIN="$SRV" \
  ZONOS3_PORT="$PORT" ZONOS3_ASSUME_YES=1 ZONOS3_NO_BROWSER=1 \
    bash "$LAUNCHER" -- --max 16 >"$WORK/launch.log" 2>&1 &
  LNCH_PID=$!
  local up=0
  for _ in $(seq 1 150); do
    if curl -fsS "$BASE/v2/health" >/dev/null 2>&1; then up=1; break; fi
    kill -0 "$LNCH_PID" 2>/dev/null || fail "launcher exited before the server came up"
    sleep 0.2
  done
  [ "$up" = 1 ] || fail "server did not answer /v2/health in time"
}

# stop_launcher: TERM the launcher, assert its trap takes the server down.
stop_launcher() {
  kill -TERM "$LNCH_PID" 2>/dev/null || true
  wait "$LNCH_PID" 2>/dev/null || true
  LNCH_PID=""
  local down=0
  for _ in $(seq 1 25); do
    if ! curl -fsS "$BASE/v2/health" >/dev/null 2>&1; then down=1; break; fi
    sleep 0.2
  done
  [ "$down" = 1 ] || fail "server still answering after the launcher was stopped"
}

echo "== launcher: local models =="
mkdir -p "$WORK/models"
cp "$MODEL" "$WORK/models/zonos3-qat.gguf"
cp "$DAC"   "$WORK/models/dac.gguf"
run_launcher
pass "local models -> server up"
curl -fsS "$BASE/v2/generate" \
  -d '{"text":"launcher smoke","stream":false,"format":"wav","seed":1,"max_tokens":16}' \
  -o "$WORK/smoke.wav" || fail "/v2/generate request failed"
is_riff "$WORK/smoke.wav" || fail "/v2/generate did not return a RIFF WAV"
pass "/v2/generate -> WAV"
stop_launcher
pass "TERM tears the server down"

echo "== launcher: ffmpeg auto-download =="
# Serve a fake single-binary ffmpeg and force the download (ZONOS3_SKIP_PATH_FFMPEG bypasses
# the system-ffmpeg check that CI/dev boxes would otherwise satisfy). Models already present,
# so only ffmpeg is fetched.
FF_PORT="${ZONOS3_TEST_FF_PORT:-18124}"
mkdir -p "$WORK/ffserve"
printf '#!/bin/sh\necho fake-ffmpeg "$@"\n' > "$WORK/ffserve/ffmpeg"
python3 -m http.server "$FF_PORT" --bind 127.0.0.1 --directory "$WORK/ffserve" >/dev/null 2>&1 &
HTTP_PID=$!
for _ in $(seq 1 50); do curl -fsS "http://127.0.0.1:$FF_PORT/" >/dev/null 2>&1 && break; sleep 0.2; done
rm -rf "$WORK/models/bin"
export ZONOS3_SKIP_PATH_FFMPEG=1 ZONOS3_FFMPEG_URL="http://127.0.0.1:$FF_PORT/ffmpeg"
run_launcher
[ -x "$WORK/models/bin/ffmpeg" ] || fail "ffmpeg was not downloaded to models/bin"
pass "ffmpeg auto-downloaded to models/bin"
stop_launcher
unset ZONOS3_SKIP_PATH_FFMPEG ZONOS3_FFMPEG_URL
kill "$HTTP_PID" 2>/dev/null || true; wait "$HTTP_PID" 2>/dev/null || true; HTTP_PID=""
echo "== launcher: missing decoder =="
rm "$WORK/models/dac.gguf"
if ZONOS3_MODEL_DIR="$WORK/models" ZONOS3_SERVER_BIN="$SRV" \
   ZONOS3_ASSUME_YES=1 ZONOS3_NO_BROWSER=1 bash "$LAUNCHER" \
   >"$WORK/missing.log" 2>&1; then
  fail "launcher accepted missing dac.gguf"
fi
pass "missing decoder rejected without download"


echo "All launcher smoke tests passed."
