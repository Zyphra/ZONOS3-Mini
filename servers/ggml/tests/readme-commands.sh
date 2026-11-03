#!/usr/bin/env bash
# Exercise the command flows shown in README.md / docs/INTERNALS.md against the tiny
# committed fixtures (random-weight backbone + DAC), so the documented commands are
# proven to actually run on CI — not just the usage string.
#
# Audio is gibberish by design; we only assert each command runs and emits the right
# kind of artifact (codes .npy, RIFF WAV, float32 PCM). Everything runs on the CPU
# backend (no --gpu) for determinism, matching the existing smoke test.
#
# Usage:  bash tests/readme-commands.sh [cli-bin] [server-bin]
#   binaries default to auto-discovery under build/.  Run from anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="tests/fixtures/dummy-zonos3.gguf"
DAC="tests/fixtures/dummy-dac.gguf"

find_bin() {  # $1 = base name; honor an explicit path arg, else search build*/
  if [ -n "${2:-}" ]; then echo "$2"; return; fi
  if [ -n "${1:-}" ] && [ -n "${CLI:-}" ]; then
    local sibling="$(dirname "$CLI")/$1"
    [ -x "$sibling" ] && { echo "$sibling"; return; }
    [ -x "$sibling.exe" ] && { echo "$sibling.exe"; return; }
  fi
  find build* \( -name "$1" -o -name "$1.exe" \) -type f 2>/dev/null | head -1
}
CLI="$(find_bin zonos3-cli "${1:-}")"
SRV="$(find_bin zonos3-server "${2:-}")"
[ -x "$CLI" ] || { echo "FAIL: zonos3-cli not found (pass it as \$1)"; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: missing $MODEL"; exit 1; }
[ -f "$DAC" ]   || { echo "FAIL: missing $DAC"; exit 1; }

WORK="$(mktemp -d)"
SRV_PID=""
cleanup() {  # preserve the script's real exit status across server teardown
  local rc=$?
  [ -n "$SRV_PID" ] && { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
  rm -rf "$WORK"
  exit "$rc"
}
trap cleanup EXIT
pass() { echo "  ok: $1"; }
fail() { echo "FAIL: $1"; exit 1; }
is_riff() { [ "$(head -c4 "$1" 2>/dev/null)" = "RIFF" ]; }

echo "== CLI ($CLI) =="

# build the prompt id matrix (docs/INTERNALS.md "Standalone components")
"$CLI" "$MODEL" --build-prompt "hello prompt" "$WORK/ids.npy" >/dev/null 2>&1 \
  || fail "--build-prompt exited nonzero"
[ -s "$WORK/ids.npy" ] || fail "--build-prompt wrote no ids.npy"
pass "--build-prompt -> ids.npy"

# generate codes from the prompt (16 rows: dac-cli below decodes them, past the 9-row shear)
"$CLI" "$MODEL" --generate "$WORK/ids.npy" "$WORK/codes.npy" --max 16 --greedy >/dev/null 2>&1 \
  || fail "--generate exited nonzero"
[ -s "$WORK/codes.npy" ] || fail "--generate wrote no codes.npy"
pass "--generate -> codes.npy"

# one-command text -> codes (.npy output, no DAC)
out="$("$CLI" "$MODEL" --tts "ci tts to codes" "$WORK/t.npy" --max 8 --greedy 2>&1)" \
  || { echo "$out"; fail "--tts (codes) exited nonzero"; }
[ -s "$WORK/t.npy" ] || fail "--tts (codes) wrote no .npy"
pass "--tts -> codes .npy"

# one-command text -> WAV, decoding through the DAC (README Quick Start / CLI section). A WAV needs
# more rows than the 9-row codebook shear, so the decode tests generate 16 frames.
"$CLI" "$MODEL" --tts "ci tts to wav" "$WORK/t.wav" --dac "$DAC" --max 16 --greedy >/dev/null 2>&1 \
  || fail "--tts --dac (wav) exited nonzero"
is_riff "$WORK/t.wav" || fail "--tts --dac did not write a RIFF WAV"
pass "--tts --dac -> WAV"

# speaker-embedding conditioning: speaker row + condition buckets + speaker CFG (two-slot twin) + silence prefill
out="$("$CLI" "$MODEL" --tts "ci speaker" "$WORK/s.wav" --speaker-embedding tests/fixtures/speaker-embedding.npy \
  --loudness -23 --condition-buckets 10,3,10 --cfg-scale 1.5 --silence-prefill 3 --max 16 --greedy --dac "$DAC" 2>&1)" \
  || { echo "$out"; fail "--tts --speaker-embedding (cfg) exited nonzero"; }
is_riff "$WORK/s.wav" || fail "--tts --speaker-embedding did not write a RIFF WAV"
pass "--tts --speaker-embedding --cfg-scale -> WAV"

# steering: raw controls move the embedding, hidden controls + a span add to AttnRes bank sources
out="$("$CLI" "$MODEL" --tts "ci steer" "$WORK/steer.wav" --speaker-embedding tests/fixtures/speaker-embedding.npy \
  --steer tests/fixtures/dummy-steer.gguf --control who.gender=1.5 --control style.tempo=-1 \
  --control-span 0:2:mood.valence=2 --cfg-scale 1.5 --max 16 --greedy --dac "$DAC" 2>&1)" \
  || { echo "$out"; fail "--tts --steer --control exited nonzero"; }
is_riff "$WORK/steer.wav" || fail "--tts --steer did not write a RIFF WAV"
pass "--tts --steer --control --control-span -> WAV"
if out="$("$CLI" "$MODEL" --tts "ci steer" "$WORK/steer-bad.wav" --steer tests/fixtures/dummy-steer.gguf \
    --control who.gender=1 --max 4 --dac "$DAC" 2>&1)"; then fail "raw control without a speaker embedding must fail"; fi
echo "$out" | grep -q "need speaker_mode=embedding" || { echo "$out"; fail "raw control without embedding gave the wrong error"; }
pass "--control (raw) without --speaker-embedding -> error"

# same, with --dac dropped: dac.gguf sitting next to the backbone is auto-detected
# (README CLI section's models/ example)
mkdir -p "$WORK/models"
cp "$MODEL" "$WORK/models/zonos3-q6_k.gguf"
cp "$DAC"   "$WORK/models/dac.gguf"
out=$("$CLI" "$WORK/models/zonos3-q6_k.gguf" --tts "ci tts autodiscover" "$WORK/auto.wav" --max 16 --greedy 2>&1) \
  || { echo "$out"; fail "--tts with sibling dac.gguf exited nonzero"; }
is_riff "$WORK/auto.wav" || fail "--tts with sibling dac.gguf did not write a RIFF WAV"
pass "--tts (sibling dac.gguf auto-detected) -> WAV"

# standalone codes -> wav (dac-cli), if present
DACCLI="$(find_bin dac-cli "")"
if [ -x "$DACCLI" ]; then
  "$DACCLI" "$DAC" "$WORK/codes.npy" "$WORK/d.wav" >/dev/null 2>&1 || fail "dac-cli exited nonzero"
  is_riff "$WORK/d.wav" || fail "dac-cli did not write a RIFF WAV"
  pass "dac-cli codes -> WAV"
  if [ -s tests/fixtures/dummy-dac-codes.npy ]; then
    "$DACCLI" --seam "$DAC" tests/fixtures/dummy-dac-codes.npy --block 3 --cpu 2>&1 | grep -q PASS \
      || fail "dac-cli --seam did not PASS"
    pass "dac-cli --seam (streamed == buffered)"
  fi
fi

# quantized backbones preserve both the bulk and expert-only load paths
QCLI="$(find_bin quantize-cli "")"
if [ -x "$QCLI" ]; then
  "$QCLI" "$MODEL" "$WORK/q8.gguf" q8_0 >"$WORK/q8.log" 2>&1 \
    || { cat "$WORK/q8.log"; fail "quantize-cli q8_0"; }
  "$QCLI" "$MODEL" "$WORK/q4exp.gguf" q4_0 --experts-only --spine-f16 >"$WORK/q4.log" 2>&1 \
    || { cat "$WORK/q4.log"; fail "quantize-cli q4_0 expert-only"; }
  for q in q8 q4exp; do
    "$CLI" "$WORK/$q.gguf" --tts "ci quantized" "$WORK/$q.npy" --max 8 --greedy >"$WORK/$q.tts.log" 2>&1 \
      || { cat "$WORK/$q.tts.log"; fail "$q backbone loading"; }
    [ -s "$WORK/$q.npy" ] || fail "$q generation wrote no codes"
  done
  pass "quantize-cli q8_0 + q4_0 --experts-only --spine-f16 -> loadable backbones"
  if [ -x "$DACCLI" ]; then
    "$QCLI" "$DAC" "$WORK/dac-q8.gguf" q8_0 >"$WORK/dacq.log" 2>&1 \
      || { cat "$WORK/dacq.log"; fail "quantize-cli codec q8_0"; }
    "$DACCLI" "$WORK/dac-q8.gguf" "$WORK/codes.npy" "$WORK/dq.wav" >/dev/null 2>&1 || fail "quantized codec loading"
    is_riff "$WORK/dq.wav" || fail "quantized codec did not write a RIFF WAV"
    pass "quantize-cli codec q8_0 -> loadable codec"
  fi
fi

# ----------------------------------------------------------------- server
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) echo "== server: skipped on Windows =="; exit 0 ;;
esac
if ! command -v curl >/dev/null 2>&1; then echo "== server: skipped (no curl) =="; exit 0; fi
[ -x "$SRV" ] || { echo "== server: skipped (zonos3-server not found) =="; exit 0; }

echo "== server ($SRV) =="
PORT="${ZONOS3_TEST_PORT:-19187}"
BASE="http://127.0.0.1:$PORT"
"$SRV" "$MODEL" --dac "$DAC" --steer tests/fixtures/dummy-steer.gguf --host 127.0.0.1 --port "$PORT" --max 16 --batch 2 \
  --studio-projects-dir "$WORK/projects" >"$WORK/srv.log" 2>&1 &
SRV_PID=$!

up=0
for _ in $(seq 1 100); do
  if curl -fsS "$BASE/v2/health" >/dev/null 2>&1; then up=1; break; fi
  kill -0 "$SRV_PID" 2>/dev/null || { echo "--- server log ---"; cat "$WORK/srv.log"; fail "server exited before coming up"; }
  sleep 0.2
done
[ "$up" = 1 ] || { echo "--- server log ---"; cat "$WORK/srv.log"; fail "server did not answer /v2/health in time"; }
pass "/v2/health"

caps="$(curl -fsS "$BASE/v2/capabilities")"
echo "$caps" | grep -q '"n_codebooks"' || fail "/v2/capabilities missing n_codebooks"
echo "$caps" | grep -q '"speaker_modes"' || fail "/v2/capabilities missing speaker_modes"
pass "/v2/capabilities"

# buffered WAV (README curl example)
curl -fsS "$BASE/v2/generate" \
  -d '{"text":"buffered wav","stream":false,"format":"wav","seed":1,"max_tokens":16}' \
  -o "$WORK/srv.wav" || fail "/v2/generate (buffered wav) request failed"
is_riff "$WORK/srv.wav" || fail "/v2/generate (buffered) did not return a RIFF WAV"
pass "/v2/generate -> buffered WAV"

# streaming float32 PCM (README curl example), the seed echoed in X-Seed
curl -fsS -N -D "$WORK/pcm.h" "$BASE/v2/generate" \
  -d '{"text":"streaming pcm","stream":true,"seed":7,"max_tokens":16}' \
  -o "$WORK/srv.pcm" || fail "/v2/generate (streaming pcm) request failed"
[ -s "$WORK/srv.pcm" ] || fail "/v2/generate (streaming) returned no PCM"
grep -qi '^x-seed: 7' "$WORK/pcm.h" || fail "/v2/generate did not echo X-Seed"
pass "/v2/generate -> streaming PCM ($(wc -c < "$WORK/srv.pcm") bytes)"

# whole-utterance streaming decode
curl -fsS -N "$BASE/v2/generate" \
  -d '{"text":"whole decode","stream":true,"stream_decode":"whole","seed":1,"max_tokens":16}' \
  -o "$WORK/whole.pcm" || fail "/v2/generate (stream_decode whole) request failed"
[ -s "$WORK/whole.pcm" ] || fail "/v2/generate (stream_decode whole) returned no PCM"
pass "/v2/generate stream_decode=whole -> PCM"

# OpenAI-compatible route
curl -fsS "$BASE/v1/audio/speech" \
  -d '{"model":"zonos3","input":"openai route","response_format":"wav","seed":1,"max_tokens":16}' \
  -o "$WORK/oai.wav" || fail "/v1/audio/speech request failed"
is_riff "$WORK/oai.wav" || fail "/v1/audio/speech did not return a RIFF WAV"
pass "/v1/audio/speech -> WAV"

# request schema: unknown and removed fields are a 422 detail list
body="$(curl -sS -w '\n%{http_code}' "$BASE/v2/generate" -d '{"text":"x","speed":1.0}')"
[ "${body##*$'\n'}" = 422 ] && echo "$body" | grep -q '"loc"' || fail "an unknown field must be a 422 with a detail list"
pass "unknown field -> 422"

# speaker embedding + speaker CFG (twin slot) + condition values: tests/fixtures/speaker-embedding.npy
EMB_B64="$(base64 < tests/fixtures/speaker-embedding.npy | tr -d '\n')"
curl -fsS "$BASE/v2/generate" \
  -d "{\"text\":\"ci speaker\",\"stream\":false,\"format\":\"wav\",\"seed\":1,\"max_tokens\":16,\"speaker_mode\":\"embedding\",\"speaker_embedding_base64\":\"$EMB_B64\",\"speaker_cfg_scale\":1.5,\"condition_values\":{\"loudness\":-23},\"silence_prefill_frames\":3}" \
  -o "$WORK/spk.wav" || fail "/v2/generate (speaker embedding + CFG) request failed"
is_riff "$WORK/spk.wav" || fail "/v2/generate (speaker embedding) did not return a RIFF WAV"
[ "$(wc -c < "$WORK/spk.wav")" -gt 44 ] || fail "/v2/generate (speaker embedding) returned empty audio"
pass "/v2/generate speaker_embedding_base64 + speaker_cfg_scale -> WAV ($(wc -c < "$WORK/spk.wav") bytes)"

# CFG without a speaker prompt is a client error
code="$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/v2/generate" \
  -d '{"text":"no speaker","stream":false,"format":"wav","max_tokens":16,"speaker_cfg_scale":1.5}')"
[ "$code" = 400 ] || fail "speaker CFG without a speaker prompt returned HTTP $code (expected 400)"
pass "speaker CFG without a speaker prompt -> 400"

# steering: controls document, request controls/spans, errors, speaker embedding, studio
curl -fsS "$BASE/v2/controls" | grep -q '"style.tempo"' || fail "/v2/controls missing style.tempo"
pass "/v2/controls"
curl -fsS -D "$WORK/steer.h" "$BASE/v2/generate" \
  -d "{\"text\":\"ci steer\",\"stream\":false,\"format\":\"wav\",\"seed\":1,\"max_tokens\":16,\"speaker_mode\":\"embedding\",\"speaker_embedding_base64\":\"$EMB_B64\",\"speaker_cfg_scale\":1.5,\"controls\":{\"who.gender\":1,\"style.tempo\":-1},\"control_spans\":[{\"start\":0,\"end\":2,\"controls\":{\"mood.valence\":2}}]}" \
  -o "$WORK/steer-srv.wav" || fail "/v2/generate (controls + spans) request failed"
is_riff "$WORK/steer-srv.wav" || fail "/v2/generate (controls) did not return a RIFF WAV"
grep -qi '^x-control-span-follow: rate' "$WORK/steer.h" || fail "control_spans must answer X-Control-Span-Follow: rate"
pass "/v2/generate controls + control_spans -> WAV"
body="$(curl -sS -w '\n%{http_code}' "$BASE/v2/generate" -d '{"text":"x","controls":{"nope.x":1}}')"
[ "${body##*$'\n'}" = 400 ] && echo "$body" | grep -q "Unknown control" || fail "unknown control must be a 400 'Unknown control'"
pass "unknown control -> 400"
body="$(curl -sS -w '\n%{http_code}' "$BASE/v2/generate" -d '{"text":"x","controls":{"who.gender":1}}')"
[ "${body##*$'\n'}" = 400 ] && echo "$body" | grep -q "need speaker_mode=embedding" || fail "raw control without a speaker must be a 400"
pass "raw control without speaker -> 400"
SPK_ID="$(curl -fsS -H 'X-TTS-Session-ID: ci' "$BASE/v2/speakers" \
  -d "{\"speaker_embedding_base64\":\"$EMB_B64\",\"speaker_embedding_name\":\"e.npy\",\"label\":\"ci\"}" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')"
[ -n "$SPK_ID" ] || fail "POST /v2/speakers returned no id"
curl -fsS -H 'X-TTS-Session-ID: ci' "$BASE/v2/speakers/$SPK_ID/embedding" | grep -q '"embedding":\[' || fail "/v2/speakers/<id>/embedding"
pass "/v2/speakers/<id>/embedding"
[ "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/tts/generate" -d '{"text":"x"}')" = 404 ] || fail "/tts/generate must be gone (404)"
pass "/tts/* -> 404"
# server-side projects: content-addressed media, then a project referencing it
printf 'media' > "$WORK/blob"
SHA="$( (sha256sum "$WORK/blob" 2>/dev/null || shasum -a 256 "$WORK/blob") | cut -d' ' -f1)"
[ "$(curl -sS -o /dev/null -w '%{http_code}' -X PUT --data-binary @"$WORK/blob" "$BASE/v2/media/$SHA")" = 201 ] || fail "PUT /v2/media must store (201)"
curl -fsS -X PUT "$BASE/v2/projects/ci" -H 'Content-Type: application/json' \
  -d "{\"name\":\"ci\",\"data\":{\"media\":{\"$SHA\":{\"type\":\"x/y\",\"size\":5}}},\"base_revision\":null}" | grep -q '"revision":1' \
  || fail "PUT /v2/projects/ci did not save revision 1"
pass "/v2/media + /v2/projects"
# the studio (studio/dist of the checkout) at /: index.html, then an immutable asset; pages are no-store;
# /studio/ is gone
curl -fsS "$BASE/" -o "$WORK/index.html" || fail "/ not served"
grep -q '<div id="app">' "$WORK/index.html" || fail "/ is not the studio"
curl -fsS -D "$WORK/index.h" -o /dev/null "$BASE/" || fail "/ not served"
grep -qi '^cache-control: no-store, max-age=0' "$WORK/index.h" || fail "/ is not no-store"
if [ -f "$ROOT/../../studio/dist/lite.html" ]; then
  curl -fsS -D "$WORK/lite.h" -o /dev/null "$BASE/lite.html" || fail "/lite.html not served"
  grep -qi '^cache-control: no-store, max-age=0' "$WORK/lite.h" || fail "/lite.html is not no-store"
fi
ASSET="$(sed -n 's/.*src="\(\/assets\/[^"]*\.js\)".*/\1/p' "$WORK/index.html" | head -1)"
[ -n "$ASSET" ] || fail "the studio index references no /assets/*.js"
curl -fsS -D "$WORK/asset.h" -o /dev/null "$BASE$ASSET" || fail "$ASSET not served"
grep -qi '^cache-control: public, max-age=31536000, immutable' "$WORK/asset.h" || fail "$ASSET is not immutable"
grep -qi '^content-type: text/javascript' "$WORK/asset.h" || fail "$ASSET content type"
[ "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/studio/")" = 404 ] || fail "/studio/ must be a 404"
pass "/ serves the studio"

echo "All README command tests passed."
