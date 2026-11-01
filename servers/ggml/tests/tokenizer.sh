#!/usr/bin/env bash
# SuperBPE tokenizer regression: zonos3-cli --build-prompt on the dummy backbone (which embeds the
# real superbpe8192.json) must reproduce the reference text ids (BOS + id+4 + EOS) recorded by the
# evaluation runtime in tests/fixtures/tokenizer-expected.json.
#
# Usage:  bash tests/tokenizer.sh [cli-bin]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="tests/fixtures/dummy-zonos3.gguf"
CASES="tests/fixtures/tokenizer-expected.json"
CLI="${1:-$(find build* \( -name zonos3-cli -o -name zonos3-cli.exe \) -type f 2>/dev/null | head -1)}"
[ -x "$CLI" ] || { echo "FAIL: zonos3-cli not found (pass it as \$1)"; exit 1; }
PY="$(command -v python3 || command -v python)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

n="$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["cases"]))' "$CASES")"
for ((i = 0; i < n; i++)); do
  text="$("$PY" -c 'import json,sys; sys.stdout.write(json.load(open(sys.argv[1], encoding="utf-8"))["cases"][int(sys.argv[2])]["text"])' "$CASES" "$i")"
  "$CLI" "$MODEL" --build-prompt "$text" "$WORK/ids.npy" --cpu >/dev/null 2>"$WORK/err" \
    || { cat "$WORK/err"; echo "FAIL: case $i: --build-prompt failed"; exit 1; }
  "$PY" - "$CASES" "$i" "$WORK/ids.npy" <<'EOF'
import json, struct, sys
cases, i, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
case = json.load(open(cases, encoding="utf-8"))["cases"][i]
raw = open(path, "rb").read()
assert raw[:6] == b"\x93NUMPY", "not an npy file"
hlen = struct.unpack("<H", raw[8:10])[0] if raw[6] == 1 else struct.unpack("<I", raw[8:12])[0]
start = (10 if raw[6] == 1 else 12) + hlen
header = raw[start - hlen:start].decode("latin1")
assert "'<f4'" in header and "False" in header, header
vals = struct.unpack("<%df" % ((len(raw) - start) // 4), raw[start:])
W = 11
# Condition and silence-prefill rows carry the text sentinel 8196 (text_vocab); text ids are below it.
got = [int(round(v)) for v in vals[W - 1::W] if int(round(v)) != 8196]
want = case["ids"]
if got != want:
    at = next((k for k in range(min(len(got), len(want))) if got[k] != want[k]), min(len(got), len(want)))
    print(f"FAIL: case {i} {case['text']!r}: first mismatch at {at}: "
          f"got {got[at:at + 5]} want {want[at:at + 5]} (lengths {len(got)} vs {len(want)})")
    sys.exit(1)
print(f"  ok: case {i} ({len(want)} ids)")
EOF
done
echo "tokenizer: all $n cases match"
