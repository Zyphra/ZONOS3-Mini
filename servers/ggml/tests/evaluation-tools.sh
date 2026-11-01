#!/usr/bin/env bash
# Exercise reference distributions, latent importance, runtime masks and on-disk pruning
# through the public command-line interfaces, using real model evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ "$#" -ne 4 ]; then
  echo 'usage: bash tests/evaluation-tools.sh CLI PERPLEXITY QUANTIZER PRUNER' >&2
  exit 2
fi
CLI="$1"; PPL="$2"; QUANT="$3"; PRUNE="$4"
MODEL="tests/fixtures/dummy-zonos3.gguf"
PY="$(command -v python3 || command -v python)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
"$CLI" "$MODEL" --tts "expert calibration trace" "$WORK/codes.npy" --dump-ids "$WORK/trace.npy" --greedy --max 16 --cpu
printf '%s\n' "$WORK/trace.npy" > "$WORK/corpus.manifest"
"$PPL" "$MODEL" --kl-divergence-base "$WORK/base.bin" --manifest "$WORK/corpus.manifest" --cpu
"$PPL" "$MODEL" --kl-divergence "$WORK/base.bin" --cpu > "$WORK/self.log"
"$PY" - "$WORK/self.log" "$WORK/base.bin" <<'PY'
import math, re, struct, sys
text = open(sys.argv[1], encoding='utf-8').read()
match = re.search(r'KLD: mean=([0-9.eE+-]+)', text)
assert match, 'no numeric KLD result'
value = float(match.group(1))
assert math.isfinite(value) and value <= 1e-6, f'self KLD is {value}'
with open(sys.argv[2], 'rb') as source:
    assert source.read(8) == b'Z3KLDIV4'
    assert struct.unpack('<I', source.read(4))[0] == 4
PY
"$PPL" "$MODEL" --imatrix-out "$WORK/imatrix.bin" --prune-stats "$WORK/stats.bin" --manifest "$WORK/corpus.manifest" --imatrix-min-hits 1 --cpu
"$QUANT" "$MODEL" "$WORK/quant.gguf" q4_0 --experts-only --spine-f16 --imatrix "$WORK/imatrix.bin"
"$PPL" "$WORK/quant.gguf" --kl-divergence "$WORK/base.bin" --cpu
"$PRUNE" "$MODEL" "$WORK/pruned.gguf" --stats "$WORK/stats.bin" --keep 2
"$PPL" "$MODEL" --prune-mask "$WORK/stats.bin" --keep 2 --perplexity "$WORK/trace.npy" --cpu > "$WORK/masked.log"
"$PPL" "$WORK/pruned.gguf" --perplexity "$WORK/trace.npy" --cpu > "$WORK/pruned.log"
"$PY" - "$WORK/masked.log" "$WORK/pruned.log" <<'PY'
import math, re, sys
values = []
for path in sys.argv[1:]:
    match = re.search(r'PPL = ([0-9.eE+-]+)', open(path, encoding='utf-8').read())
    assert match, 'no numeric perplexity result'
    value = float(match.group(1))
    assert math.isfinite(value) and value > 0
    values.append(value)
assert abs(values[0] - values[1]) <= 1e-4 * max(values), f'mask/prune mismatch: {values}'
PY
# A base without this model generation's schema cannot be silently reused.
"$PY" - "$WORK/base.bin" "$WORK/incompatible.bin" <<'PY'
import sys
raw = bytearray(open(sys.argv[1], 'rb').read())
raw[7] ^= 1
open(sys.argv[2], 'wb').write(raw)
PY
if "$PPL" "$MODEL" --kl-divergence "$WORK/incompatible.bin" --cpu; then
  echo 'FAIL: incompatible KLD base accepted' >&2
  exit 1
fi
# A one-code error must fail even when float relative tolerance would accept it.
"$PY" - "$WORK" <<'PY'
import pathlib, subprocess, sys
import numpy as np
root = pathlib.Path(sys.argv[1])
golden, native = root / 'golden', root / 'native'
golden.mkdir()
native.mkdir()
expected = np.full((1, 10), 4095, dtype=np.float32)
actual = expected.copy()
actual[0, 0] = 4094
np.save(golden / 'codes.npy', expected)
np.save(native / 'codes.npy', actual)
result = subprocess.run([sys.executable, '../../tools/compare_golden.py', str(golden), str(native)],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
assert result.returncode == 1, f'one wrong codec ID accepted: {result.stdout} {result.stderr}'
PY
echo 'Evaluation tool integration passed.'
