#!/usr/bin/env bash
# Parity goldens for harness/parity.html, from a CPU ggml build and (for the decode argmax check) a Metal build:
#   cmake -S servers/ggml -B /tmp/ggml-build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
#   cmake --build /tmp/ggml-build -j 8 --target zonos3-cli zac-cli zac-encoder-cli spk-encoder-cli
#   cmake -S servers/ggml -B /tmp/ggml-build-metal -DCMAKE_BUILD_TYPE=Release
#   cmake --build /tmp/ggml-build-metal -j 8 --target zonos3-cli
# The CPU build accumulates F16 dot products in fp16 (NEON), which flips near-tie MoE expert picks against any GPU
# backend (ggml Metal agrees with it on 94.5% of decode argmaxes); the GPU golden validate_metal/ is the decode
# argmax reference. Run from the repository root; writes servers/web/public/goldens/.
set -euo pipefail
B=${GGML_BUILD:-/tmp/ggml-build}
BM=${GGML_METAL_BUILD:-/tmp/ggml-build-metal}
M=servers/ggml/models
E=servers/ggml/tests/fixtures/speaker-embedding.npy
O=servers/web/public/goldens
TEXT="Hello from the browser engine."
mkdir -p "$O"
"$B/zonos3-cli" "$M/zonos3-qat.gguf" --cpu --speaker-embedding "$E" --condition-buckets 3,5,10 --silence-prefill 3 \
  --build-prompt "$TEXT" "$O/ids.npy"
"$B/zonos3-cli" "$M/zonos3-qat.gguf" --cpu --speaker-embedding "$E" --condition-buckets 3,5,10 \
  --tts "$TEXT" "$O/tts.npy" --seed 1 --max 60 --dump-ids "$O/full_ids.npy" --dump-logits "$O/logits.npy" \
  --probe-heads laci_heads/zonos3_mini_mect60k_heads.json --dump-probe "$O/probe.npy"
# --validate reads the conditioning from full_ids.cond.json (written by --dump-ids)
"$B/zonos3-cli" "$M/zonos3-qat.gguf" --cpu --validate "$O/full_ids.npy" "$O/validate"
"$BM/zonos3-cli" "$M/zonos3-qat.gguf" --gpu --validate "$O/full_ids.npy" "$O/validate_metal"
"$B/zac-cli" "$M/zac.gguf" "$O/tts.npy" "$O/zac_audio.npy" --cpu
"$B/spk-encoder-cli" "$M/spk-encoder.gguf" --audio default_voices/AmericanFemale.mp3 "$O/spk.npy" --dump-dir "$O/spk" --cpu
# The exact 44.1 kHz PCM the CLI's ffmpeg decode produced (audio-decode.cpp flags), for speaker_embed_pcm.
ffmpeg -nostdin -v error -i default_voices/AmericanFemale.mp3 -ac 1 -ar 44100 -t 60 -f f32le - |
  uv run --package zonos3-tools python -c "import sys, numpy as np; np.save('$O/spk/pcm44k.npy', np.frombuffer(sys.stdin.buffer.read(), dtype='<f4'))"
cp "$E" "$O/speaker-embedding.npy"
echo "goldens: $O"
