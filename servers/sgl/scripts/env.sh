#!/bin/bash
# Runtime environment of the ZONOS3 Mini LM (gated DeltaNet + Full AttnRes + latent MoE): CUDA toolchain for the
# JIT kernels, the TileLang GDN backend, and the exact-mode kernels the release LM was qualified with.
# Source from the repo root before the server (servers/sgl/scripts/serve.sh does), the offline TTSLLM API, or the tests.
zonos3_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export FLA_TILELANG=1
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$zonos3_repo_root/.cache/tilelang}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" QUACK_COMPILE_WORKERS="${QUACK_COMPILE_WORKERS:-1}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export ZONOS3_USE_TRAINING_RMSNORM=1 ZONOS3_USE_TRAINING_ROUTER_WEIGHTS=1
export ZONOS3_USE_TRAINING_SONIC_EXPERTS=1 ZONOS3_USE_TRAINING_ATTNRES=1 ZONOS3_USE_TRAINING_ROPE=1
