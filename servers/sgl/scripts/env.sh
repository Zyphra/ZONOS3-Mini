#!/bin/bash
# Runtime environment of the ZONOS3 Mini LM (gated DeltaNet + Full AttnRes + latent MoE): CUDA toolchain for the
# JIT kernels, the TileLang GDN backend, and the exact-mode kernels the release LM was qualified with.
# Source from the repo root before the server (servers/sgl/scripts/serve.sh does), the offline TTSLLM API, or the tests.
zonos3_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
# Two of the kernels below are Hopper-only: SonicMoE's quack grouped GEMM (the exact-mode experts; "Gemm Sm80 is not
# implemented yet") and FLA's TileLang chunk kernels (FLA itself enables them on Hopper only). Before sm_90 (A100) the
# experts run the Triton FusedMoE and the GDN chunks FLA's Triton kernels; a value set beforehand wins (forcing the
# A100 kernels on a Hopper GPU: FLA_TILELANG=0 ZONOS3_USE_TRAINING_SONIC_EXPERTS=0). The node's GPUs are assumed alike.
zonos3_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)
zonos3_hopper=1
[[ -n "$zonos3_cc" && "${zonos3_cc%%.*}" -lt 9 ]] && zonos3_hopper=0
export FLA_TILELANG="${FLA_TILELANG:-$zonos3_hopper}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$zonos3_repo_root/.cache/tilelang}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" QUACK_COMPILE_WORKERS="${QUACK_COMPILE_WORKERS:-1}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export ZONOS3_USE_TRAINING_RMSNORM=1 ZONOS3_USE_TRAINING_ROUTER_WEIGHTS=1
export ZONOS3_USE_TRAINING_SONIC_EXPERTS="${ZONOS3_USE_TRAINING_SONIC_EXPERTS:-$zonos3_hopper}"
export ZONOS3_USE_TRAINING_ATTNRES=1 ZONOS3_USE_TRAINING_ROPE=1
