#!/usr/bin/env bash
# Reproduces the toolchain install that worked in Milestone 0 (see PROGRESS.md) on
# Ubuntu 24.04 x86_64 without root-level apt access to Intel's repository:
#   * DPC++ compiler (icpx) + OpenCL CPU runtime from conda-forge, via micromamba
#   * SYCLomatic (c2s) from its GitHub release tarball
#   * CUDA 12.3 headers (needed by c2s to parse .cu files) from conda-forge
# Usage: scripts/install-toolchain.sh [prefix-root]   (default /opt)
set -euo pipefail

ROOT="${1:-/opt}"
SYCLOMATIC_TAG="${SYCLOMATIC_TAG:-20260701}"
MAMBA_VERSION="${MAMBA_VERSION:-2.9.0-0}"
DL="$(mktemp -d)"
trap 'rm -rf "$DL"' EXIT

echo "== micromamba $MAMBA_VERSION"
curl -fsSL -o "$DL/mm.tar.bz2" "https://conda.anaconda.org/conda-forge/linux-64/micromamba-${MAMBA_VERSION}.tar.bz2"
mkdir -p "$DL/mm" && tar xjf "$DL/mm.tar.bz2" -C "$DL/mm"
MM="$DL/mm/bin/micromamba"
export MAMBA_ROOT_PREFIX="$ROOT/mamba"

echo "== DPC++ (icpx) + OpenCL CPU runtime -> $ROOT/dpcpp"
"$MM" create -y -p "$ROOT/dpcpp" -c conda-forge dpcpp_linux-64 intel-opencl-rt

echo "== CUDA 12.3 headers (for c2s parsing only; no NVIDIA driver/GPU needed) -> $ROOT/cuda-headers"
"$MM" create -y -p "$ROOT/cuda-headers" -c conda-forge "cuda-version=12.3" \
  cuda-cudart-dev cuda-crt-dev_linux-64 libcurand-dev cuda-nvcc-dev_linux-64

echo "== SYCLomatic $SYCLOMATIC_TAG -> $ROOT/syclomatic"
curl -fsSL -o "$DL/syclomatic.tgz" \
  "https://github.com/oneapi-src/SYCLomatic/releases/download/${SYCLOMATIC_TAG}/linux_release.tgz"
mkdir -p "$ROOT/syclomatic" && tar xzf "$DL/syclomatic.tgz" -C "$ROOT/syclomatic"

cat <<EOF

Done. Activate with:
  DPCPP_PREFIX=$ROOT/dpcpp SYCLOMATIC_PREFIX=$ROOT/syclomatic CUDA_HEADERS_PREFIX=$ROOT/cuda-headers \\
    source scripts/oneapi-env.sh
  oneapi-agent doctor
EOF
