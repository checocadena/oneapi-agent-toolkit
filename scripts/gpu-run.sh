#!/usr/bin/env bash
# Unattended NVIDIA GPU run for a Runpod pod (or any Linux box with an NVIDIA GPU + CUDA toolkit).
#
# Serves its progress at http://<host>:8000/ (via Runpod: https://<pod-id>-8000.proxy.runpod.net/)
# so a remote agent can follow it without SSH:
#   run.log      full log            status.txt   current phase
#   results/     per-step ToolResult JSON         SUMMARY.md / DONE   when finished
#
# Phases:
#   A. CUDA: build each example with nvcc, verify against its C++ reference, benchmark.
#   B. Get a SYCL compiler that targets NVIDIA (intel/llvm release, else oneAPI + Codeplay plugin).
#   C. SYCL on the GPU: build the idiomatic ports for nvptx64, verify vs C++ reference AND vs the
#      CUDA binary, benchmark on the cuda:0 device.
# Every phase records failures instead of stopping, and nothing is ever written as a result
# unless a tool produced it.
set -u
OUT="${GPU_RUN_OUT:-/workspace/gpu-run}"
REPO_DIR="${GPU_RUN_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$OUT/results"
cd "$OUT"

# Serve the output directory first so progress is visible from the very beginning.
if ! pgrep -f "http.server 8000" >/dev/null 2>&1; then
  nohup python3 -m http.server 8000 --directory "$OUT" >/dev/null 2>&1 &
fi
exec >>"$OUT/run.log" 2>&1
phase() { echo "$*" >"$OUT/status.txt"; echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }

phase "0 environment"
rm -f "$OUT/DONE"
nvidia-smi || echo "WARN: nvidia-smi failed"
NVCC="$(command -v nvcc || ls /usr/local/cuda/bin/nvcc 2>/dev/null || true)"
echo "nvcc=$NVCC"; [ -n "$NVCC" ] && "$NVCC" --version
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git g++ curl bzip2 xz-utils gnupg python3-venv >/dev/null \
  || echo "WARN: apt-get install failed"

phase "0 python env"
python3 -m venv /opt/gpu-venv && . /opt/gpu-venv/bin/activate
pip install -q -e "$REPO_DIR" || { echo "FATAL: pip install failed"; phase "FAILED pip"; touch "$OUT/DONE"; exit 1; }
python "$REPO_DIR/examples/gen_inputs.py"

phase "B sycl compiler"
SYCL_CXX=""
# B1: open-source DPC++ release from intel/llvm (built with the CUDA backend).
if [ ! -x /opt/sycl/bin/clang++ ]; then
  URL="$(curl -fsSL 'https://api.github.com/repos/intel/llvm/releases?per_page=30' | python3 -c '
import json,sys
for r in json.load(sys.stdin):
    for a in r.get("assets", []):
        n = a["name"]
        if n.endswith(".tar.gz") and "linux" in n and "windows" not in n:
            print(a["browser_download_url"]); sys.exit()
' 2>/dev/null)"
  echo "intel/llvm asset: ${URL:-<none found>}"
  if [ -n "$URL" ]; then
    mkdir -p /opt/sycl && curl -fsSL "$URL" | tar xz -C /opt/sycl || echo "WARN: intel/llvm download failed"
  fi
fi
if [ -x /opt/sycl/bin/clang++ ] && ls /opt/sycl/lib | grep -qi 'cuda'; then
  SYCL_CXX=/opt/sycl/bin/clang++
  export PATH="/opt/sycl/bin:$PATH" LD_LIBRARY_PATH="/opt/sycl/lib:${LD_LIBRARY_PATH:-}"
  echo "using intel/llvm DPC++: $SYCL_CXX"
else
  echo "intel/llvm build missing or has no CUDA adapter; trying oneAPI + Codeplay plugin"
  # B2: Intel oneAPI DPC++ + Codeplay "oneAPI for NVIDIA GPUs" plugin.
  if [ ! -x /opt/intel/oneapi/compiler/latest/bin/icpx ]; then
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg --yes
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
      >/etc/apt/sources.list.d/oneAPI.list
    apt-get update -qq && apt-get install -y -qq intel-oneapi-compiler-dpcpp-cpp >/dev/null \
      || echo "WARN: oneAPI apt install failed"
  fi
  if [ -x /opt/intel/oneapi/compiler/latest/bin/icpx ]; then
    FULLVER="$(/opt/intel/oneapi/compiler/latest/bin/icpx --version | head -1 | grep -oE '20[0-9]{2}\.[0-9]+\.[0-9]+' | head -1)"
    echo "oneAPI version: $FULLVER"
    for CU in 12.0 12.2 12.6 12.8 13.0; do
      curl -fsSL -o /tmp/codeplay.sh \
        "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=nvidia&version=${FULLVER}&filters[]=${CU}&filters[]=linux" \
        && head -c 2 /tmp/codeplay.sh | grep -q '#!' && { echo "codeplay plugin for CUDA $CU"; bash /tmp/codeplay.sh -y; break; }
    done
    # shellcheck disable=SC1091
    . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    SYCL_CXX=/opt/intel/oneapi/compiler/latest/bin/icpx
  fi
fi
echo "SYCL_CXX=${SYCL_CXX:-<none>}"
command -v sycl-ls >/dev/null && sycl-ls

phase "A+C examples"
SYCL_CXX="$SYCL_CXX" NVCC="$NVCC" python "$REPO_DIR/scripts/gpu_run.py" "$OUT/results" || echo "WARN: gpu_run.py exited $?"

phase "DONE"
date -u >"$OUT/DONE"
# Keep the pod (and the HTTP server) alive so results can be fetched; the operator deletes the pod.
sleep infinity
