#!/usr/bin/env bash
# Unattended AMD GPU run (Runpod ROCm pod, or any Linux box with an AMD GPU + ROCm).
#
# Same contract as scripts/gpu-run.sh: progress is served at http://<host>:8000/
# (Runpod: https://<pod-id>-8000.proxy.runpod.net/) so a remote agent can follow it without SSH.
#   run.log  status.txt  results/  SUMMARY.md  DONE
#
# Phases:
#   0. ROCm sanity: rocminfo finds a gfx* agent, and a tiny HIP kernel runs.
#   A. The original CUDA sources, hipified (hipify-perl) and built with hipcc, verified against
#      the C++ references: the "native" baseline on this GPU.
#   B. An open-source DPC++ (intel/llvm) release with the HIP adapter.
#   C. The idiomatic SYCL ports built for amd_gpu_<arch>, verified against the C++ reference
#      AND the hipified CUDA binary, and benchmarked on hip:0.
# Failures are recorded, never papered over; nothing is written as a result unless a tool produced it.
set -u
OUT="${GPU_RUN_OUT:-/workspace/gpu-run}"
REPO_DIR="${GPU_RUN_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$OUT/results"
cd "$OUT"
if ! pgrep -f "http.server 8000" >/dev/null 2>&1; then
  nohup python3 -m http.server 8000 --directory "$OUT" >/dev/null 2>&1 &
fi
exec >>"$OUT/run.log" 2>&1
phase() { echo "$*" >"$OUT/status.txt"; echo; echo "=== [$(date -u +%H:%M:%S)] $*"; }
finish() { date -u >"$OUT/DONE"; sleep infinity; }

phase "0 environment"
rm -f "$OUT/DONE"
export PATH="/opt/rocm/bin:$PATH"
rocm-smi || echo "WARN: rocm-smi failed"
AMD_ARCH="${AMD_ARCH:-$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | grep -v gfx000 | head -1)}"
echo "AMD_ARCH=${AMD_ARCH:-<none>}"
HIPCC="$(command -v hipcc || true)"
HIPIFY="$(command -v hipify-perl || true)"
echo "hipcc=$HIPCC hipify=$HIPIFY"
[ -n "$HIPCC" ] && "$HIPCC" --version | head -3
if [ -z "$AMD_ARCH" ]; then
  phase "FAILED gpu-unusable: rocminfo shows no gfx agent"; finish
fi
if [ -n "$HIPCC" ]; then
  cat >/tmp/hip_smoke.cpp <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k(int* p) { p[threadIdx.x] = threadIdx.x; }
int main() {
  int* d; int h[64];
  if (hipMalloc(&d, sizeof(h)) != hipSuccess) { printf("hipMalloc failed\n"); return 1; }
  hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0, d);
  if (hipMemcpy(h, d, sizeof(h), hipMemcpyDeviceToHost) != hipSuccess || h[63] != 63) {
    printf("kernel/memcpy failed\n"); return 1; }
  printf("HIP smoke test OK\n"); return 0;
}
EOF
  if ! { "$HIPCC" --offload-arch="$AMD_ARCH" -o /tmp/hip_smoke /tmp/hip_smoke.cpp && /tmp/hip_smoke; }; then
    phase "FAILED gpu-unusable: HIP smoke test failed on this host (driver/runtime problem, not the code)"
    finish
  fi
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git g++ curl python3-venv >/dev/null \
  || echo "WARN: apt-get install failed"

phase "0 python env"
if python3 -m venv /opt/gpu-venv; then . /opt/gpu-venv/bin/activate; fi
python3 -m pip install -q -e "$REPO_DIR" || { phase "FAILED pip"; finish; }
python3 "$REPO_DIR/examples/gen_inputs.py"

phase "B sycl compiler (intel/llvm with the HIP adapter)"
SYCL_CXX=""
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
  [ -n "$URL" ] && { mkdir -p /opt/sycl && curl -fsSL "$URL" | tar xz -C /opt/sycl || echo "WARN: download failed"; }
fi
ls /opt/sycl/lib 2>/dev/null | grep -iE 'adapter|plugin' || true
if [ -x /opt/sycl/bin/clang++ ] && ls /opt/sycl/lib | grep -qi 'hip'; then
  SYCL_CXX=/opt/sycl/bin/clang++
  export PATH="/opt/sycl/bin:$PATH" LD_LIBRARY_PATH="/opt/sycl/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
else
  echo "no intel/llvm build with a HIP adapter: SYCL phase skipped"
fi
echo "SYCL_CXX=${SYCL_CXX:-<none>}"
command -v sycl-ls >/dev/null && sycl-ls

phase "A+C examples"
GPU_VENDOR=amd AMD_ARCH="$AMD_ARCH" HIPCC="$HIPCC" HIPIFY="$HIPIFY" SYCL_CXX="$SYCL_CXX" \
  python3 "$REPO_DIR/scripts/gpu_run.py" "$OUT/results" || echo "WARN: gpu_run.py exited $?"

phase "DONE"
finish
