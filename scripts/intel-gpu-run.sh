#!/usr/bin/env bash
# Intel GPU run, for a machine with an Intel GPU and oneAPI (e.g. Intel Tiber AI Cloud).
# Run it yourself after cloning the repo:
#
#   git clone https://github.com/checocadena/oneapi-agent-toolkit && cd oneapi-agent-toolkit
#   bash scripts/intel-gpu-run.sh            # ~10 minutes
#
# It builds the idiomatic SYCL ports ahead of time for the GPU (spir64), verifies each against
# its C++ reference on the level_zero GPU, benchmarks it, and writes everything to
# docs/gpu-run/intel-<gpu>/ (SUMMARY.md, run.log, results/*.json). Then either push that
# folder (git add docs/gpu-run && git commit -m "Intel GPU run" && git push) or send SUMMARY.md.
# There is no CUDA on an Intel GPU, so the C++ reference is the ground truth here; the NVIDIA and
# AMD runs already compare the SYCL ports against the original CUDA.
set -u
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

if [ -z "${ONEAPI_ROOT:-}" ]; then
  for f in /opt/intel/oneapi/setvars.sh "$HOME/intel/oneapi/setvars.sh"; do
    # shellcheck disable=SC1090
    [ -f "$f" ] && { . "$f" --force >/dev/null 2>&1; break; }
  done
fi
command -v icpx >/dev/null || { echo "icpx not found: load oneAPI first (source setvars.sh, or 'module load intel')."; exit 1; }
command -v sycl-ls >/dev/null && sycl-ls
GPU_NAME="$(sycl-ls 2>/dev/null | grep -i 'level_zero:gpu' | head -1 | sed -E 's/.*\] ?//; s/[^A-Za-z0-9]+/-/g; s/^-|-$//g' | tr 'A-Z' 'a-z')"
if [ -z "$GPU_NAME" ]; then
  echo "sycl-ls shows no level_zero:gpu device. Is this a GPU node (not a login node)?"; exit 1
fi
OUT="$REPO_DIR/docs/gpu-run/intel-${GPU_NAME:0:40}"
mkdir -p "$OUT/results"
echo "writing to $OUT"

{
  echo "=== $(date -u) environment"
  icpx --version | head -2
  sycl-ls
  python3 -m venv .gpu-venv && . .gpu-venv/bin/activate
  python3 -m pip install -q -e .
  python3 examples/gen_inputs.py
  echo "=== $(date -u) examples"
  GPU_VENDOR=intel SYCL_CXX="$(command -v icpx)" python3 scripts/gpu_run.py "$OUT/results"
  echo "=== $(date -u) done"
} 2>&1 | tee "$OUT/run.log"

rm -rf "$OUT/work"  # build outputs; results/*.json keep the evidence
sed -i "s#$REPO_DIR/##g" "$OUT/run.log" "$OUT"/SUMMARY.md "$OUT"/results/*.json 2>/dev/null || true
echo
echo "Results: $OUT/SUMMARY.md"
echo "Next: git add docs/gpu-run && git commit -m 'Intel GPU run' && git push   (or send SUMMARY.md)"
