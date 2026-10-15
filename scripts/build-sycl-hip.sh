#!/usr/bin/env bash
# Build intel/llvm (DPC++) with the HIP (AMD) backend and pack it as a relocatable tarball for
# `oneapi-agent remote-test --vendor amd` (its --sycl-url, or the default in remote.py).
#
# Why: intel/llvm's own prebuilts (nightlies, v7.x releases) are configured with --cuda only, so
# they ship no libur_adapter_hip and no amdgcn libspirv, and sycl-ls finds no AMD GPU.
#
# Runs as root on Ubuntu 22.04 x86_64 (the oldest glibc we target, so the result also runs on
# 24.04), e.g. in CI (.github/workflows/sycl-hip-toolchain.yml) or locally:
#   docker run --rm -v "$PWD:/src:ro" -v /big/disk:/work ubuntu:22.04 \
#     bash /src/scripts/build-sycl-hip.sh /work
# Output: <work>/out/sycl-hip-linux-x86_64.tar.gz and .sha256. No GPU needed: the smoke test
# compiles and links a SYCL program for gfx942 (MI300X) and gfx950 (MI350X/MI355X).
#
# Inputs (environment):
#   INTEL_LLVM_REF  intel/llvm commit to build (default: the nightly-2026-10-14 tag's commit)
#   ROCM_VERSION    ROCm apt release to build against (default 7.0: the first with gfx950).
#                   The HIP adapter links libamdhip64.so.7, so the GPU host needs ROCm 7.x.
#   JOBS            parallel compile jobs (default: nproc)
set -euo pipefail

WORK="${1:-/work}"
INTEL_LLVM_REF="${INTEL_LLVM_REF:-dfcf0080a27749a5c8c73b9d83658593726b01ea}"
ROCM_VERSION="${ROCM_VERSION:-7.0}"
JOBS="${JOBS:-$(nproc)}"
OUT="$WORK/out"
mkdir -p "$WORK" "$OUT"
export DEBIAN_FRONTEND=noninteractive

echo "== dependencies (ROCm $ROCM_VERSION)"
apt-get update -qq
apt-get install -y -qq ca-certificates curl gnupg git python3 python3-pip build-essential \
  ninja-build zstd file binutils autoconf automake libtool pkg-config >/dev/null
python3 -m pip install -q 'cmake>=3.27,<4'
mkdir -p --mode=0755 /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg
. /etc/os-release
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION $VERSION_CODENAME main" \
  > /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600
apt-get update -qq
# hip-dev: HIP headers + libamdhip64 for the UR HIP adapter; rocm-device-libs: for the smoke test.
apt-get install -y -qq hip-dev rocm-device-libs >/dev/null
ls -l /opt/rocm/lib/libamdhip64.so*

echo "== intel/llvm $INTEL_LLVM_REF"
SRC="$WORK/llvm"
if [ ! -d "$SRC/.git" ]; then
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/intel/llvm.git
fi
git -C "$SRC" fetch -q --depth 1 origin "$INTEL_LLVM_REF"
git -C "$SRC" checkout -q FETCH_HEAD

echo "== configure"
BUILD="$WORK/build"
# --hip: HIP adapter + amdgcn libspirv + AMDGPU backend. --disable-jit: the runtime kernel
# compiler is not used by gpu_test and doubles the LLVM link work. (LLVM_PARALLEL_LINK_JOBS is
# not set: it makes ninja fail with "duplicate pool 'link_job_pool'" in this tree.)
# The two runtime overrides work around an intel/llvm bug (as of nightly-2026-10-14, which CI
# never hits because no intel/llvm CI job builds --hip): configure.py names the AMD libclc
# runtime "amdgcn-amd-amdhsa-llvm", but sycl/CMakeLists.txt requires "amdgcn-amd-amdhsa" in the
# libclc targets, and the driver looks for libspirv under lib/amdgcn-amd-amdhsa/ too
# (clang/lib/Driver/ToolChains/SYCL.cpp, findLibspirvPath). Later -D options win.
rm -f "$BUILD/CMakeCache.txt"
python3 "$SRC/buildbot/configure.py" -s "$SRC" -o "$BUILD" --hip --no-assertions --disable-jit \
  --cmake-opt=-DUR_HIP_ROCM_DIR=/opt/rocm \
  "--cmake-opt=-DLLVM_RUNTIME_TARGETS=default;amdgcn-amd-amdhsa;spirv64-unknown-unknown" \
  --cmake-opt=-DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=libclc

echo "== build (jobs=$JOBS)"
cmake --build "$BUILD" --target deploy-sycl-toolchain -- -j "$JOBS"
INSTALL="$BUILD/install"
# ROCm's apt packages do not register /opt/rocm/lib with ld.so; the GPU host gets it from the
# agent's LD_LIBRARY_PATH, and the checks below name it the same way.
export LD_LIBRARY_PATH="/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "== pack (before the checks, so a failed check never throws the build away)"
cat > "$INSTALL/BUILD-INFO.txt" <<EOF
intel/llvm commit: $INTEL_LLVM_REF
configure: buildbot/configure.py --hip --no-assertions --disable-jit, AMD libclc runtime target amdgcn-amd-amdhsa
ROCm: $ROCM_VERSION ($(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)); the HIP adapter needs libamdhip64.so.7 (ROCm 7.x) on the GPU host
built on: $PRETTY_NAME, $(date -u +%Y-%m-%dT%H:%M:%SZ)
AMD target flags: -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfxXXX
checked before publishing: HIP adapter resolves libamdhip64.so.7; amdgcn libspirv installed; a SYCL program compiles and links for gfx942 and gfx950
EOF
TAR="$OUT/sycl-hip-linux-x86_64.tar.gz"
tar -czf "$TAR" -C "$INSTALL" .
(cd "$OUT" && sha256sum "$(basename "$TAR")" > "$(basename "$TAR").sha256")
cp "$INSTALL/BUILD-INFO.txt" "$OUT/"

echo "== checks"
ls "$INSTALL/lib"/libur_adapter_* "$INSTALL/bin/sycl-ls" "$INSTALL/bin/clang++"
test -e "$INSTALL/lib/libur_adapter_hip.so" || { echo "FAIL: no libur_adapter_hip.so"; exit 1; }
find "$INSTALL/lib" -path '*amdgcn*' -name 'libspirv*' | tee "$OUT/libspirv-amdgcn.txt"
test -s "$OUT/libspirv-amdgcn.txt" || { echo "FAIL: no amdgcn libspirv"; exit 1; }
ldd "$INSTALL/lib/libur_adapter_hip.so" | tee "$OUT/ldd-adapter-hip.txt"
grep -q 'libamdhip64.so.7 => /' "$OUT/ldd-adapter-hip.txt" || { echo "FAIL: adapter does not resolve libamdhip64.so.7"; exit 1; }

echo "== smoke test: compile + link for gfx942 and gfx950"
T="$WORK/smoke"; mkdir -p "$T"
cat > "$T/smoke.cpp" <<'EOF'
#include <sycl/sycl.hpp>
int main() {
  sycl::queue q;
  float *x = sycl::malloc_shared<float>(16, q);
  q.parallel_for(sycl::range<1>(16), [=](sycl::id<1> i) { x[i] = 2.0f * i[0]; }).wait();
  float s = x[15];
  sycl::free(x, q);
  return s == 30.0f ? 0 : 1;
}
EOF
for arch in gfx942 gfx950; do
  "$INSTALL/bin/clang++" -fsycl -fsycl-targets=amdgcn-amd-amdhsa \
    -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=$arch -O2 \
    "$T/smoke.cpp" -o "$T/smoke-$arch"
  echo "built smoke-$arch: $(stat -c %s "$T/smoke-$arch") bytes"
done
LD_LIBRARY_PATH="$INSTALL/lib:$LD_LIBRARY_PATH" "$INSTALL/bin/sycl-ls" || true  # no GPU here
echo PASS > "$OUT/CHECKS-PASSED"
ls -l "$OUT"
cat "$OUT"/*.sha256
