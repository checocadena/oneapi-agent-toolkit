# Source this file to point oneapi-agent at a toolchain installed the way
# scripts/install-toolchain.sh does it (conda-forge DPC++ + SYCLomatic release + CUDA headers).
# Override the prefixes if yours differ.
DPCPP_PREFIX="${DPCPP_PREFIX:-/opt/dpcpp}"
SYCLOMATIC_PREFIX="${SYCLOMATIC_PREFIX:-/opt/syclomatic}"
CUDA_HEADERS_PREFIX="${CUDA_HEADERS_PREFIX:-/opt/cuda-headers}"

export PATH="$DPCPP_PREFIX/bin:$SYCLOMATIC_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$DPCPP_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ONEAPI_AGENT_CUDA_INCLUDE="$CUDA_HEADERS_PREFIX/targets/x86_64-linux/include"
export ONEAPI_AGENT_DPCT_INCLUDE="$SYCLOMATIC_PREFIX/include"
# Optional math libraries (examples/gemm_cublas): oneMKL SYCL from conda-forge and oneMath
# built from source on top of it (WITH_ONEMATH=1 install-toolchain.sh). task.json flags use them.
export ONEMKL_PREFIX="${ONEMKL_PREFIX:-/opt/onemkl}"
export ONEMATH_PREFIX="${ONEMATH_PREFIX:-/opt/onemath}"
