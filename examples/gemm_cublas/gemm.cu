// CUDA + cuBLAS SGEMM (the code to be ported): C = alpha * A * B + beta * C, column-major.
// Inputs come from a deterministic LCG; the M x N result is written as raw float32 in
// column-major order. Usage: gemm <out.bin> [M=384] [N=320] [K=256]
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),  \
                   __FILE__, __LINE__);                                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                          \
    cublasStatus_t s_ = (call);                                                 \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(s_),  \
                   __FILE__, __LINE__);                                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// Uniform in [-0.5, 0.5): 24 high bits of a 32-bit LCG.
static void fill(std::vector<float>& v, uint32_t seed) {
  uint32_t s = seed;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<float>(s >> 8) * (1.0f / 16777216.0f) - 0.5f;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.bin [M] [N] [K]\n", argv[0]);
    return 2;
  }
  const int m = argc > 2 ? std::atoi(argv[2]) : 384;
  const int n = argc > 3 ? std::atoi(argv[3]) : 320;
  const int k = argc > 4 ? std::atoi(argv[4]) : 256;
  const float alpha = 1.5f, beta = -0.5f;
  const int lda = m, ldb = k, ldc = m;  // column-major, no padding

  std::vector<float> a(static_cast<size_t>(lda) * k), b(static_cast<size_t>(ldb) * n),
      c(static_cast<size_t>(ldc) * n);
  fill(a, 1u);
  fill(b, 2u);
  fill(c, 3u);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, c.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_c, c.data(), c.size() * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, lda, d_b,
                           ldb, &beta, d_c, ldc));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::printf("kernel_ms=%f\n", ms);

  CUDA_CHECK(cudaMemcpy(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(c.data(), sizeof(float), c.size(), f) != c.size()) return 1;
  std::fclose(f);

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return 0;
}
