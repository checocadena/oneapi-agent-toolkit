// CUDA SAXPY (the code to be ported). Unified memory, error checks, event timing.
// Usage: saxpy <out.bin> [n=1048576]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void saxpy(int n, float a, const float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.bin [n]\n", argv[0]);
    return 2;
  }
  const int n = argc > 2 ? std::atoi(argv[2]) : (1 << 20);
  const float a = 2.5f;
  float *x, *y;
  cudaMallocManaged(&x, n * sizeof(float));
  cudaMallocManaged(&y, n * sizeof(float));
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i % 1000) * 0.001f;
    y[i] = static_cast<float>((i * 7) % 1000) * 0.002f;
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  saxpy<<<(n + 255) / 256, 256>>>(n, a, x, y);
  cudaEventRecord(stop);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }
  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf("kernel_ms=%f\n", ms);
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(y, sizeof(float), n, f) != static_cast<size_t>(n)) return 1;
  std::fclose(f);
  cudaFree(x);
  cudaFree(y);
  return 0;
}
