// CUDA version of the tanh soft-clip fuzz (the code to be ported).
// Usage: fuzz <in.wav> <out.wav> [drive=8] [gain=0.8]
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "wav.hpp"

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                   __FILE__, __LINE__);                                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

__global__ void fuzz_kernel(const float* __restrict__ in, float* __restrict__ out, int n,
                            float drive, float norm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = norm * tanhf(drive * in[i]);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s in.wav out.wav [drive] [gain]\n", argv[0]);
    return 2;
  }
  const float drive = argc > 3 ? std::strtof(argv[3], nullptr) : 8.0f;
  const float gain = argc > 4 ? std::strtof(argv[4], nullptr) : 0.8f;
  WavData wav;
  if (!wav_read(argv[1], wav)) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  const int n = static_cast<int>(wav.samples.size());
  const float norm = gain / std::tanh(drive);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, wav.samples.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const int block = 256;
  const int grid = (n + block - 1) / block;
  CUDA_CHECK(cudaEventRecord(start));
  fuzz_kernel<<<grid, block>>>(d_in, d_out, n, drive, norm);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::printf("kernel_ms=%f\n", ms);

  std::vector<float> out(n);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  if (!wav_write_f32(argv[2], out.data(), out.size(), wav.channels, wav.sample_rate)) return 1;
  return 0;
}
