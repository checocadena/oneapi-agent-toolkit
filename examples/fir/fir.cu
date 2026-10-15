// CUDA version of the 64-tap FIR (the code to be ported).
// Coefficients live in __constant__ memory; each thread computes one output sample of
// one channel; the input tile + history is staged through shared memory.
// Usage: fir <in.wav> <out.wav> [cutoff_hz=2000]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "fir_coeffs.hpp"
#include "wav.hpp"

#define BLOCK 256

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                   __FILE__, __LINE__);                                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

__constant__ float c_taps[kTaps];

// x and y are one de-interleaved channel of length n.
__global__ void fir_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
  __shared__ float tile[BLOCK + kTaps - 1];
  const int base = blockIdx.x * blockDim.x;
  const int t = threadIdx.x;
  // Load history (kTaps-1 samples before the block) and the block itself.
  for (int i = t; i < BLOCK + kTaps - 1; i += blockDim.x) {
    const int src = base - (kTaps - 1) + i;
    tile[i] = (src >= 0 && src < n) ? x[src] : 0.0f;
  }
  __syncthreads();
  const int idx = base + t;
  if (idx >= n) return;
  float acc = 0.0f;
  for (int k = 0; k < kTaps; ++k) acc += c_taps[k] * tile[t + (kTaps - 1) - k];
  y[idx] = acc;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s in.wav out.wav [cutoff_hz]\n", argv[0]);
    return 2;
  }
  const double cutoff = argc > 3 ? std::strtod(argv[3], nullptr) : 2000.0;
  WavData wav;
  if (!wav_read(argv[1], wav)) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  float h[kTaps];
  make_lowpass(h, cutoff, wav.sample_rate);
  CUDA_CHECK(cudaMemcpyToSymbol(c_taps, h, sizeof(h)));

  const int ch = wav.channels;
  const int n = static_cast<int>(wav.frames());
  std::vector<float> chan(n), result(n), out(wav.samples.size());
  float *d_x = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));

  for (int c = 0; c < ch; ++c) {
    for (int i = 0; i < n; ++i) chan[i] = wav.samples[i * ch + c];
    CUDA_CHECK(cudaMemcpy(d_x, chan.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    fir_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) out[i * ch + c] = result[i];
  }
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate)) return 1;
  return 0;
}
