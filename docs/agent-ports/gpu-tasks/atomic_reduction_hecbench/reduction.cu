// CUDA atomic reduction (the code to be ported): HeCBench atomicReduction-cuda, five
// grid-stride kernels (1, 2, 4, 8 or 16 loads per iteration) that each finish with one
// atomicAdd per thread, run for four block sizes.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/atomicReduction-cuda,
// commit f0c817aa). Copyright (c) 2015-2016 Advanced Micro Devices, Inc. MIT License; see
// LICENSE.AMD in this directory.
// Adaptations: kernels.h unchanged; deterministic LCG input (instead of rand() % 2); the 20
// sums (5 kernels x 4 block sizes) are written to a file (as float64) instead of being
// checked in-process; no device-name query; argument order changed.
// Usage: reduction <out.bin> [arrayLength=16777216, a multiple of 16] [repeat=10]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include "kernels.h"
#include "reduction_input.hpp"

int main(int argc, char** argv)
{
  if (argc < 2) {
    printf("Usage: %s <out.bin> [array length, a multiple of 16] [repeat]\n", argv[0]);
    return 1;
  }
  const int arrayLength = argc > 2 ? atoi(argv[2]) : 16777216;
  const int N = argc > 3 ? atoi(argv[3]) : 10;
  if (arrayLength % 16 != 0) return 1;
  const int block_sizes[] = {128, 256, 512, 1024};

  std::vector<int> array(arrayLength);
  fill_input(array.data(), arrayLength);

  int *in, *out;
  cudaMalloc(&in, sizeof(int) * arrayLength);
  cudaMalloc(&out, sizeof(int));
  cudaMemcpy(in, array.data(), sizeof(int) * arrayLength, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  std::vector<double> sums;
  int sum;
  double total_ns = 0;
  int launches = 0;

#define benchmark(kernel_name, grid_size) \
  { \
    auto t1 = std::chrono::steady_clock::now(); \
    for (int i = 0; i < N; i++) { \
      cudaMemset(out, 0, sizeof(int)); \
      kernel_name<<< dim3(grid_size), dim3(block_size) >>>(in, out, arrayLength); \
    } \
    cudaDeviceSynchronize(); \
    auto t2 = std::chrono::steady_clock::now(); \
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(); \
    launches += N; \
    cudaMemcpy(&sum, out, sizeof(int), cudaMemcpyDeviceToHost); \
    sums.push_back(sum); \
  }

  for (size_t k = 0; k < sizeof(block_sizes) / sizeof(int); k++) {
    int block_size = block_sizes[k];
    int blocks = std::min((arrayLength + block_size - 1) / block_size, 2048);
    benchmark(atomic_reduction, blocks);
    benchmark(atomic_reduction_v2, blocks/2);
    benchmark(atomic_reduction_v4, blocks/4);
    benchmark(atomic_reduction_v8, blocks/8);
    benchmark(atomic_reduction_v16, blocks/16);
  }
  printf("kernel_ms=%f\n", total_ns * 1e-6 / launches);

  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(sums.data(), sizeof(double), sums.size(), f) != sums.size()) return 1;
  fclose(f);
  cudaFree(in);
  cudaFree(out);
  return 0;
}
