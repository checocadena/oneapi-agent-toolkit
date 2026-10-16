// CUDA 1D stencil (the code to be ported): HeCBench stencil1d-cuda, a shared-memory tile with
// a halo of RADIUS on each side and one barrier.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/stencil1d-cuda/stencil_1d.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: kernel unchanged; the result is written to a file (as float64) instead of being
// checked in-process; argument order changed.
// Usage: stencil_1d <out.bin> [length=1048576, a multiple of 256] [repeat=1]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda.h>

#define RADIUS 7
#define BLOCK_SIZE 256

__global__
void stencil_1d(const int *__restrict__ in, int *__restrict__ out)
{
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + RADIUS;

  // Read input elements into shared memory
  temp[lindex] = in[gindex];

  // At both end of a block, the sliding window moves beyond the block boundary.
  if (threadIdx.x < RADIUS) {
    temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  // Synchronize (ensure all the threads will be completed before continue)
  __syncthreads();

  // Apply the 1D stencil
  int result = 0;
  for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
    result += temp[lindex + offset];

  // Store the result
  out[gindex] = result;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <out.bin> [length, a multiple of %d] [repeat]\n", argv[0], BLOCK_SIZE);
    return 1;
  }
  const int length = argc > 2 ? atoi(argv[2]) : 1048576;
  const int repeat = argc > 3 ? atoi(argv[3]) : 1;
  if (length % BLOCK_SIZE != 0) return 1;

  std::vector<int> a(length + RADIUS), b(length);
  for (int i = 0; i < length + RADIUS; i++) a[i] = i;

  int *d_a, *d_b;
  cudaMalloc((void **)&d_a, (length + RADIUS) * sizeof(int));
  cudaMalloc((void **)&d_b, length * sizeof(int));
  cudaMemcpy(d_a, a.data(), (length + RADIUS) * sizeof(int), cudaMemcpyHostToDevice);

  dim3 grids (length/BLOCK_SIZE);
  dim3 blocks (BLOCK_SIZE);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    stencil_1d <<< grids, blocks >>> (d_a, d_b);
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  cudaMemcpy(b.data(), d_b, length * sizeof(int), cudaMemcpyDeviceToHost);
  std::vector<double> out(b.begin(), b.end());
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(out.data(), sizeof(double), length, f) != size_t(length)) return 1;
  fclose(f);
  cudaFree(d_a);
  cudaFree(d_b);
  return 0;
}
