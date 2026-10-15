// CUDA block-wise exclusive scan (the code to be ported): HeCBench scan-cuda, the work-efficient
// Blelloch up-sweep / down-sweep in shared memory, with and without bank-conflict-avoiding
// padding, over a grid-stride loop of blocks.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/scan-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: both kernels unchanged; run for one type (int) and one block size (N = 512)
// instead of five sizes and four types; deterministic LCG input (instead of rand()); both
// results are written to a file (as float64) instead of being checked in-process.
// Usage: scan <out.bin> [number of elements=1048576] [repeat=1]
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda.h>

#include "scan_input.hpp"

// bank conflict aware optimization

#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)

// N is the number of elements to scan in a thread block
template<typename T, int N>
__global__ void scan_bcao (
  const int64_t nblocks,
        T *__restrict__ g_odata,
  const T *__restrict__ g_idata)
{
  __shared__ T temp[2*N];

  for (int64_t bid = blockIdx.x; bid < nblocks; bid += gridDim.x)
  {
    auto gi = g_idata + bid * N;
    auto go = g_odata + bid * N;

    int thid = threadIdx.x;
    int a = thid;
    int b = a + (N/2);
    int oa = OFFSET(a);
    int ob = OFFSET(b);

    temp[a + oa] = gi[a];
    temp[b + ob] = gi[b];

    int offset = 1;
    for (int d = N >> 1; d > 0; d >>= 1)
    {
      __syncthreads();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        ai += OFFSET(ai);
        bi += OFFSET(bi);
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    if (thid == 0) temp[N-1+OFFSET(N-1)] = 0; // clear the last elem
    for (int d = 1; d < N; d *= 2) // traverse down
    {
      offset >>= 1;
      __syncthreads();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        ai += OFFSET(ai);
        bi += OFFSET(bi);
        T t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
    __syncthreads(); // required

    go[a] = temp[a + oa];
    go[b] = temp[b + ob];
  }
}

template<typename T, int N>
__global__ void scan(
  const int64_t nblocks,
        T *__restrict__ g_odata,
  const T *__restrict__ g_idata)
{
  __shared__ T temp[N];

  for (int64_t bid = blockIdx.x; bid < nblocks; bid += gridDim.x)
  {
    auto gi = g_idata + bid * N;
    auto go = g_odata + bid * N;

    int thid = threadIdx.x;
    int offset = 1;
    temp[2*thid]   = gi[2*thid];
    temp[2*thid+1] = gi[2*thid+1];
    for (int d = N >> 1; d > 0; d >>= 1)
    {
      __syncthreads();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    if (thid == 0) temp[N-1] = 0; // clear the last elem
    for (int d = 1; d < N; d *= 2) // traverse down
    {
      offset >>= 1;
      __syncthreads();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        T t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
    go[2*thid] = temp[2*thid];
    go[2*thid+1] = temp[2*thid+1];
  }
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage: %s <out.bin> [number of elements] [repeat]\n", argv[0]);
    return 1;
  }
  typedef int T;
  const int N = 512;
  const int64_t n = argc > 2 ? atol(argv[2]) : 1048576;
  const int repeat = argc > 3 ? atoi(argv[3]) : 1;

  const int64_t num_blocks = (n + N - 1) / N;
  const int64_t nelems = num_blocks * N; // actual total number of elements
  const int64_t bytes = nelems * sizeof(T);

  std::vector<T> in(nelems), out(2 * nelems);
  fill_input(in.data(), nelems);

  T *d_in, *d_out;
  cudaMalloc((void**)&d_in, bytes);
  cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice);
  cudaMalloc((void**)&d_out, bytes);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  dim3 grids (16 * prop.multiProcessorCount);
  dim3 blocks (N/2);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    scan<T, N><<<grids, blocks>>>(num_blocks, d_out, d_in);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    scan_bcao<T, N><<<grids, blocks>>>(num_blocks, d_out, d_in);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  cudaMemcpy(out.data() + nelems, d_out, bytes, cudaMemcpyDeviceToHost);
  printf("kernel_ms=%f\n", (time * 1e-6f) / (2 * repeat));

  std::vector<double> res(out.begin(), out.end());
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(res.data(), sizeof(double), res.size(), f) != res.size()) return 1;
  fclose(f);
  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
