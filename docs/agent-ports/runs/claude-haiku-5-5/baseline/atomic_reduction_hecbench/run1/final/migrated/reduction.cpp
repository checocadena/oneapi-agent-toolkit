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
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
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

  sycl::queue q{sycl::property::queue::in_order()};
  int* in = sycl::malloc_device<int>(arrayLength, q);
  int* out = sycl::malloc_device<int>(1, q);
  q.memcpy(in, array.data(), sizeof(int) * arrayLength).wait();

  std::vector<double> sums;
  double total_ns = 0;
  int launches = 0;

  // Launches Kernel<V> over `grid` work-groups of `bs` work-items, N times, and
  // records the final sum of the last launch.
  auto benchmark = [&](auto kernel, std::size_t grid, std::size_t bs) {
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
      q.memset(out, 0, sizeof(int));
      q.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid * bs), sycl::range<1>(bs)),
                     kernel);
    }
    q.wait();
    auto t2 = std::chrono::steady_clock::now();
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    launches += N;
    int sum = 0;
    q.memcpy(&sum, out, sizeof(int)).wait();
    sums.push_back(sum);
  };

  for (int block_size : block_sizes) {
    std::size_t blocks = std::min((arrayLength + block_size - 1) / block_size, 2048);
    std::size_t bs = static_cast<std::size_t>(block_size);
    benchmark(atomic_reduction_kernel<1>{in, out, (std::size_t)arrayLength}, blocks, bs);
    benchmark(atomic_reduction_kernel<2>{in, out, (std::size_t)arrayLength}, blocks / 2, bs);
    benchmark(atomic_reduction_kernel<4>{in, out, (std::size_t)arrayLength}, blocks / 4, bs);
    benchmark(atomic_reduction_kernel<8>{in, out, (std::size_t)arrayLength}, blocks / 8, bs);
    benchmark(atomic_reduction_kernel<16>{in, out, (std::size_t)arrayLength}, blocks / 16, bs);
  }
  printf("kernel_ms=%f\n", total_ns * 1e-6 / launches);

  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(sums.data(), sizeof(double), sums.size(), f) != sums.size()) return 1;
  fclose(f);
  sycl::free(in, q);
  sycl::free(out, q);
  return 0;
}
