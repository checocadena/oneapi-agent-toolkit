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

// Launches one reduction variant `repeat` times (each launch zeroes the output
// first), records the final sum, and accumulates the timing.
template <typename Kernel>
void run_variant(sycl::queue &q, Kernel kernel, int grid_size, int block_size,
                 int *in, int *out, int arrayLength, int repeat,
                 double &total_ns, int &launches, std::vector<double> &sums) {
  auto t1 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.memset(out, 0, sizeof(int));
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size * block_size),
                                     sycl::range<1>(block_size)),
                   [=](sycl::nd_item<1> item) {
                     kernel(in, out, arrayLength, item);
                   });
  }
  q.wait();
  auto t2 = std::chrono::steady_clock::now();
  total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  launches += repeat;

  int sum = 0;
  q.memcpy(&sum, out, sizeof(int)).wait();
  sums.push_back(sum);
}

int main(int argc, char **argv) {
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
  int *in = sycl::malloc_device<int>(arrayLength, q);
  int *out = sycl::malloc_device<int>(1, q);
  q.memcpy(in, array.data(), sizeof(int) * arrayLength).wait();

  std::vector<double> sums;
  double total_ns = 0;
  int launches = 0;

  for (int block_size : block_sizes) {
    int blocks = std::min((arrayLength + block_size - 1) / block_size, 2048);
    run_variant(q, [](int *a, int *o, int n, sycl::nd_item<1> it) { atomic_reduction(a, o, n, it); },
                blocks, block_size, in, out, arrayLength, N, total_ns, launches, sums);
    run_variant(q, [](int *a, int *o, int n, sycl::nd_item<1> it) { atomic_reduction_v2(a, o, n, it); },
                blocks / 2, block_size, in, out, arrayLength, N, total_ns, launches, sums);
    run_variant(q, [](int *a, int *o, int n, sycl::nd_item<1> it) { atomic_reduction_v4(a, o, n, it); },
                blocks / 4, block_size, in, out, arrayLength, N, total_ns, launches, sums);
    run_variant(q, [](int *a, int *o, int n, sycl::nd_item<1> it) { atomic_reduction_v8(a, o, n, it); },
                blocks / 8, block_size, in, out, arrayLength, N, total_ns, launches, sums);
    run_variant(q, [](int *a, int *o, int n, sycl::nd_item<1> it) { atomic_reduction_v16(a, o, n, it); },
                blocks / 16, block_size, in, out, arrayLength, N, total_ns, launches, sums);
  }
  printf("kernel_ms=%f\n", total_ns * 1e-6 / launches);

  FILE *f = fopen(argv[1], "wb");
  if (!f || fwrite(sums.data(), sizeof(double), sums.size(), f) != sums.size()) return 1;
  fclose(f);
  sycl::free(in, q);
  sycl::free(out, q);
  return 0;
}
