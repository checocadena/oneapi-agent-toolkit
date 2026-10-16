// CUDA atomic reduction (the code to be ported): HeCBench atomicReduction-cuda, five
// grid-stride kernels (1, 2, 4, 8 or 16 loads per iteration) that each finish with one
// atomicAdd per thread, run for four block sizes.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/atomicReduction-cuda,
// commit f0c817aa). Copyright (c) 2015-2016 Advanced Micro Devices, Inc. MIT License; see
// LICENSE.AMD in this directory.
// Usage: reduction <out.bin> [arrayLength=16777216, a multiple of 16] [repeat=10]
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kernels.h"
#include "reduction_input.hpp"

template <int V>
static void launch_reduction(sycl::queue& q, const int* in, int* out, int n, int blocks,
                             int block_size) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(blocks) * block_size),
                                   sycl::range<1>(static_cast<size_t>(block_size))),
                 atomic_reduction_kernel<V>{in, out, n});
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("Usage: %s <out.bin> [array length, a multiple of 16] [repeat]\n", argv[0]);
    return 1;
  }
  const int arrayLength = argc > 2 ? std::atoi(argv[2]) : 16777216;
  const int N = argc > 3 ? std::atoi(argv[3]) : 10;
  if (arrayLength % 16 != 0) return 1;
  const int block_sizes[] = {128, 256, 512, 1024};

  std::vector<int> array(arrayLength);
  fill_input(array.data(), arrayLength);

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};

  int* in = sycl::malloc_device<int>(arrayLength, q);
  int* out = sycl::malloc_device<int>(1, q);
  q.memcpy(in, array.data(), sizeof(int) * arrayLength).wait();

  std::vector<double> sums;
  int sum = 0;
  double total_ns = 0;
  int launches = 0;

  // Runs N launches of one kernel variant, then reads back its single sum.
  auto benchmark = [&](auto launch, int blocks, int block_size) {
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
      q.memset(out, 0, sizeof(int));
      launch(blocks, block_size);
    }
    q.wait();
    auto t2 = std::chrono::steady_clock::now();
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    launches += N;
    q.memcpy(&sum, out, sizeof(int)).wait();
    sums.push_back(sum);
  };

  for (int block_size : block_sizes) {
    int blocks = std::min((arrayLength + block_size - 1) / block_size, 2048);
    benchmark([&](int b, int bs) { launch_reduction<1>(q, in, out, arrayLength, b, bs); },
              blocks, block_size);
    benchmark([&](int b, int bs) { launch_reduction<2>(q, in, out, arrayLength, b, bs); },
              blocks / 2, block_size);
    benchmark([&](int b, int bs) { launch_reduction<4>(q, in, out, arrayLength, b, bs); },
              blocks / 4, block_size);
    benchmark([&](int b, int bs) { launch_reduction<8>(q, in, out, arrayLength, b, bs); },
              blocks / 8, block_size);
    benchmark([&](int b, int bs) { launch_reduction<16>(q, in, out, arrayLength, b, bs); },
              blocks / 16, block_size);
  }
  std::printf("kernel_ms=%f\n", total_ns * 1e-6 / launches);

  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(sums.data(), sizeof(double), sums.size(), f) != sums.size()) return 1;
  std::fclose(f);
  sycl::free(in, q);
  sycl::free(out, q);
  return 0;
}
