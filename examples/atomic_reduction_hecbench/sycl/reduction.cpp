// Hand-written idiomatic SYCL 2020 port of reduction.cu (the target end state; no dpct helpers).
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/atomicReduction-cuda,
// commit f0c817aa). Copyright (c) 2015-2016 Advanced Micro Devices, Inc. MIT License; see
// LICENSE.AMD in this directory.
//
// Port notes: the five CUDA kernels differ only in how many consecutive elements each
// iteration loads (1, 2, 4, 8, 16), so they become one template on that count. atomicAdd
// becomes a relaxed, device-scope sycl::atomic_ref on global memory, and cudaMemset becomes
// queue::memset; the in-order queue keeps each reset before its kernel, as the CUDA stream does.
// Usage: reduction <out.bin> [arrayLength=16777216, a multiple of 16] [repeat=10]
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../reduction_input.hpp"

template <int V>
void atomic_reduction(sycl::queue& q, const int* in, int* out, int arrayLength, int grid,
                      int block) {
  q.parallel_for(sycl::nd_range<1>{size_t(grid) * block, size_t(block)},
                 [=](sycl::nd_item<1> it) {
    int sum = 0;
    const int idx = it.get_global_id(0);
    const int stride = it.get_global_range(0) * V;
    for (int i = idx * V; i < arrayLength; i += stride) {
#pragma unroll
      for (int k = 0; k < V; k++) sum += in[i + k];
    }
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*out)
        .fetch_add(sum);
  });
}

int main(int argc, char* argv[]) {
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
  std::vector<double> sums;

  try {
    sycl::queue q{sycl::property::queue::in_order()};
    int* in = sycl::malloc_device<int>(arrayLength, q);
    int* out = sycl::malloc_device<int>(1, q);
    q.memcpy(in, array.data(), sizeof(int) * arrayLength).wait();

    double total_ns = 0;
    int launches = 0;
    auto run = [&](auto kernel, int grid, int block) {
      auto t1 = std::chrono::steady_clock::now();
      for (int i = 0; i < N; i++) {
        q.memset(out, 0, sizeof(int));
        kernel(q, in, out, arrayLength, grid, block);
      }
      q.wait_and_throw();
      auto t2 = std::chrono::steady_clock::now();
      total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
      launches += N;
      int sum = 0;
      q.memcpy(&sum, out, sizeof(int)).wait();
      sums.push_back(sum);
    };

    for (int block : block_sizes) {
      const int blocks = std::min((arrayLength + block - 1) / block, 2048);
      run(atomic_reduction<1>, blocks, block);
      run(atomic_reduction<2>, blocks / 2, block);
      run(atomic_reduction<4>, blocks / 4, block);
      run(atomic_reduction<8>, blocks / 8, block);
      run(atomic_reduction<16>, blocks / 16, block);
    }
    std::printf("kernel_ms=%f\n", total_ns * 1e-6 / launches);
    sycl::free(in, q);
    sycl::free(out, q);
  } catch (const sycl::exception& ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    return 1;
  }
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(sums.data(), sizeof(double), sums.size(), f) != sums.size()) return 1;
  std::fclose(f);
  return 0;
}
