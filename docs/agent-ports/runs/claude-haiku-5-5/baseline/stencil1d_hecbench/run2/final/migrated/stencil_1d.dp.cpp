// CUDA 1D stencil (the code to be ported): HeCBench stencil1d-cuda, a shared-memory tile with
// a halo of RADIUS on each side and one barrier.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/stencil1d-cuda/stencil_1d.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: kernel unchanged; the result is written to a file (as float64) instead of being
// checked in-process; argument order changed.
// Usage: stencil_1d <out.bin> [length=1048576, a multiple of 256] [repeat=1]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int RADIUS = 7;
constexpr int BLOCK_SIZE = 256;

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

  sycl::queue q{sycl::property::queue::in_order()};
  int *d_a = sycl::malloc_device<int>(length + RADIUS, q);
  int *d_b = sycl::malloc_device<int>(length, q);
  q.memcpy(d_a, a.data(), (length + RADIUS) * sizeof(int)).wait();

  const sycl::range<1> global(length), local(BLOCK_SIZE);

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int, 1> temp(sycl::range<1>(BLOCK_SIZE + 2 * RADIUS), cgh);
      cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
        const int in_idx = static_cast<int>(item.get_global_id(0));
        const int tid = static_cast<int>(item.get_local_id(0));
        const int lindex = tid + RADIUS;
        const int *in = d_a;
        int *out = d_b;

        // Read input elements into shared memory
        temp[lindex] = in[in_idx];

        // At both end of a block, the sliding window moves beyond the block boundary.
        if (tid < RADIUS) {
          temp[lindex - RADIUS] = (in_idx < RADIUS) ? 0 : in[in_idx - RADIUS];
          temp[lindex + BLOCK_SIZE] = in[in_idx + BLOCK_SIZE];
        }

        // Ensure all work items have finished populating the tile
        sycl::group_barrier(item.get_group());

        // Apply the 1D stencil
        int result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; offset++)
          result += temp[lindex + offset];

        // Store the result
        out[in_idx] = result;
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(b.data(), d_b, length * sizeof(int)).wait();
  std::vector<double> out(b.begin(), b.end());
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(out.data(), sizeof(double), length, f) != size_t(length)) return 1;
  fclose(f);
  sycl::free(d_a, q);
  sycl::free(d_b, q);
  return 0;
}
