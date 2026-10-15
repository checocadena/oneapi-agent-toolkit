// Hand-written idiomatic SYCL 2020 port of stencil_1d.cu (the target end state; no dpct helpers).
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/stencil1d-cuda/stencil_1d.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
//
// Port notes: __shared__ becomes a sycl::local_accessor sized for the tile plus both halos,
// and __syncthreads() becomes sycl::group_barrier.
// Usage: stencil_1d <out.bin> [length=1048576, a multiple of 256] [repeat=1]
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int kRadius = 7;
constexpr int kBlockSize = 256;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::printf("Usage: %s <out.bin> [length, a multiple of %d] [repeat]\n", argv[0], kBlockSize);
    return 1;
  }
  const int length = argc > 2 ? std::atoi(argv[2]) : 1048576;
  const int repeat = argc > 3 ? std::atoi(argv[3]) : 1;
  if (length % kBlockSize != 0) return 1;

  std::vector<int> a(length + kRadius), b(length);
  for (int i = 0; i < length + kRadius; i++) a[i] = i;

  try {
    sycl::queue q;
    int* in = sycl::malloc_device<int>(length + kRadius, q);
    int* out = sycl::malloc_device<int>(length, q);
    q.memcpy(in, a.data(), sizeof(int) * (length + kRadius)).wait();

    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> temp(sycl::range<1>(kBlockSize + 2 * kRadius), h);
        h.parallel_for(sycl::nd_range<1>{size_t(length), size_t(kBlockSize)},
                       [=](sycl::nd_item<1> it) {
          const int lid = it.get_local_id(0);
          const int gindex = it.get_global_id(0);
          const int lindex = lid + kRadius;
          temp[lindex] = in[gindex];
          if (lid < kRadius) {
            temp[lindex - kRadius] = (gindex < kRadius) ? 0 : in[gindex - kRadius];
            temp[lindex + kBlockSize] = in[gindex + kBlockSize];
          }
          sycl::group_barrier(it.get_group());
          int result = 0;
          for (int offset = -kRadius; offset <= kRadius; offset++) result += temp[lindex + offset];
          out[gindex] = result;
        });
      });
    }
    q.wait_and_throw();
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("kernel_ms=%f\n", (ns * 1e-6) / repeat);

    q.memcpy(b.data(), out, sizeof(int) * length).wait();
    sycl::free(in, q);
    sycl::free(out, q);
  } catch (const sycl::exception& ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    return 1;
  }
  std::vector<double> res(b.begin(), b.end());
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(res.data(), sizeof(double), length, f) != size_t(length)) return 1;
  std::fclose(f);
  return 0;
}
