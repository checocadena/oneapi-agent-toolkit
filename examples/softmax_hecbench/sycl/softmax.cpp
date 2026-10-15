// Hand-written idiomatic SYCL 2020 port of softmax.cu (the target end state; no dpct helpers).
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/softmax-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
//
// Port notes: the CUDA "optimized" kernel gives each slice one 32-thread warp tile, eight
// tiles per 256-thread block. Here each slice gets one sub-group at the device's native width
// (32 on NVIDIA, 64 on AMD, 16 or 32 on Intel) with sycl::reduce_over_group over it, several
// slices per 256-wide work-group; groups stride over the slices, so no sub-group size is
// hard-coded. The CUDA default stream is in-order, and so is the queue: on an MI350X that cut
// the per-launch cost from 103 to 24 us (an out-of-order queue tracks dependencies on every
// submission). Kernel 0 (one work-item per slice) is ported 1:1.
// Usage: softmax <out.bin> [numSlice=8192] [sliceSize=256] [kernel=1] [repeat=1]
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "softmax_input.hpp"

constexpr size_t kBlockSize = 256;
constexpr size_t kSliceGroup = 32;  // kernel 1 grid hint: slices per group at sub-group width 32

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::printf("Usage: %s <out.bin> [number of slices] [slice size] [kernel 0|1] [repeat]\n",
                argv[0]);
    return 1;
  }
  const int numSlice = argc > 2 ? std::atoi(argv[2]) : 8192;
  const int sliceSize = argc > 3 ? std::atoi(argv[3]) : 256;
  const int kernel = argc > 4 ? std::atoi(argv[4]) : 1;
  const int repeat = argc > 5 ? std::atoi(argv[5]) : 1;
  const size_t numElem = size_t(numSlice) * sliceSize;

  std::vector<float> input(numElem), output(numElem);
  fill_input(input.data(), static_cast<int>(numElem));

  try {
    sycl::queue q{sycl::property::queue::in_order()};
    float* src = sycl::malloc_device<float>(numElem, q);
    float* dest = sycl::malloc_device<float>(numElem, q);
    q.memcpy(src, input.data(), sizeof(float) * numElem).wait();

    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      if (kernel == 1) {
        // One sub-group per slice (the CUDA tile of 32, at the device's native width), several
        // slices per 256-wide work-group. Groups stride over the slices, so the grid below is
        // only a size hint and any sub-group size (16, 32 or 64) covers every slice.
        const size_t groups = (size_t(numSlice) + kBlockSize / kSliceGroup - 1) /
                              (kBlockSize / kSliceGroup);
        q.parallel_for(sycl::nd_range<1>{groups * kBlockSize, kBlockSize},
                       [=](sycl::nd_item<1> it) {
          const auto sg = it.get_sub_group();
          const size_t per_group = sg.get_group_linear_range();
          const size_t width = sg.get_local_linear_range();
          const size_t lid = sg.get_local_linear_id();
          for (size_t i = it.get_group(0) * per_group + sg.get_group_linear_id();
               i < size_t(numSlice); i += it.get_group_range(0) * per_group) {
            const float* s = src + i * sliceSize;
            float* d = dest + i * sliceSize;
            float max_ = s[0];
            for (size_t j = lid; j < size_t(sliceSize); j += width) max_ = sycl::fmax(max_, s[j]);
            max_ = sycl::reduce_over_group(sg, max_, sycl::maximum<float>{});
            float sum = 0.0f;
            for (size_t j = lid; j < size_t(sliceSize); j += width) sum += sycl::exp(s[j] - max_);
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});
            for (size_t j = lid; j < size_t(sliceSize); j += width)
              d[j] = sycl::exp(s[j] - max_) / sum;
          }
        });
      } else {
        const size_t global = (size_t(numSlice) + kBlockSize - 1) / kBlockSize * kBlockSize;
        q.parallel_for(sycl::nd_range<1>{global, kBlockSize}, [=](sycl::nd_item<1> it) {
          const size_t i = it.get_global_id(0);
          if (i >= size_t(numSlice)) return;
          const float* s = src + i * sliceSize;
          float* d = dest + i * sliceSize;
          float max_ = s[0];
          for (int j = 0; j < sliceSize; j++) max_ = sycl::fmax(max_, s[j]);
          float sum = 0.0f;
          for (int j = 0; j < sliceSize; j++) sum += sycl::exp(s[j] - max_);
          for (int j = 0; j < sliceSize; j++) d[j] = sycl::exp(s[j] - max_) / sum;
        });
      }
    }
    q.wait_and_throw();
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("kernel_ms=%f\n", (ns * 1e-6) / repeat);

    q.memcpy(output.data(), dest, sizeof(float) * numElem).wait();
    sycl::free(src, q);
    sycl::free(dest, q);
  } catch (const sycl::exception& ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    return 1;
  }
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(output.data(), sizeof(float), numElem, f) != numElem) return 1;
  std::fclose(f);
  return 0;
}
