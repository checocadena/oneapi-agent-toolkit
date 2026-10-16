// CUDA softmax (the code to be ported): the two kernels of HeCBench softmax-cuda.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/softmax-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: kernels unchanged; deterministic LCG input; the result is written to a file
// instead of being checked in-process; argument order changed.
// Usage: softmax <out.bin> [numSlice=8192] [sliceSize=256] [kernel=1] [repeat=1]
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "softmax_input.hpp"

constexpr size_t BLOCK_SIZE = 256;

// One work-item per slice.
void softmax_per_item(const int numSlice, const int sliceSize,
                      const float *src, float *dest, sycl::nd_item<1> it) {
  int i = static_cast<int>(it.get_global_id(0));
  if (i >= numSlice) return;
  const float *in = src + static_cast<size_t>(i) * sliceSize;
  float *out = dest + static_cast<size_t>(i) * sliceSize;

  float max_ = in[0];
  for (int j = 0; j < sliceSize; j++) max_ = sycl::max(max_, in[j]);
  float sum = 0.f;
  for (int j = 0; j < sliceSize; j++) sum += sycl::exp(in[j] - max_);
  for (int j = 0; j < sliceSize; j++) out[j] = sycl::exp(in[j] - max_) / sum;
}

// One sub-group per slice; the sub-group cooperates on the reductions.
// Every work-item of a sub-group computes the same slice index, so the
// collective calls are uniform within the sub-group.
void softmax_per_subgroup(const int numSlice, const int sliceSize,
                          const float *src, float *dest, sycl::nd_item<1> it) {
  sycl::sub_group sg = it.get_sub_group();
  const size_t sgSize = sg.get_max_local_range()[0];
  const int i = static_cast<int>(it.get_global_linear_id() / sgSize);
  if (i >= numSlice) return;
  const size_t lane = sg.get_local_linear_id();
  const float *in = src + static_cast<size_t>(i) * sliceSize;
  float *out = dest + static_cast<size_t>(i) * sliceSize;

  float max_ = in[0];
  for (size_t j = lane; j < static_cast<size_t>(sliceSize); j += sgSize)
    max_ = sycl::max(max_, in[j]);
  max_ = sycl::reduce_over_group(sg, max_, sycl::maximum<float>());

  float sum = 0.f;
  for (size_t j = lane; j < static_cast<size_t>(sliceSize); j += sgSize)
    sum += sycl::exp(in[j] - max_);
  sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());

  for (size_t j = lane; j < static_cast<size_t>(sliceSize); j += sgSize)
    out[j] = sycl::exp(in[j] - max_) / sum;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <out.bin> [number of slices] [slice size] [kernel 0|1] [repeat]\n", argv[0]);
    return 1;
  }
  int numSlice = argc > 2 ? atoi(argv[2]) : 8192;
  int sliceSize = argc > 3 ? atoi(argv[3]) : 256;
  int kernel = argc > 4 ? atoi(argv[4]) : 1;
  int repeat = argc > 5 ? atoi(argv[5]) : 1;
  size_t numElem = static_cast<size_t>(numSlice) * sliceSize;

  sycl::queue q{sycl::property::queue::in_order()};

  std::vector<float> input(numElem), output_gpu(numElem);
  fill_input(input.data(), numElem);

  float *d_input = sycl::malloc_device<float>(numElem, q);
  float *d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  // Sub-group size supported by the device; the reduction kernel maps one
  // sub-group to one slice, so the launch is sized with the largest one.
  const auto sgSizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  const size_t maxSG = sgSizes.empty() ? 1 : *std::max_element(sgSizes.begin(), sgSizes.end());

  auto roundUp = [](size_t n, size_t m) { return (n + m - 1) / m * m; };

  q.wait();
  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    // Each sub-group of size maxSG covers one slice; surplus lanes/sub-groups exit early.
    sycl::nd_range<1> range(sycl::range<1>(roundUp(static_cast<size_t>(numSlice) * maxSG, BLOCK_SIZE)),
                            sycl::range<1>(BLOCK_SIZE));
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) {
        softmax_per_subgroup(numSlice, sliceSize, d_input, d_output, it);
      });
    }
  } else {
    sycl::nd_range<1> range(sycl::range<1>(roundUp(static_cast<size_t>(numSlice), BLOCK_SIZE)),
                            sycl::range<1>(BLOCK_SIZE));
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) {
        softmax_per_item(numSlice, sliceSize, d_input, d_output, it);
      });
    }
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(output_gpu.data(), d_output, sizeof(float) * numElem).wait();
  FILE *f = fopen(argv[1], "wb");
  if (!f || fwrite(output_gpu.data(), sizeof(float), numElem, f) != numElem) return 1;
  fclose(f);
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  return 0;
}
