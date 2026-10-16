// CUDA softmax (the code to be ported): the two kernels of HeCBench softmax-cuda.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/softmax-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: kernels unchanged; deterministic LCG input; the result is written to a file
// instead of being checked in-process; argument order changed.
// Usage: softmax <out.bin> [numSlice=8192] [sliceSize=256] [kernel=1] [repeat=1]
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "softmax_input.hpp"

#define BLOCK_SIZE 256

// One work-item per slice (CUDA softMax equivalent).
void softMax(sycl::nd_item<1> item, const int numSlice, const int sliceSize,
             const float* src, float* dest)
{
  int i = static_cast<int>(item.get_global_id(0));
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = 0; j < sliceSize; j++) {
    max_ = sycl::max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) {
    sum += sycl::exp(src[i * sliceSize + j] - max_);
  }
  for (int j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] = sycl::exp(src[i * sliceSize + j] - max_) / sum;
  }
}

// One sub-group per slice (CUDA softMax2 equivalent). The sub-group width is not
// assumed: each sub-group strides over slices, and the grid stride is taken from
// the number of sub-groups actually present in the work-group.
void softMax2(sycl::nd_item<1> item, const int numSlice, const int sliceSize,
              const float* src, float* dest)
{
  sycl::sub_group sg = item.get_sub_group();
  const size_t sgs_per_group = sg.get_group_range()[0];
  const size_t num_groups = item.get_group_range(0);
  const size_t sg_width = sg.get_local_range()[0];
  const size_t lane = sg.get_local_id()[0];
  const size_t sg_global = item.get_group(0) * sgs_per_group + sg.get_group_id()[0];
  const size_t sg_stride = num_groups * sgs_per_group;

  for (size_t i = sg_global; i < static_cast<size_t>(numSlice); i += sg_stride) {
    float max_ = src[i * sliceSize];
    for (size_t j = lane; j < static_cast<size_t>(sliceSize); j += sg_width) {
      max_ = sycl::max(max_, src[i * sliceSize + j]);
    }
    max_ = sycl::reduce_over_group(sg, max_, sycl::maximum<float>());
    float sum = 0;
    for (size_t j = lane; j < static_cast<size_t>(sliceSize); j += sg_width) {
      sum += sycl::exp(src[i * sliceSize + j] - max_);
    }
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    for (size_t j = lane; j < static_cast<size_t>(sliceSize); j += sg_width) {
      dest[i * sliceSize + j] = sycl::exp(src[i * sliceSize + j] - max_) / sum;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <out.bin> [number of slices] [slice size] [kernel 0|1] [repeat]\n", argv[0]);
    return 1;
  }
  int numSlice = argc > 2 ? atoi(argv[2]) : 8192;
  int sliceSize = argc > 3 ? atoi(argv[3]) : 256;
  int kernel = argc > 4 ? atoi(argv[4]) : 1;
  int repeat = argc > 5 ? atoi(argv[5]) : 1;
  int numElem = numSlice * sliceSize;

  sycl::queue q{sycl::property::queue::in_order()};

  std::vector<float> input(numElem), output_gpu(numElem);
  fill_input(input.data(), numElem);

  float* d_input = sycl::malloc_device<float>(numElem, q);
  float* d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  q.wait_and_throw();
  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    // Smallest sub-group width the device supports gives the maximum number of
    // sub-groups per work-group, so this group count covers all slices for any width.
    const auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    const size_t min_sg = *std::min_element(sg_sizes.begin(), sg_sizes.end());
    const size_t max_sgs_per_group = BLOCK_SIZE / min_sg;
    const size_t groups = (numSlice + max_sgs_per_group - 1) / max_sgs_per_group;
    sycl::nd_range<1> range(sycl::range<1>(groups * BLOCK_SIZE),
                            sycl::range<1>(BLOCK_SIZE));
    for (int n = 0; n < repeat; n++) {
      q.submit([&](sycl::handler& h) {
        h.parallel_for(range, [=](sycl::nd_item<1> item) {
          softMax2(item, numSlice, sliceSize, d_input, d_output);
        });
      });
    }
  } else {
    const size_t groups = (numSlice + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sycl::nd_range<1> range(sycl::range<1>(groups * BLOCK_SIZE),
                            sycl::range<1>(BLOCK_SIZE));
    for (int n = 0; n < repeat; n++) {
      q.submit([&](sycl::handler& h) {
        h.parallel_for(range, [=](sycl::nd_item<1> item) {
          softMax(item, numSlice, sliceSize, d_input, d_output);
        });
      });
    }
  }
  q.wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(output_gpu.data(), d_output, sizeof(float) * numElem).wait();
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(output_gpu.data(), sizeof(float), numElem, f) != size_t(numElem)) return 1;
  fclose(f);
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  return 0;
}
