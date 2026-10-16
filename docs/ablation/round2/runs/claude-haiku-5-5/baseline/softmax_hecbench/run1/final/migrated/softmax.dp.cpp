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

#define BLOCK_SIZE 256

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

  std::vector<float> input(numElem), output_gpu(numElem);
  fill_input(input.data(), numElem);

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};
  float* d_input = sycl::malloc_device<float>(numElem, q);
  float* d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  q.wait_and_throw();
  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    // One sub-group per slice. The sub-group size is not fixed in the source: each
    // sub-group strides over slices, so any sub-group size the device chooses is correct.
    // Launch with at least 8 sub-groups' worth of slices per work-group (a conservative
    // lower bound on sub-group count); extra sub-groups simply find no work.
    size_t numGroups = (numSlice + (BLOCK_SIZE / 8) - 1) / (BLOCK_SIZE / 8);
    if (numGroups == 0) numGroups = 1;
    sycl::nd_range<1> range(sycl::range<1>(numGroups * BLOCK_SIZE),
                            sycl::range<1>(BLOCK_SIZE));
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) {
        sycl::sub_group sg = it.get_sub_group();
        const size_t sgsPerGroup = sg.get_group_range()[0];
        const size_t totalSg = it.get_group_range(0) * sgsPerGroup;
        const size_t lane = sg.get_local_id()[0];
        const size_t laneCount = sg.get_local_range()[0];
        for (size_t i = it.get_group(0) * sgsPerGroup + sg.get_group_id()[0];
             i < static_cast<size_t>(numSlice); i += totalSg) {
          const float* src_row = d_input + i * sliceSize;
          float* dst_row = d_output + i * sliceSize;

          float max_ = src_row[0];
          for (size_t j = lane; j < static_cast<size_t>(sliceSize); j += laneCount) {
            max_ = sycl::max(max_, src_row[j]);
          }
          max_ = sycl::reduce_over_group(sg, max_, sycl::maximum<float>());

          float sum = 0.0f;
          for (size_t j = lane; j < static_cast<size_t>(sliceSize); j += laneCount) {
            sum += sycl::exp(src_row[j] - max_);
          }
          sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());

          for (size_t j = lane; j < static_cast<size_t>(sliceSize); j += laneCount) {
            dst_row[j] = sycl::exp(src_row[j] - max_) / sum;
          }
        }
      });
    }
  } else {
    // One work-item per slice.
    size_t numGroups = (numSlice + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sycl::nd_range<1> range(sycl::range<1>(numGroups * BLOCK_SIZE),
                            sycl::range<1>(BLOCK_SIZE));
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) {
        size_t i = it.get_global_id(0);
        if (i >= static_cast<size_t>(numSlice)) return;
        const float* src_row = d_input + i * sliceSize;
        float* dst_row = d_output + i * sliceSize;

        float max_ = src_row[0];
        for (int j = 0; j < sliceSize; j++) {
          max_ = sycl::max(max_, src_row[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < sliceSize; j++) {
          sum += sycl::exp(src_row[j] - max_);
        }
        for (int j = 0; j < sliceSize; j++) {
          dst_row[j] = sycl::exp(src_row[j] - max_) / sum;
        }
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
