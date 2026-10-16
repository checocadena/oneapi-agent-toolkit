// SYCL softmax: the two kernels of HeCBench softmax-cuda.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/softmax-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Usage: softmax <out.bin> [numSlice=8192] [sliceSize=256] [kernel=1] [repeat=1]
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "softmax_input.hpp"

constexpr int BLOCK_SIZE = 256;

int main(int argc, char *argv[]) {
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

  sycl::queue q{sycl::property::queue::in_order()};

  float *d_input = sycl::malloc_device<float>(numElem, q);
  float *d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  // The sub-group size is chosen by the runtime. Size the grid for the smallest
  // supported sub-group so that every slice gets a sub-group; surplus ones exit early.
  size_t min_sg = BLOCK_SIZE;
  for (auto s : q.get_device().get_info<sycl::info::device::sub_group_sizes>())
    min_sg = std::min<size_t>(min_sg, s);
  const size_t max_sg_per_group = BLOCK_SIZE / min_sg;

  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    const size_t groups = (numSlice + max_sg_per_group - 1) / max_sg_per_group;
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(
          sycl::nd_range<1>(groups * BLOCK_SIZE, BLOCK_SIZE),
          [=](sycl::nd_item<1> it) {
            sycl::sub_group sg = it.get_sub_group();
            const int i = int(it.get_group(0) * sg.get_group_range()[0] +
                              sg.get_group_id()[0]);
            if (i >= numSlice) return;
            const int lane = int(sg.get_local_id()[0]);
            const int step = int(sg.get_local_range()[0]);
            const float *row = d_input + size_t(i) * sliceSize;
            float *out = d_output + size_t(i) * sliceSize;
            float max_ = row[0];
            for (int j = lane; j < sliceSize; j += step)
              max_ = sycl::max(max_, row[j]);
            max_ = sycl::reduce_over_group(sg, max_, sycl::maximum<float>{});
            float sum = 0;
            for (int j = lane; j < sliceSize; j += step)
              sum += sycl::exp(row[j] - max_);
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});
            for (int j = lane; j < sliceSize; j += step)
              out[j] = sycl::exp(row[j] - max_) / sum;
          });
    }
  } else {
    const size_t groups = (numSlice + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(
          sycl::nd_range<1>(groups * BLOCK_SIZE, BLOCK_SIZE),
          [=](sycl::nd_item<1> it) {
            const int i = int(it.get_global_id(0));
            if (i >= numSlice) return;
            const float *row = d_input + size_t(i) * sliceSize;
            float *out = d_output + size_t(i) * sliceSize;
            float max_ = row[0];
            for (int j = 0; j < sliceSize; j++) max_ = sycl::max(max_, row[j]);
            float sum = 0;
            for (int j = 0; j < sliceSize; j++) sum += sycl::exp(row[j] - max_);
            for (int j = 0; j < sliceSize; j++)
              out[j] = sycl::exp(row[j] - max_) / sum;
          });
    }
  }
  q.wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(output_gpu.data(), d_output, sizeof(float) * numElem).wait();
  FILE *f = fopen(argv[1], "wb");
  if (!f || fwrite(output_gpu.data(), sizeof(float), numElem, f) != size_t(numElem)) return 1;
  fclose(f);
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  return 0;
}
