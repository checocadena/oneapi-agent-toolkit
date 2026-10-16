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

// One work-item per slice.
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

// One sub-group per slice; works for any sub-group size the device chooses.
// Grid-stride over sub-groups so coverage does not depend on the sub-group size.
void softMax2(sycl::nd_item<1> item, const int numSlice, const int sliceSize,
              const float* src, float* dest)
{
  sycl::sub_group sg = item.get_sub_group();
  const size_t sgsPerGroup = sg.get_group_range().size();
  const size_t totalSgs = item.get_group_range(0) * sgsPerGroup;
  const size_t globalSg = item.get_group(0) * sgsPerGroup + sg.get_group_linear_id();
  const size_t lid = sg.get_local_linear_id();
  const size_t lrange = sg.get_local_linear_range();

  for (size_t i = globalSg; i < static_cast<size_t>(numSlice); i += totalSgs) {
    const float* row = src + i * sliceSize;
    float* out = dest + i * sliceSize;

    float max_ = row[0];
    for (size_t j = lid; j < static_cast<size_t>(sliceSize); j += lrange) {
      max_ = sycl::max(max_, row[j]);
    }
    max_ = sycl::reduce_over_group(sg, max_, sycl::maximum<float>{});

    float sum = 0;
    for (size_t j = lid; j < static_cast<size_t>(sliceSize); j += lrange) {
      sum += sycl::exp(row[j] - max_);
    }
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});

    for (size_t j = lid; j < static_cast<size_t>(sliceSize); j += lrange) {
      out[j] = sycl::exp(row[j] - max_) / sum;
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

  std::vector<float> input(numElem), output_gpu(numElem);
  fill_input(input.data(), numElem);

  sycl::queue q{sycl::property::queue::in_order{}};
  float* d_input = sycl::malloc_device<float>(numElem, q);
  float* d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  q.wait_and_throw();
  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    // Enough work-groups so that, for a 32-wide sub-group, every slice gets a sub-group.
    // Other sub-group sizes are still covered by the grid-stride loop in softMax2.
    const size_t sgsPerBlock = BLOCK_SIZE / 32;
    const size_t groups = (numSlice + sgsPerBlock - 1) / sgsPerBlock;
    sycl::nd_range<1> range(sycl::range<1>(groups * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE));
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(range, [=](sycl::nd_item<1> item) {
        softMax2(item, numSlice, sliceSize, d_input, d_output);
      });
    }
  } else {
    const size_t groups = (numSlice + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sycl::nd_range<1> range(sycl::range<1>(groups * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE));
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(range, [=](sycl::nd_item<1> item) {
        softMax(item, numSlice, sliceSize, d_input, d_output);
      });
    }
  }
  q.wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(output_gpu.data(), d_output, sizeof(float) * numElem).wait();
  sycl::free(d_input, q);
  sycl::free(d_output, q);

  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(output_gpu.data(), sizeof(float), numElem, f) != size_t(numElem)) return 1;
  fclose(f);
  return 0;
}
