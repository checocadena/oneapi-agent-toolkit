// SYCL 2020 port of the CUDA softmax (HeCBench softmax-cuda). Two kernels:
//   kernel 0: one work-item per slice (softmax_per_item)
//   kernel 1: one work-group per slice, group-level reductions (softmax_per_group);
//             independent of the device sub-group size.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "softmax_input.hpp"

constexpr size_t BLOCK_SIZE = 256;

static void softmax_per_item(sycl::nd_item<1> it, int numSlice, int sliceSize,
                             const float* src, float* dest) {
  size_t i = it.get_global_id(0);
  if (i >= static_cast<size_t>(numSlice)) return;
  const float* x = src + i * sliceSize;
  float* y = dest + i * sliceSize;

  float max_ = x[0];
  for (int j = 0; j < sliceSize; j++) max_ = sycl::max(max_, x[j]);

  float sum = 0.0f;
  for (int j = 0; j < sliceSize; j++) sum += sycl::exp(x[j] - max_);

  for (int j = 0; j < sliceSize; j++) y[j] = sycl::exp(x[j] - max_) / sum;
}

static void softmax_per_group(sycl::nd_item<1> it, int numSlice, int sliceSize,
                              const float* src, float* dest) {
  size_t i = it.get_group(0);
  if (i >= static_cast<size_t>(numSlice)) return;
  const float* x = src + i * sliceSize;
  float* y = dest + i * sliceSize;

  auto grp = it.get_group();
  size_t lid = it.get_local_id(0);
  size_t lrange = it.get_local_range(0);

  float max_ = -std::numeric_limits<float>::infinity();
  for (size_t j = lid; j < static_cast<size_t>(sliceSize); j += lrange)
    max_ = sycl::max(max_, x[j]);
  max_ = sycl::reduce_over_group(grp, max_, sycl::maximum<float>());

  float sum = 0.0f;
  for (size_t j = lid; j < static_cast<size_t>(sliceSize); j += lrange)
    sum += sycl::exp(x[j] - max_);
  sum = sycl::reduce_over_group(grp, sum, sycl::plus<float>());

  for (size_t j = lid; j < static_cast<size_t>(sliceSize); j += lrange)
    y[j] = sycl::exp(x[j] - max_) / sum;
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
  size_t numElem = static_cast<size_t>(numSlice) * sliceSize;

  std::vector<float> input(numElem), output_gpu(numElem);
  fill_input(input.data(), static_cast<int>(numElem));

  sycl::queue q{sycl::property::queue::in_order{}};
  float* d_input = sycl::malloc_device<float>(numElem, q);
  float* d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  q.wait();
  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    // one work-group of BLOCK_SIZE work-items per slice
    sycl::nd_range<1> range(sycl::range<1>(size_t(numSlice) * BLOCK_SIZE),
                            sycl::range<1>(BLOCK_SIZE));
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) {
        softmax_per_group(it, numSlice, sliceSize, d_input, d_output);
      });
    }
  } else {
    // one work-item per slice
    size_t global = (size_t(numSlice) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    sycl::nd_range<1> range{sycl::range<1>{global}, sycl::range<1>{BLOCK_SIZE}};
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) {
        softmax_per_item(it, numSlice, sliceSize, d_input, d_output);
      });
    }
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(output_gpu.data(), d_output, sizeof(float) * numElem).wait();
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(output_gpu.data(), sizeof(float), numElem, f) != numElem) return 1;
  fclose(f);
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  return 0;
}
