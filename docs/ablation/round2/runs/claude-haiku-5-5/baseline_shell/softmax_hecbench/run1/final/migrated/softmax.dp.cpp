// CUDA softmax (the code to be ported): the two kernels of HeCBench softmax-cuda.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/softmax-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: kernels unchanged; deterministic LCG input; the result is written to a file
// instead of being checked in-process; argument order changed.
// Usage: softmax <out.bin> [numSlice=8192] [sliceSize=256] [kernel=1] [repeat=1]
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "softmax_input.hpp"

#define BLOCK_SIZE 256


void softMax (const int numSlice, const int sliceSize,
              const float* src, float* dest)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2);
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = 0; j < sliceSize; j++) {
    max_ = sycl::max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) {
    sum += sycl::native::exp(src[i * sliceSize + j] - max_);
  }
  for (int j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] =
        sycl::native::exp(src[i * sliceSize + j] - max_) / sum;
  }
}


void softMax2 (const int numSlice, const int sliceSize,
              const float* src, float* dest)
{

  sycl::group<3> block = sycl::ext::oneapi::this_work_item::get_work_group<3>();
  sycl::sub_group warp = sycl::ext::oneapi::this_work_item::get_sub_group();
  int i =
      sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_group(2) *
          sycl::ext::oneapi::this_work_item::get_sub_group()
              .get_group_linear_range() +
      sycl::ext::oneapi::this_work_item::get_sub_group().get_group_linear_id();
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = sycl::ext::oneapi::this_work_item::get_sub_group()
                   .get_local_linear_id();
       j < sliceSize; j += sycl::ext::oneapi::this_work_item::get_sub_group()
                               .get_local_linear_range()) {
    max_ = sycl::max(max_, src[i * sliceSize + j]);
  }
  max_ = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), max_,
      sycl::maximum<float>{});
  float sum = 0;
  for (int j = sycl::ext::oneapi::this_work_item::get_sub_group()
                   .get_local_linear_id();
       j < sliceSize; j += sycl::ext::oneapi::this_work_item::get_sub_group()
                               .get_local_linear_range()) {
    sum += sycl::native::exp(src[i * sliceSize + j] - max_);
  }
  sum = sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), sum,
      sycl::plus<float>{});
  for (int j = sycl::ext::oneapi::this_work_item::get_sub_group()
                   .get_local_linear_id();
       j < sliceSize; j += sycl::ext::oneapi::this_work_item::get_sub_group()
                               .get_local_linear_range()) {
    dest[i * sliceSize + j] =
        sycl::native::exp(src[i * sliceSize + j] - max_) / sum;
  }
}

int main(int argc, char *argv[]) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
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

  float *d_input, *d_output;
  d_input = sycl::malloc_device<float>(numElem, q_ct1);
  d_output = sycl::malloc_device<float>(numElem, q_ct1);
  q_ct1.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  dev_ct1.queues_wait_and_throw();
  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    dpct::dim3 grids((numSlice + BLOCK_SIZE / 32 - 1) / (BLOCK_SIZE / 32));
    dpct::dim3 blocks(BLOCK_SIZE);
    for (int n = 0; n < repeat; n++) {
      q_ct1.parallel_for(sycl::nd_range<3>(grids * blocks, blocks),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(32)]] {
                               softMax2(numSlice, sliceSize, d_input, d_output);
                             });
    }
  } else {
    dpct::dim3 grids((numSlice + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dpct::dim3 blocks(BLOCK_SIZE);
    for (int n = 0; n < repeat; n++) {
      q_ct1.parallel_for(sycl::nd_range<3>(grids * blocks, blocks),
                         [=](sycl::nd_item<3> item_ct1) {
                           softMax(numSlice, sliceSize, d_input, d_output);
                         });
    }
  }
  dev_ct1.queues_wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q_ct1.memcpy(output_gpu.data(), d_output, sizeof(float) * numElem).wait();
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(output_gpu.data(), sizeof(float), numElem, f) != size_t(numElem)) return 1;
  fclose(f);
  dpct::dpct_free(d_input, q_ct1);
  dpct::dpct_free(d_output, q_ct1);
  return 0;
}
