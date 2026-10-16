// CUDA atomic reduction (the code to be ported): HeCBench atomicReduction-cuda, five
// grid-stride kernels (1, 2, 4, 8 or 16 loads per iteration) that each finish with one
// atomicAdd per thread, run for four block sizes.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/atomicReduction-cuda,
// commit f0c817aa). Copyright (c) 2015-2016 Advanced Micro Devices, Inc. MIT License; see
// LICENSE.AMD in this directory.
// Adaptations: kernels.h unchanged; deterministic LCG input (instead of rand() % 2); the 20
// sums (5 kernels x 4 block sizes) are written to a file (as float64) instead of being
// checked in-process; no device-name query; argument order changed.
// Usage: reduction <out.bin> [arrayLength=16777216, a multiple of 16] [repeat=10]
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kernels.h"
#include "reduction_input.hpp"
#include <cmath>

int main(int argc, char** argv)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (argc < 2) {
    printf("Usage: %s <out.bin> [array length, a multiple of 16] [repeat]\n", argv[0]);
    return 1;
  }
  const int arrayLength = argc > 2 ? atoi(argv[2]) : 16777216;
  const int N = argc > 3 ? atoi(argv[3]) : 10;
  if (arrayLength % 16 != 0) return 1;
  const int block_sizes[] = {128, 256, 512, 1024};

  std::vector<int> array(arrayLength);
  fill_input(array.data(), arrayLength);

  int *in, *out;
  in = sycl::malloc_device<int>(arrayLength, q_ct1);
  out = sycl::malloc_device<int>(1, q_ct1);
  q_ct1.memcpy(in, array.data(), sizeof(int) * arrayLength).wait();
  dev_ct1.queues_wait_and_throw();

  std::vector<double> sums;
  int sum;
  double total_ns = 0;
  int launches = 0;

/*
DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
work-group size if needed.
*/
/*
DPCT1064:6: Migrated cudaMemcpy call is used in a macro/template definition and
may not be valid for all macro/template uses. Adjust the code.
*/
#define benchmark(kernel_name, grid_size)                                      \
 {                                                                             \
  auto t1 = std::chrono::steady_clock::now();                                  \
  for (int i = 0; i < N; i++) {                                                \
   dpct::get_in_order_queue().memset(out, 0, sizeof(int)).wait();              \
   q_ct1.submit([&](sycl::handler &cgh) {                                      \
    auto in_ct0 = in;                                                          \
    auto out_ct1 = out;                                                        \
    auto arrayLength_ct2 = arrayLength;                                        \
                                                                               \
    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid_size) *       \
                                           sycl::range<3>(1, 1, block_size),   \
                                       sycl::range<3>(1, 1, block_size)),      \
                     [=](sycl::nd_item<3> item_ct1) {                          \
                      atomic_reduction(in_ct0, out_ct1, arrayLength_ct2);      \
                     });                                                       \
   });                                                                         \
  }                                                                            \
  dpct::get_current_device().queues_wait_and_throw();                          \
  auto t2 = std::chrono::steady_clock::now();                                  \
  total_ns +=                                                                  \
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();   \
  launches += N;                                                               \
  dpct::get_in_order_queue().memcpy(&sum, out, sizeof(int)).wait();            \
  sums.push_back(sum);                                                         \
 }

  for (size_t k = 0; k < sizeof(block_sizes) / sizeof(int); k++) {
    int block_size = block_sizes[k];
    int blocks = std::min((arrayLength + block_size - 1) / block_size, 2048);
    /*
    DPCT1038:0: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    benchmark(atomic_reduction, blocks);
    /*
    DPCT1038:2: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    benchmark(atomic_reduction_v2, blocks / 2);
    /*
    DPCT1038:3: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    benchmark(atomic_reduction_v4, blocks / 4);
    /*
    DPCT1038:4: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    benchmark(atomic_reduction_v8, blocks / 8);
    /*
    DPCT1038:5: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    benchmark(atomic_reduction_v16, blocks / 16);
  }
  printf("kernel_ms=%f\n", total_ns * 1e-6 / launches);

  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(sums.data(), sizeof(double), sums.size(), f) != sums.size()) return 1;
  fclose(f);
  dpct::dpct_free(in, q_ct1);
  dpct::dpct_free(out, q_ct1);
  return 0;
}
