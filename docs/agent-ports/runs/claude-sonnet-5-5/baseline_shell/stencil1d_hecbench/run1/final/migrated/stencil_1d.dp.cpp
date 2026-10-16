// CUDA 1D stencil (the code to be ported): HeCBench stencil1d-cuda, a shared-memory tile with
// a halo of RADIUS on each side and one barrier.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/stencil1d-cuda/stencil_1d.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: kernel unchanged; the result is written to a file (as float64) instead of being
// checked in-process; argument order changed.
// Usage: stencil_1d <out.bin> [length=1048576, a multiple of 256] [repeat=1]
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define RADIUS 7
#define BLOCK_SIZE 256


void stencil_1d(const int *__restrict__ in, int *__restrict__ out, int *temp)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

  int gindex = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range(2);
  int lindex = item_ct1.get_local_id(2) + RADIUS;

  // Read input elements into shared memory
  temp[lindex] = in[gindex];

  // At both end of a block, the sliding window moves beyond the block boundary.
  if (item_ct1.get_local_id(2) < RADIUS) {
    temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : in[gindex - RADIUS];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }

  // Synchronize (ensure all the threads will be completed before continue)
  /*
  DPCT1113:0: Consider replacing
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "stencil_1d" is called in a
  multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // Apply the 1D stencil
  int result = 0;
  for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
    result += temp[lindex + offset];

  // Store the result
  out[gindex] = result;
}

int main(int argc, char *argv[]) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (argc < 2) {
    printf("Usage: %s <out.bin> [length, a multiple of %d] [repeat]\n", argv[0], BLOCK_SIZE);
    return 1;
  }
  const int length = argc > 2 ? atoi(argv[2]) : 1048576;
  const int repeat = argc > 3 ? atoi(argv[3]) : 1;
  if (length % BLOCK_SIZE != 0) return 1;

  std::vector<int> a(length + RADIUS), b(length);
  for (int i = 0; i < length + RADIUS; i++) a[i] = i;

  int *d_a, *d_b;
  d_a = sycl::malloc_device<int>((length + RADIUS), q_ct1);
  d_b = sycl::malloc_device<int>(length, q_ct1);
  q_ct1.memcpy(d_a, a.data(), (length + RADIUS) * sizeof(int)).wait();

  dpct::dim3 grids(length / BLOCK_SIZE);
  dpct::dim3 blocks(BLOCK_SIZE);

  dev_ct1.queues_wait_and_throw();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    q_ct1.submit([&](sycl::handler &cgh) {
      /*
      DPCT1101:1: 'BLOCK_SIZE + 2 * RADIUS' expression was replaced with a
      value. Modify the code to use the original expression, provided in
      comments, if it is correct.
      */
      sycl::local_accessor<int, 1> temp_acc_ct1(
          sycl::range<1>(270 /*BLOCK_SIZE + 2 * RADIUS*/), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(grids * blocks, blocks),
          [=](sycl::nd_item<3> item_ct1) {
            stencil_1d(d_a, d_b,
                       temp_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                           .get());
          });
    });
  dev_ct1.queues_wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q_ct1.memcpy(b.data(), d_b, length * sizeof(int)).wait();
  std::vector<double> out(b.begin(), b.end());
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(out.data(), sizeof(double), length, f) != size_t(length)) return 1;
  fclose(f);
  dpct::dpct_free(d_a, q_ct1);
  dpct::dpct_free(d_b, q_ct1);
  return 0;
}
