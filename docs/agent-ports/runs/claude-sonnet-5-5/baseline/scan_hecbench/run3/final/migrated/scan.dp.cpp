// CUDA block-wise exclusive scan (the code to be ported): HeCBench scan-cuda, the work-efficient
// Blelloch up-sweep / down-sweep in shared memory, with and without bank-conflict-avoiding
// padding, over a grid-stride loop of blocks.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/scan-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: both kernels unchanged; run for one type (int) and one block size (N = 512)
// instead of five sizes and four types; deterministic LCG input (instead of rand()); both
// results are written to a file (as float64) instead of being checked in-process.
// Usage: scan <out.bin> [number of elements=1048576] [repeat=1]
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "scan_input.hpp"

// bank conflict aware optimization

#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)

// N is the number of elements to scan in a thread block
template<typename T, int N>
void scan_bcao (
  const int64_t nblocks,
        T *__restrict__ g_odata,
  const T *__restrict__ g_idata,
  T *temp)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

  for (int64_t bid = item_ct1.get_group(2); bid < nblocks;
       bid += item_ct1.get_group_range(2))
  {
    auto gi = g_idata + bid * N;
    auto go = g_odata + bid * N;

    int thid = item_ct1.get_local_id(2);
    int a = thid;
    int b = a + (N/2);
    int oa = OFFSET(a);
    int ob = OFFSET(b);

    temp[a + oa] = gi[a];
    temp[b + ob] = gi[b];

    int offset = 1;
    for (int d = N >> 1; d > 0; d >>= 1)
    {
      /*
      DPCT1118:1: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        ai += OFFSET(ai);
        bi += OFFSET(bi);
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    if (thid == 0) temp[N-1+OFFSET(N-1)] = 0; // clear the last elem
    for (int d = 1; d < N; d *= 2) // traverse down
    {
      offset >>= 1;
      /*
      DPCT1118:2: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        ai += OFFSET(ai);
        bi += OFFSET(bi);
        T t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
    /*
    DPCT1118:0: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // required

    go[a] = temp[a + oa];
    go[b] = temp[b + ob];
  }
}

template<typename T, int N>
void scan(
  const int64_t nblocks,
        T *__restrict__ g_odata,
  const T *__restrict__ g_idata,
  T *temp)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

  for (int64_t bid = item_ct1.get_group(2); bid < nblocks;
       bid += item_ct1.get_group_range(2))
  {
    auto gi = g_idata + bid * N;
    auto go = g_odata + bid * N;

    int thid = item_ct1.get_local_id(2);
    int offset = 1;
    temp[2*thid]   = gi[2*thid];
    temp[2*thid+1] = gi[2*thid+1];
    for (int d = N >> 1; d > 0; d >>= 1)
    {
      /*
      DPCT1118:3: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:10: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    if (thid == 0) temp[N-1] = 0; // clear the last elem
    for (int d = 1; d < N; d *= 2) // traverse down
    {
      offset >>= 1;
      /*
      DPCT1118:4: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        T t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
    go[2*thid] = temp[2*thid];
    go[2*thid+1] = temp[2*thid+1];
  }
}

int main(int argc, char* argv[])
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (argc < 2) {
    printf("Usage: %s <out.bin> [number of elements] [repeat]\n", argv[0]);
    return 1;
  }
  typedef int T;
  const int N = 512;
  const int64_t n = argc > 2 ? atol(argv[2]) : 1048576;
  const int repeat = argc > 3 ? atoi(argv[3]) : 1;

  const int64_t num_blocks = (n + N - 1) / N;
  const int64_t nelems = num_blocks * N; // actual total number of elements
  const int64_t bytes = nelems * sizeof(T);

  std::vector<T> in(nelems), out(2 * nelems);
  fill_input(in.data(), nelems);

  T *d_in, *d_out;
  d_in = (T *)sycl::malloc_device(bytes, q_ct1);
  q_ct1.memcpy(d_in, in.data(), bytes).wait();
  d_out = (T *)sycl::malloc_device(bytes, q_ct1);

  dpct::device_info prop;
  dpct::get_device(0).get_device_info(prop);
  dpct::dim3 grids(16 * prop.get_max_compute_units());
  dpct::dim3 blocks(N / 2);

  dev_ct1.queues_wait_and_throw();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    /*
    DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 1> temp_acc_ct1(sycl::range<1>(N), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(grids * blocks, blocks),
          [=](sycl::nd_item<3> item_ct1) {
            scan<T, N>(
                num_blocks, d_out, d_in,
                temp_acc_ct1
                    .template get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
  }
  dev_ct1.queues_wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  q_ct1.memcpy(out.data(), d_out, bytes).wait();

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    /*
    DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 1> temp_acc_ct1(sycl::range<1>(2 * N), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(grids * blocks, blocks),
          [=](sycl::nd_item<3> item_ct1) {
            scan_bcao<T, N>(
                num_blocks, d_out, d_in,
                temp_acc_ct1
                    .template get_multi_ptr<sycl::access::decorated::no>()
                    .get());
          });
    });
  }
  dev_ct1.queues_wait_and_throw();
  end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  q_ct1.memcpy(out.data() + nelems, d_out, bytes).wait();
  printf("kernel_ms=%f\n", (time * 1e-6f) / (2 * repeat));

  std::vector<double> res(out.begin(), out.end());
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(res.data(), sizeof(double), res.size(), f) != res.size()) return 1;
  fclose(f);
  dpct::dpct_free(d_in, q_ct1);
  dpct::dpct_free(d_out, q_ct1);
  return 0;
}
