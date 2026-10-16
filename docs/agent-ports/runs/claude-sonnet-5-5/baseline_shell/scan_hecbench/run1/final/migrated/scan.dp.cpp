// SYCL 2020 port of HeCBench scan-cuda: work-efficient Blelloch block-wise exclusive scan in
// work-group local memory, with and without bank-conflict-avoiding padding, over a grid-stride
// loop of work-groups.
// Usage: scan <out.bin> [number of elements=1048576] [repeat=1]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "scan_input.hpp"

#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)

using T = int;
constexpr int N = 512;  // elements scanned per work-group

static inline void local_barrier(const sycl::nd_item<1> &it) {
  sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <out.bin> [number of elements] [repeat]\n", argv[0]);
    return 1;
  }
  const int64_t n = argc > 2 ? atol(argv[2]) : 1048576;
  const int repeat = argc > 3 ? atoi(argv[3]) : 1;

  const int64_t num_blocks = (n + N - 1) / N;
  const int64_t nelems = num_blocks * N;
  const int64_t bytes = nelems * sizeof(T);

  std::vector<T> in(nelems), out(2 * nelems);
  fill_input(in.data(), nelems);

  sycl::queue q{sycl::property::queue::in_order()};
  T *d_in = sycl::malloc_device<T>(nelems, q);
  T *d_out = sycl::malloc_device<T>(nelems, q);
  q.memcpy(d_in, in.data(), bytes).wait();

  const size_t num_groups =
      16 * q.get_device().get_info<sycl::info::device::max_compute_units>();
  const size_t wg = N / 2;
  const sycl::nd_range<1> range{sycl::range<1>{num_groups * wg},
                                sycl::range<1>{wg}};

  auto run_scan = [&] {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 1> temp(sycl::range<1>(N), cgh);
      cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
        const int thid = it.get_local_id(0);
        for (int64_t bid = it.get_group(0); bid < num_blocks;
             bid += it.get_group_range(0)) {
          const T *gi = d_in + bid * N;
          T *go = d_out + bid * N;
          int offset = 1;
          temp[2 * thid] = gi[2 * thid];
          temp[2 * thid + 1] = gi[2 * thid + 1];
          for (int d = N >> 1; d > 0; d >>= 1) {
            local_barrier(it);
            if (thid < d) {
              int ai = offset * (2 * thid + 1) - 1;
              int bi = offset * (2 * thid + 2) - 1;
              temp[bi] += temp[ai];
            }
            offset *= 2;
          }
          if (thid == 0) temp[N - 1] = 0;
          for (int d = 1; d < N; d *= 2) {
            offset >>= 1;
            local_barrier(it);
            if (thid < d) {
              int ai = offset * (2 * thid + 1) - 1;
              int bi = offset * (2 * thid + 2) - 1;
              T t = temp[ai];
              temp[ai] = temp[bi];
              temp[bi] += t;
            }
          }
          go[2 * thid] = temp[2 * thid];
          go[2 * thid + 1] = temp[2 * thid + 1];
        }
      });
    });
  };

  auto run_scan_bcao = [&] {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 1> temp(sycl::range<1>(2 * N), cgh);
      cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
        const int thid = it.get_local_id(0);
        for (int64_t bid = it.get_group(0); bid < num_blocks;
             bid += it.get_group_range(0)) {
          const T *gi = d_in + bid * N;
          T *go = d_out + bid * N;
          int a = thid;
          int b = a + (N / 2);
          int oa = OFFSET(a);
          int ob = OFFSET(b);

          temp[a + oa] = gi[a];
          temp[b + ob] = gi[b];

          int offset = 1;
          for (int d = N >> 1; d > 0; d >>= 1) {
            local_barrier(it);
            if (thid < d) {
              int ai = offset * (2 * thid + 1) - 1;
              int bi = offset * (2 * thid + 2) - 1;
              ai += OFFSET(ai);
              bi += OFFSET(bi);
              temp[bi] += temp[ai];
            }
            offset *= 2;
          }

          if (thid == 0) temp[N - 1 + OFFSET(N - 1)] = 0;
          for (int d = 1; d < N; d *= 2) {
            offset >>= 1;
            local_barrier(it);
            if (thid < d) {
              int ai = offset * (2 * thid + 1) - 1;
              int bi = offset * (2 * thid + 2) - 1;
              ai += OFFSET(ai);
              bi += OFFSET(bi);
              T t = temp[ai];
              temp[ai] = temp[bi];
              temp[bi] += t;
            }
          }
          local_barrier(it);

          go[a] = temp[a + oa];
          go[b] = temp[b + ob];
        }
      });
    });
  };

  q.wait_and_throw();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) run_scan();
  q.wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  q.memcpy(out.data(), d_out, bytes).wait();

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) run_scan_bcao();
  q.wait_and_throw();
  end = std::chrono::steady_clock::now();
  time +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  q.memcpy(out.data() + nelems, d_out, bytes).wait();
  printf("kernel_ms=%f\n", (time * 1e-6f) / (2 * repeat));

  std::vector<double> res(out.begin(), out.end());
  FILE *f = fopen(argv[1], "wb");
  if (!f || fwrite(res.data(), sizeof(double), res.size(), f) != res.size())
    return 1;
  fclose(f);
  sycl::free(d_in, q);
  sycl::free(d_out, q);
  return 0;
}
