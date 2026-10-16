// Block-wise exclusive scan (Blelloch up-sweep / down-sweep in work-group local memory),
// plain and bank-conflict-avoiding, over a grid-stride loop of blocks. SYCL 2020 port of
// HeCBench scan-cuda.
// Usage: scan <out.bin> [number of elements=1048576] [repeat=1]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "scan_input.hpp"

constexpr int LOG_MEM_BANKS = 5;
constexpr int offset_of(int n) { return n >> LOG_MEM_BANKS; }

template <typename T, int N>
void run_scan_bcao(sycl::queue &q, int64_t nblocks, size_t ngroups, T *g_odata, const T *g_idata) {
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<T, 1> temp(sycl::range<1>(2 * N), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(ngroups * (N / 2), N / 2), [=](sycl::nd_item<1> it) {
          auto grp = it.get_group();
          const int thid = it.get_local_id(0);
          for (int64_t bid = it.get_group(0); bid < nblocks; bid += it.get_group_range(0)) {
            const T *gi = g_idata + bid * N;
            T *go = g_odata + bid * N;

            int a = thid;
            int b = a + (N / 2);
            int oa = offset_of(a);
            int ob = offset_of(b);

            temp[a + oa] = gi[a];
            temp[b + ob] = gi[b];

            int offset = 1;
            for (int d = N >> 1; d > 0; d >>= 1) {
              sycl::group_barrier(grp);
              if (thid < d) {
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;
                ai += offset_of(ai);
                bi += offset_of(bi);
                temp[bi] += temp[ai];
              }
              offset *= 2;
            }

            if (thid == 0) temp[N - 1 + offset_of(N - 1)] = 0;  // clear the last elem
            for (int d = 1; d < N; d *= 2) {                    // traverse down
              offset >>= 1;
              sycl::group_barrier(grp);
              if (thid < d) {
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;
                ai += offset_of(ai);
                bi += offset_of(bi);
                T t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
              }
            }
            sycl::group_barrier(grp);

            go[a] = temp[a + oa];
            go[b] = temp[b + ob];
          }
        });
  });
}

template <typename T, int N>
void run_scan(sycl::queue &q, int64_t nblocks, size_t ngroups, T *g_odata, const T *g_idata) {
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<T, 1> temp(sycl::range<1>(N), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(ngroups * (N / 2), N / 2), [=](sycl::nd_item<1> it) {
          auto grp = it.get_group();
          const int thid = it.get_local_id(0);
          for (int64_t bid = it.get_group(0); bid < nblocks; bid += it.get_group_range(0)) {
            const T *gi = g_idata + bid * N;
            T *go = g_odata + bid * N;

            int offset = 1;
            temp[2 * thid] = gi[2 * thid];
            temp[2 * thid + 1] = gi[2 * thid + 1];
            for (int d = N >> 1; d > 0; d >>= 1) {
              sycl::group_barrier(grp);
              if (thid < d) {
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;
                temp[bi] += temp[ai];
              }
              offset *= 2;
            }

            if (thid == 0) temp[N - 1] = 0;  // clear the last elem
            for (int d = 1; d < N; d *= 2) { // traverse down
              offset >>= 1;
              sycl::group_barrier(grp);
              if (thid < d) {
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;
                T t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
              }
            }
            sycl::group_barrier(grp);
            go[2 * thid] = temp[2 * thid];
            go[2 * thid + 1] = temp[2 * thid + 1];
          }
        });
  });
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <out.bin> [number of elements] [repeat]\n", argv[0]);
    return 1;
  }
  typedef int T;
  constexpr int N = 512;
  const int64_t n = argc > 2 ? atol(argv[2]) : 1048576;
  const int repeat = argc > 3 ? atoi(argv[3]) : 1;

  const int64_t num_blocks = (n + N - 1) / N;
  const int64_t nelems = num_blocks * N;
  const size_t bytes = nelems * sizeof(T);

  std::vector<T> in(nelems), out(2 * nelems);
  fill_input(in.data(), nelems);

  sycl::queue q{sycl::property::queue::in_order()};
  T *d_in = sycl::malloc_device<T>(nelems, q);
  T *d_out = sycl::malloc_device<T>(nelems, q);
  q.memcpy(d_in, in.data(), bytes).wait();

  const size_t ngroups = 16 * q.get_device().get_info<sycl::info::device::max_compute_units>();

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) run_scan<T, N>(q, num_blocks, ngroups, d_out, d_in);
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  q.memcpy(out.data(), d_out, bytes).wait();

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) run_scan_bcao<T, N>(q, num_blocks, ngroups, d_out, d_in);
  q.wait();
  end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  q.memcpy(out.data() + nelems, d_out, bytes).wait();
  printf("kernel_ms=%f\n", (time * 1e-6f) / (2 * repeat));

  std::vector<double> res(out.begin(), out.end());
  FILE *f = fopen(argv[1], "wb");
  if (!f || fwrite(res.data(), sizeof(double), res.size(), f) != res.size()) return 1;
  fclose(f);
  sycl::free(d_in, q);
  sycl::free(d_out, q);
  return 0;
}
