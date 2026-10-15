// Hand-written idiomatic SYCL 2020 port of scan.cu (the target end state; no dpct helpers).
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/scan-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
//
// Port notes: __shared__ arrays become sycl::local_accessor, __syncthreads() becomes
// sycl::group_barrier, and the CUDA grid of 16 blocks per multiprocessor becomes 16
// work-groups per compute unit. The barriers stay where CUDA has them, so both kernels keep
// their grid-stride loop over blocks.
// Usage: scan <out.bin> [number of elements=1048576] [repeat=1]
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../scan_input.hpp"

constexpr int kLogMemBanks = 5;
constexpr int offset_of(int n) { return n >> kLogMemBanks; }

// Plain Blelloch scan of N elements per work-group of N/2 work-items.
template <typename T, int N>
void scan(sycl::queue& q, size_t groups, int64_t nblocks, T* g_odata, const T* g_idata) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<T, 1> temp(sycl::range<1>(N), h);
    h.parallel_for(sycl::nd_range<1>{groups * (N / 2), N / 2}, [=](sycl::nd_item<1> it) {
      const auto g = it.get_group();
      const int thid = it.get_local_id(0);
      for (int64_t bid = it.get_group(0); bid < nblocks; bid += it.get_group_range(0)) {
        const T* gi = g_idata + bid * N;
        T* go = g_odata + bid * N;
        int offset = 1;
        temp[2 * thid] = gi[2 * thid];
        temp[2 * thid + 1] = gi[2 * thid + 1];
        for (int d = N >> 1; d > 0; d >>= 1) {
          sycl::group_barrier(g);
          if (thid < d) {
            const int ai = offset * (2 * thid + 1) - 1;
            const int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
          }
          offset *= 2;
        }
        if (thid == 0) temp[N - 1] = 0;
        for (int d = 1; d < N; d *= 2) {
          offset >>= 1;
          sycl::group_barrier(g);
          if (thid < d) {
            const int ai = offset * (2 * thid + 1) - 1;
            const int bi = offset * (2 * thid + 2) - 1;
            const T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
          }
        }
        go[2 * thid] = temp[2 * thid];
        go[2 * thid + 1] = temp[2 * thid + 1];
      }
    });
  });
}

// The same scan with one padding slot every 2^kLogMemBanks elements (bank-conflict avoiding).
template <typename T, int N>
void scan_bcao(sycl::queue& q, size_t groups, int64_t nblocks, T* g_odata, const T* g_idata) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<T, 1> temp(sycl::range<1>(2 * N), h);
    h.parallel_for(sycl::nd_range<1>{groups * (N / 2), N / 2}, [=](sycl::nd_item<1> it) {
      const auto g = it.get_group();
      const int thid = it.get_local_id(0);
      for (int64_t bid = it.get_group(0); bid < nblocks; bid += it.get_group_range(0)) {
        const T* gi = g_idata + bid * N;
        T* go = g_odata + bid * N;
        const int a = thid;
        const int b = a + N / 2;
        const int oa = offset_of(a);
        const int ob = offset_of(b);
        temp[a + oa] = gi[a];
        temp[b + ob] = gi[b];
        int offset = 1;
        for (int d = N >> 1; d > 0; d >>= 1) {
          sycl::group_barrier(g);
          if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += offset_of(ai);
            bi += offset_of(bi);
            temp[bi] += temp[ai];
          }
          offset *= 2;
        }
        if (thid == 0) temp[N - 1 + offset_of(N - 1)] = 0;
        for (int d = 1; d < N; d *= 2) {
          offset >>= 1;
          sycl::group_barrier(g);
          if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += offset_of(ai);
            bi += offset_of(bi);
            const T t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
          }
        }
        sycl::group_barrier(g);
        go[a] = temp[a + oa];
        go[b] = temp[b + ob];
      }
    });
  });
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::printf("Usage: %s <out.bin> [number of elements] [repeat]\n", argv[0]);
    return 1;
  }
  using T = int;
  constexpr int N = 512;
  const int64_t n = argc > 2 ? std::atol(argv[2]) : 1048576;
  const int repeat = argc > 3 ? std::atoi(argv[3]) : 1;
  const int64_t num_blocks = (n + N - 1) / N;
  const int64_t nelems = num_blocks * N;

  std::vector<T> in(nelems), out(2 * nelems);
  fill_input(in.data(), nelems);

  try {
    sycl::queue q{sycl::property::queue::in_order()};
    T* d_in = sycl::malloc_device<T>(nelems, q);
    T* d_out = sycl::malloc_device<T>(nelems, q);
    q.memcpy(d_in, in.data(), sizeof(T) * nelems).wait();
    const size_t groups =
        16 * q.get_device().get_info<sycl::info::device::max_compute_units>();

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) scan<T, N>(q, groups, num_blocks, d_out, d_in);
    q.wait_and_throw();
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    q.memcpy(out.data(), d_out, sizeof(T) * nelems).wait();

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) scan_bcao<T, N>(q, groups, num_blocks, d_out, d_in);
    q.wait_and_throw();
    end = std::chrono::steady_clock::now();
    ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    q.memcpy(out.data() + nelems, d_out, sizeof(T) * nelems).wait();
    std::printf("kernel_ms=%f\n", (ns * 1e-6) / (2 * repeat));

    sycl::free(d_in, q);
    sycl::free(d_out, q);
  } catch (const sycl::exception& ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    return 1;
  }
  std::vector<double> res(out.begin(), out.end());
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(res.data(), sizeof(double), res.size(), f) != res.size()) return 1;
  std::fclose(f);
  return 0;
}
