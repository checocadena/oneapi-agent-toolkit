// Block-wise exclusive scan (Blelloch up-sweep / down-sweep in local memory), plain and
// bank-conflict-avoiding, SYCL 2020 port of HeCBench scan-cuda.
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

constexpr int N = 512;  // elements scanned per work-group
using T = int;

static constexpr auto fence = sycl::access::fence_space::local_space;

void scan_bcao(sycl::nd_item<1> it, const int64_t nblocks, T *__restrict g_odata,
               const T *__restrict g_idata, T *temp) {
  for (int64_t bid = it.get_group(0); bid < nblocks; bid += it.get_group_range(0)) {
    auto gi = g_idata + bid * N;
    auto go = g_odata + bid * N;

    int thid = it.get_local_id(0);
    int a = thid;
    int b = a + (N / 2);
    int oa = OFFSET(a);
    int ob = OFFSET(b);

    temp[a + oa] = gi[a];
    temp[b + ob] = gi[b];

    int offset = 1;
    for (int d = N >> 1; d > 0; d >>= 1) {
      it.barrier(fence);
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
      it.barrier(fence);
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
    it.barrier(fence);

    go[a] = temp[a + oa];
    go[b] = temp[b + ob];
    // protect temp before the next grid-stride iteration overwrites it
    it.barrier(fence);
  }
}

void scan(sycl::nd_item<1> it, const int64_t nblocks, T *__restrict g_odata,
          const T *__restrict g_idata, T *temp) {
  for (int64_t bid = it.get_group(0); bid < nblocks; bid += it.get_group_range(0)) {
    auto gi = g_idata + bid * N;
    auto go = g_odata + bid * N;

    int thid = it.get_local_id(0);
    int offset = 1;
    temp[2 * thid] = gi[2 * thid];
    temp[2 * thid + 1] = gi[2 * thid + 1];
    for (int d = N >> 1; d > 0; d >>= 1) {
      it.barrier(fence);
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
      it.barrier(fence);
      if (thid < d) {
        int ai = offset * (2 * thid + 1) - 1;
        int bi = offset * (2 * thid + 2) - 1;
        T t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
    // the last down-sweep step must be visible before the copy-out, and the copy-out
    // must finish before the next iteration overwrites temp
    it.barrier(fence);
    go[2 * thid] = temp[2 * thid];
    go[2 * thid + 1] = temp[2 * thid + 1];
    it.barrier(fence);
  }
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
  q.memcpy(d_in, in.data(), bytes).wait();
  T *d_out = sycl::malloc_device<T>(nelems, q);

  const size_t groups =
      16 * q.get_device().get_info<sycl::info::device::max_compute_units>();
  const sycl::nd_range<1> range{sycl::range<1>(groups * (N / 2)), sycl::range<1>(N / 2)};

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 1> temp(sycl::range<1>(N), cgh);
      cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
        scan(it, num_blocks, d_out, d_in, &temp[0]);
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  q.memcpy(out.data(), d_out, bytes).wait();

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 1> temp(sycl::range<1>(2 * N), cgh);
      cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
        scan_bcao(it, num_blocks, d_out, d_in, &temp[0]);
      });
    });
  }
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
