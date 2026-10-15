// Hand-written idiomatic SYCL 2020 + oneMath port of gemm.cu (the target end state; no dpct
// helpers, no oneMKL-product-only API). cublasSgemm becomes
// oneapi::math::blas::column_major::gemm with run-time dispatch: the backend (MKL CPU here,
// cuBLAS/rocBLAS/MKL GPU elsewhere) is picked from the queue's device.
// Build: icpx -fsycl -I$ONEMATH_PREFIX/include gemm.cpp -L$ONEMATH_PREFIX/lib -lonemath
// Usage: gemm <out.bin> [M=384] [N=320] [K=256]
#include <sycl/sycl.hpp>

#include <oneapi/math.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

// Uniform in [-0.5, 0.5): 24 high bits of a 32-bit LCG.
static void fill(std::vector<float>& v, uint32_t seed) {
  uint32_t s = seed;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<float>(s >> 8) * (1.0f / 16777216.0f) - 0.5f;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.bin [M] [N] [K]\n", argv[0]);
    return 2;
  }
  const std::int64_t m = argc > 2 ? std::atoll(argv[2]) : 384;
  const std::int64_t n = argc > 3 ? std::atoll(argv[3]) : 320;
  const std::int64_t k = argc > 4 ? std::atoll(argv[4]) : 256;
  const float alpha = 1.5f, beta = -0.5f;
  const std::int64_t lda = m, ldb = k, ldc = m;  // column-major, no padding

  std::vector<float> a(static_cast<size_t>(lda * k)), b(static_cast<size_t>(ldb * n)),
      c(static_cast<size_t>(ldc * n));
  fill(a, 1u);
  fill(b, 2u);
  fill(c, 3u);

  try {
    sycl::queue q{sycl::property::queue::in_order{}};
    float* d_a = sycl::malloc_device<float>(a.size(), q);
    float* d_b = sycl::malloc_device<float>(b.size(), q);
    float* d_c = sycl::malloc_device<float>(c.size(), q);
    q.copy(a.data(), d_a, a.size());
    q.copy(b.data(), d_b, b.size());
    q.copy(c.data(), d_c, c.size()).wait();

    // Timed with std::chrono: includes backend dispatch (and, on first call, library loading).
    const auto t0 = std::chrono::steady_clock::now();
    oneapi::math::blas::column_major::gemm(q, oneapi::math::transpose::nontrans,
                                           oneapi::math::transpose::nontrans, m, n, k, alpha,
                                           d_a, lda, d_b, ldb, beta, d_c, ldc)
        .wait_and_throw();
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("kernel_ms=%f\n", std::chrono::duration<double, std::milli>(t1 - t0).count());

    q.copy(d_c, c.data(), c.size()).wait();
    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);
  } catch (const sycl::exception& ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    return 1;
  } catch (const std::exception& ex) {  // oneapi::math::exception derives from std::exception
    std::fprintf(stderr, "oneMath error: %s\n", ex.what());
    return 1;
  }

  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(c.data(), sizeof(float), c.size(), f) != c.size()) return 1;
  std::fclose(f);
  return 0;
}
