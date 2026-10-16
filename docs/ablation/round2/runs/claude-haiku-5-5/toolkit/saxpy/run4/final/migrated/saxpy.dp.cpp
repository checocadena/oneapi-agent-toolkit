// SYCL SAXPY port. Unified (shared) memory, exception-based error handling, event profiling.
// Usage: saxpy <out.bin> [n=1048576]
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.bin [n]\n", argv[0]);
    return 2;
  }
  try {
    const int n = argc > 2 ? std::atoi(argv[2]) : (1 << 20);
    const float a = 2.5f;

    sycl::queue q{sycl::default_selector_v,
                  [](sycl::exception_list el) {
                    for (auto& e : el) std::rethrow_exception(e);
                  },
                  sycl::property::queue::enable_profiling{}};

    float* x = sycl::malloc_shared<float>(n, q);
    float* y = sycl::malloc_shared<float>(n, q);
    if (!x || !y) {
      std::fprintf(stderr, "USM allocation failed\n");
      return 1;
    }

    for (int i = 0; i < n; ++i) {
      x[i] = static_cast<float>(i % 1000) * 0.001f;
      y[i] = static_cast<float>((i * 7) % 1000) * 0.002f;
    }

    sycl::event ev = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      y[i] = a * x[i] + y[i];
    });
    q.wait_and_throw();

    const auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
    const float ms = static_cast<float>(t1 - t0) * 1e-6f;
    std::printf("kernel_ms=%f\n", ms);

    std::FILE* f = std::fopen(argv[1], "wb");
    if (!f || std::fwrite(y, sizeof(float), n, f) != static_cast<size_t>(n)) return 1;
    std::fclose(f);

    sycl::free(x, q);
    sycl::free(y, q);
    return 0;
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL error: " << e.what() << std::endl;
    return 1;
  }
}
