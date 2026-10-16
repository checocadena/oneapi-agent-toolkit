// SYCL 2020 SAXPY port (no dpct helpers). Unified shared memory, exception-based
// error handling, event-profiled kernel timing.
// Usage: saxpy <out.bin> [n=1048576]
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.bin [n]\n", argv[0]);
    return 2;
  }
  const int n = argc > 2 ? std::atoi(argv[2]) : (1 << 20);
  const float a = 2.5f;

  try {
    sycl::queue q{sycl::property::queue::enable_profiling{}};

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

    const std::size_t local = 256;
    const std::size_t global = ((static_cast<std::size_t>(n) + local - 1) / local) * local;

    sycl::event ev = q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>{sycl::range<1>{global}, sycl::range<1>{local}},
                     [=](sycl::nd_item<1> item) {
                       const int i = static_cast<int>(item.get_global_id(0));
                       if (i < n) y[i] = a * x[i] + y[i];
                     });
    });
    ev.wait_and_throw();

    const auto t_start = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto t_end = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
    const float ms = static_cast<float>(t_end - t_start) / 1.0e6f;
    std::printf("kernel_ms=%f\n", ms);

    FILE* f = std::fopen(argv[1], "wb");
    if (!f || std::fwrite(y, sizeof(float), n, f) != static_cast<size_t>(n)) {
      if (f) std::fclose(f);
      sycl::free(x, q);
      sycl::free(y, q);
      return 1;
    }
    std::fclose(f);

    sycl::free(x, q);
    sycl::free(y, q);
    return 0;
  } catch (const sycl::exception& e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
