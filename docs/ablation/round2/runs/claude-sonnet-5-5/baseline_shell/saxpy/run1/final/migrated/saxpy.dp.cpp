// SYCL SAXPY. Usage: saxpy <out.bin> [n=1048576]
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

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
      std::fprintf(stderr, "allocation failed\n");
      return 1;
    }
    for (int i = 0; i < n; ++i) {
      x[i] = static_cast<float>(i % 1000) * 0.001f;
      y[i] = static_cast<float>((i * 7) % 1000) * 0.002f;
    }
    sycl::event e = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
      y[id] = a * x[id] + y[id];
    });
    e.wait_and_throw();
    double ms = (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                 e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
    std::printf("kernel_ms=%f\n", ms);
    FILE* f = std::fopen(argv[1], "wb");
    if (!f || std::fwrite(y, sizeof(float), n, f) != static_cast<size_t>(n)) return 1;
    std::fclose(f);
    sycl::free(x, q);
    sycl::free(y, q);
  } catch (const sycl::exception& ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    return 1;
  }
  return 0;
}
