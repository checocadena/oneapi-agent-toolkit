// SYCL SAXPY. Unified shared memory, exception-based errors, event profiling.
// Usage: saxpy <out.bin> [n=1048576]
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) try {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.bin [n]\n", argv[0]);
    return 2;
  }
  const int n = argc > 2 ? std::atoi(argv[2]) : (1 << 20);
  const float a = 2.5f;

  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{},
                                    sycl::property::queue::enable_profiling{}}};

  float *x = sycl::malloc_shared<float>(n, q);
  float *y = sycl::malloc_shared<float>(n, q);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i % 1000) * 0.001f;
    y[i] = static_cast<float>((i * 7) % 1000) * 0.002f;
  }

  sycl::event ev = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
    const size_t i = idx[0];
    y[i] = a * x[i] + y[i];
  });
  q.wait_and_throw();

  const float ms =
      (ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
       ev.get_profiling_info<sycl::info::event_profiling::command_start>()) /
      1000000.0f;
  std::printf("kernel_ms=%f\n", ms);

  FILE *f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(y, sizeof(float), n, f) != static_cast<size_t>(n))
    return 1;
  std::fclose(f);
  sycl::free(x, q);
  sycl::free(y, q);
  return 0;
} catch (sycl::exception const &exc) {
  std::cerr << "SYCL error: " << exc.what() << std::endl;
  return 1;
}
