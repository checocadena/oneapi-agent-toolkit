// In-place matrix rotation in SYCL 2020: one work-item per concentric layer.
// Usage: matrix_rotate <out.bin> [n=2048] [repeat=3]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <out.bin> [matrix size] [repeat]\n", argv[0]);
    return 1;
  }
  const int n = argc > 2 ? atoi(argv[2]) : 2048;
  const int repeat = argc > 3 ? atoi(argv[3]) : 3;

  std::vector<float> m(size_t(n) * n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      m[size_t(i) * n + j] = i * n + j;

  sycl::queue q{sycl::property::queue::in_order()};
  float *d_m = sycl::malloc_device<float>(m.size(), q);
  q.memcpy(d_m, m.data(), m.size() * sizeof(float)).wait();

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    q.parallel_for(sycl::range<1>(n / 2), [=](sycl::id<1> idx) {
      const int first = idx[0];
      const int last = n - 1 - first;
      for (int i = first; i < last; ++i) {
        const int offset = i - first;
        float top = d_m[first * n + i];
        d_m[first * n + i] = d_m[(last - offset) * n + first];
        d_m[(last - offset) * n + first] = d_m[last * n + (last - offset)];
        d_m[last * n + (last - offset)] = d_m[i * n + last];
        d_m[i * n + last] = top;
      }
    });
  }
  q.wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(m.data(), d_m, m.size() * sizeof(float)).wait();
  FILE *f = fopen(argv[1], "wb");
  if (!f || fwrite(m.data(), sizeof(float), m.size(), f) != m.size()) return 1;
  fclose(f);
  sycl::free(d_m, q);
  return 0;
}
