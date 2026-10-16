// SYCL 2020 port of HeCBench matrix-rotate-cuda: in-place 90-degree rotation,
// one work-item per concentric layer.
// Usage: matrix_rotate <out.bin> [n=2048] [repeat=3]   (the output depends on repeat)
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void rotate_layers(sycl::queue &q, float *matrix, int n) {
  const int layers = n / 2;
  const size_t local = 256;
  const size_t global = ((static_cast<size_t>(layers) + local - 1) / local) * local;
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> it) {
                     const int layer = static_cast<int>(it.get_global_id(0));
                     if (layer < layers) {
                       const int first = layer;
                       const int last = n - 1 - layer;
                       for (int i = first; i < last; ++i) {
                         const int offset = i - first;

                         float top = matrix[first * n + i]; // save top
                         // left -> top
                         matrix[first * n + i] = matrix[(last - offset) * n + first];
                         // bottom -> left
                         matrix[(last - offset) * n + first] = matrix[last * n + (last - offset)];
                         // right -> bottom
                         matrix[last * n + (last - offset)] = matrix[i * n + last];
                         // top -> right
                         matrix[i * n + last] = top;
                       }
                     }
                   });
  });
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("Usage: %s <out.bin> [matrix size] [repeat]\n", argv[0]);
    return 1;
  }
  const int n = argc > 2 ? std::atoi(argv[2]) : 2048;
  const int repeat = argc > 3 ? std::atoi(argv[3]) : 3;

  std::vector<float> m(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      m[static_cast<size_t>(i) * n + j] = static_cast<float>(i * n + j);

  sycl::queue q{sycl::property::queue::in_order{}};
  float *d_m = sycl::malloc_device<float>(m.size(), q);
  q.memcpy(d_m, m.data(), m.size() * sizeof(float)).wait();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    rotate_layers(q, d_m, n);
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(m.data(), d_m, m.size() * sizeof(float)).wait();
  sycl::free(d_m, q);

  FILE *f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(m.data(), sizeof(float), m.size(), f) != m.size()) return 1;
  std::fclose(f);
  return 0;
}
