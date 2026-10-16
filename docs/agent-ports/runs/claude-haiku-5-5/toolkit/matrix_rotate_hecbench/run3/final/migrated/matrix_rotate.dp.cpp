// CUDA in-place matrix rotation (the code to be ported): HeCBench matrix-rotate-cuda, one
// thread per concentric layer, each rotating its ring by 90 degrees.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/matrix-rotate-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: kernel unchanged; the result is written to a file instead of being compared with
// an in-process serial rotation; argument order changed.
// Usage: matrix_rotate <out.bin> [n=2048] [repeat=3]   (the output depends on repeat)
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
      m[size_t(i)*n+j] = i*n+j;

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};

  float *d_m = sycl::malloc_device<float>(m.size(), q);
  q.memcpy(d_m, m.data(), m.size() * sizeof(float)).wait();

  const size_t layers = size_t(n / 2);
  const size_t local = 256;
  const size_t global = (layers + local - 1) / local * local;

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> item) {
      int layer = int(item.get_global_id(0));
      if (layer < n/2) {
        int first = layer;
        int last = n - 1 - layer;
        for(int i = first; i < last; ++i) {
          int offset = i - first;

          float top = d_m[first*n+i]; // save top
          // left -> top
          d_m[first*n+i] = d_m[(last-offset)*n+first];

          // bottom -> left
          d_m[(last-offset)*n+first] = d_m[last*n+(last-offset)];

          // right -> bottom
          d_m[last*n+(last-offset)] = d_m[i*n+last];

          // top -> right
          d_m[i*n+last] = top; // right <- saved top
        }
      }
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(m.data(), d_m, m.size() * sizeof(float)).wait();
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(m.data(), sizeof(float), m.size(), f) != m.size()) return 1;
  fclose(f);
  sycl::free(d_m, q);
  return 0;
}
