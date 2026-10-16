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

static void rotate_matrix_parallel(float *matrix, const int n, sycl::nd_item<1> item) {
  int layer = static_cast<int>(item.get_global_id(0));
  if (layer < n / 2) {
    int first = layer;
    int last = n - 1 - layer;
    for (int i = first; i < last; ++i) {
      int offset = i - first;

      float top = matrix[first * n + i]; // save top
      // left -> top
      matrix[first * n + i] = matrix[(last - offset) * n + first];

      // bottom -> left
      matrix[(last - offset) * n + first] = matrix[last * n + (last - offset)];

      // right -> bottom
      matrix[last * n + (last - offset)] = matrix[i * n + last];

      // top -> right
      matrix[i * n + last] = top; // right <- saved top
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <out.bin> [matrix size] [repeat]\n", argv[0]);
    return 1;
  }
  const int n = argc > 2 ? atoi(argv[2]) : 2048;
  const int repeat = argc > 3 ? atoi(argv[3]) : 3;

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};

  std::vector<float> m(size_t(n) * n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      m[size_t(i) * n + j] = i * n + j;

  float *d_m = sycl::malloc_device<float>(m.size(), q);
  q.memcpy(d_m, m.data(), m.size() * sizeof(float)).wait();

  const size_t work_items = size_t(n / 2);
  const size_t local = 256;
  const size_t global = (work_items + local - 1) / local * local;

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                   [=](sycl::nd_item<1> item) {
                     rotate_matrix_parallel(d_m, n, item);
                   });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(m.data(), d_m, m.size() * sizeof(float)).wait();
  FILE *f = fopen(argv[1], "wb");
  if (!f || fwrite(m.data(), sizeof(float), m.size(), f) != m.size()) {
    sycl::free(d_m, q);
    return 1;
  }
  fclose(f);
  sycl::free(d_m, q);
  return 0;
}
