// Hand-written idiomatic SYCL 2020 port of matrix_rotate.cu (the target end state; no dpct
// helpers).
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/matrix-rotate-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
//
// Port notes: one work-item per concentric layer, as in CUDA. Layers touch disjoint rings, so
// the in-place update needs no synchronisation. The global size is rounded up to the 256-wide
// work-group and surplus work-items return early.
// Usage: matrix_rotate <out.bin> [n=2048] [repeat=3]   (the output depends on repeat)
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr size_t kBlockSize = 256;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::printf("Usage: %s <out.bin> [matrix size] [repeat]\n", argv[0]);
    return 1;
  }
  const int n = argc > 2 ? std::atoi(argv[2]) : 2048;
  const int repeat = argc > 3 ? std::atoi(argv[3]) : 3;

  std::vector<float> m(size_t(n) * n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) m[size_t(i) * n + j] = i * n + j;

  try {
    sycl::queue q;
    float* matrix = sycl::malloc_device<float>(m.size(), q);
    q.memcpy(matrix, m.data(), sizeof(float) * m.size()).wait();

    const size_t global = (size_t(n / 2) + kBlockSize - 1) / kBlockSize * kBlockSize;
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      q.parallel_for(sycl::nd_range<1>{global, kBlockSize}, [=](sycl::nd_item<1> it) {
        const int layer = it.get_global_id(0);
        if (layer >= n / 2) return;
        const int first = layer;
        const int last = n - 1 - layer;
        for (int i = first; i < last; ++i) {
          const int offset = i - first;
          const float top = matrix[first * n + i];
          matrix[first * n + i] = matrix[(last - offset) * n + first];
          matrix[(last - offset) * n + first] = matrix[last * n + (last - offset)];
          matrix[last * n + (last - offset)] = matrix[i * n + last];
          matrix[i * n + last] = top;
        }
      });
    }
    q.wait_and_throw();
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("kernel_ms=%f\n", (ns * 1e-6) / repeat);

    q.memcpy(m.data(), matrix, sizeof(float) * m.size()).wait();
    sycl::free(matrix, q);
  } catch (const sycl::exception& ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    return 1;
  }
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(m.data(), sizeof(float), m.size(), f) != m.size()) return 1;
  std::fclose(f);
  return 0;
}
