// CUDA in-place matrix rotation (the code to be ported): HeCBench matrix-rotate-cuda, one
// thread per concentric layer, each rotating its ring by 90 degrees.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/matrix-rotate-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: kernel unchanged; the result is written to a file instead of being compared with
// an in-process serial rotation; argument order changed.
// Usage: matrix_rotate <out.bin> [n=2048] [repeat=3]   (the output depends on repeat)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda.h>

__global__ void rotate_matrix_parallel (float *matrix, const int n) {
  int layer = blockIdx.x * blockDim.x + threadIdx.x;
  if (layer < n/2) {
    int first = layer;
    int last = n - 1 - layer;
    for(int i = first; i < last; ++i) {
      int offset = i - first;

      float top = matrix[first*n+i]; // save top
      // left -> top
      matrix[first*n+i] = matrix[(last-offset)*n+first];

      // bottom -> left
      matrix[(last-offset)*n+first] = matrix[last*n+(last-offset)];

      // right -> bottom
      matrix[last*n+(last-offset)] = matrix[i*n+last];

      // top -> right
      matrix[i*n+last] = top; // right <- saved top
    }
  }
}

int main(int argc, char** argv) {
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

  float *d_m;
  cudaMalloc((void**)&d_m, m.size()*sizeof(float));
  cudaMemcpy(d_m, m.data(), m.size()*sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    rotate_matrix_parallel<<<(n/2+255)/256, 256>>>(d_m, n);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  cudaMemcpy(m.data(), d_m, m.size()*sizeof(float), cudaMemcpyDeviceToHost);
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(m.data(), sizeof(float), m.size(), f) != m.size()) return 1;
  fclose(f);
  cudaFree(d_m);
  return 0;
}
