// Reference (ground truth) for the matrix rotation: HeCBench's serial rotate_matrix_serial,
// unchanged, applied `repeat` times to m[i][j] = i*n + j.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/matrix-rotate-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Usage: matrix_rotate_ref <out.bin> [n=2048] [repeat=3]
#include <cstdio>
#include <cstdlib>
#include <vector>

void rotate_matrix_serial(float *matrix, int n) {
  for (int layer = 0; layer < n / 2; ++layer) {
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

int main(int argc, char* argv[]) {
  if (argc < 2) return 1;
  const int n = argc > 2 ? std::atoi(argv[2]) : 2048;
  const int repeat = argc > 3 ? std::atoi(argv[3]) : 3;
  std::vector<float> m(size_t(n) * n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) m[size_t(i) * n + j] = i * n + j;
  for (int r = 0; r < repeat; r++) rotate_matrix_serial(m.data(), n);
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(m.data(), sizeof(float), m.size(), f) != m.size()) return 1;
  std::fclose(f);
  return 0;
}
