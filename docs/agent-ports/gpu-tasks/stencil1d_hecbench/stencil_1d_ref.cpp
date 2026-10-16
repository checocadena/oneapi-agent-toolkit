// Reference (ground truth) for the 1D stencil: each output is the sum of the input within
// RADIUS of it, where the input is a[i] = i and positions before the start count as 0
// (the kernel zero-fills the left halo of the first block).
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/stencil1d-cuda/stencil_1d.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Usage: stencil_1d_ref <out.bin> [length=1048576]
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int kRadius = 7;

int main(int argc, char* argv[]) {
  if (argc < 2) return 1;
  const int length = argc > 2 ? std::atoi(argv[2]) : 1048576;
  std::vector<double> out(length);
  for (int i = 0; i < length; i++) {
    long long s = 0;
    for (int j = i - kRadius; j <= i + kRadius; j++) s += j < 0 ? 0 : j;
    out[i] = static_cast<double>(s);
  }
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(out.data(), sizeof(double), length, f) != size_t(length)) return 1;
  std::fclose(f);
  return 0;
}
