// Reference (ground truth) for the scan: HeCBench's serial block-wise exclusive scan, written
// twice (once for each GPU kernel: plain and bank-conflict-avoiding).
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/scan-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Usage: scan_ref <out.bin> [number of elements=1048576]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "scan_input.hpp"

int main(int argc, char* argv[]) {
  if (argc < 2) return 1;
  const int N = 512;
  const int64_t n = argc > 2 ? std::atol(argv[2]) : 1048576;
  const int64_t num_blocks = (n + N - 1) / N;
  const int64_t nelems = num_blocks * N;
  std::vector<int> in(nelems);
  fill_input(in.data(), nelems);
  std::vector<double> out(2 * nelems);
  for (int64_t b = 0; b < num_blocks; b++) {
    const int* t_in = in.data() + b * N;
    double* t_out = out.data() + b * N;
    t_out[0] = 0;
    for (int i = 1; i < N; i++) t_out[i] = t_out[i - 1] + t_in[i - 1];
  }
  for (int64_t i = 0; i < nelems; i++) out[nelems + i] = out[i];
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(out.data(), sizeof(double), out.size(), f) != out.size()) return 1;
  std::fclose(f);
  return 0;
}
