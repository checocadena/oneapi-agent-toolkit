// Reference (ground truth) for the atomic reduction: the plain sum of the input (HeCBench's
// checksum), once for each of the 20 kernel / block-size runs.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/atomicReduction-cuda,
// commit f0c817aa). Copyright (c) 2015-2016 Advanced Micro Devices, Inc. MIT License; see
// LICENSE.AMD in this directory.
// Usage: reduction_ref <out.bin> [arrayLength=16777216]
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reduction_input.hpp"

int main(int argc, char* argv[]) {
  if (argc < 2) return 1;
  const int arrayLength = argc > 2 ? std::atoi(argv[2]) : 16777216;
  std::vector<int> array(arrayLength);
  fill_input(array.data(), arrayLength);
  long long checksum = 0;
  for (int v : array) checksum += v;
  std::vector<double> sums(20, static_cast<double>(checksum));
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(sums.data(), sizeof(double), sums.size(), f) != sums.size()) return 1;
  std::fclose(f);
  return 0;
}
