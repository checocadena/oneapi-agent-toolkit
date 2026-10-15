// Reference (ground truth) for the softmax example: the CPU model from HeCBench
// softmax-cuda (softMax_cpu), unchanged in substance.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/softmax-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: deterministic LCG input (instead of rand()), result written to a file.
// Usage: softmax_ref <out.bin> [numSlice=8192] [sliceSize=256]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "softmax_input.hpp"

void softMax_cpu(const int numSlice, const int sliceSize, const float* src, float* dest) {
  for (int i = 0; i < numSlice; i++) {
    float max_ = src[i * sliceSize];
    for (int j = 0; j < sliceSize; j++) {
      max_ = (max_ < src[i * sliceSize + j]) ? src[i * sliceSize + j] : max_;
    }
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) {
      float e = expf(src[i * sliceSize + j] - max_);
      sum += e;
      dest[i * sliceSize + j] = e;
    }
    for (int j = 0; j < sliceSize; j++) {
      dest[i * sliceSize + j] /= sum;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.bin [numSlice] [sliceSize]\n", argv[0]);
    return 2;
  }
  const int numSlice = argc > 2 ? std::atoi(argv[2]) : 8192;
  const int sliceSize = argc > 3 ? std::atoi(argv[3]) : 256;
  const int numElem = numSlice * sliceSize;
  std::vector<float> input(numElem), output(numElem);
  fill_input(input.data(), numElem);
  softMax_cpu(numSlice, sliceSize, input.data(), output.data());
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(output.data(), sizeof(float), numElem, f) != size_t(numElem)) return 1;
  std::fclose(f);
  return 0;
}
