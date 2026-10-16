// Reference (ground truth) for the Adam step: HeCBench's CPU reference() (reference.h,
// unchanged) on the same input.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/adam-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Usage: adam_ref <out.bin> [vector size=1048576] [time steps=10] [repeat=1]
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "adam_input.hpp"
#include "reference.h"

int main(int argc, char* argv[]) {
  if (argc < 2) return 1;
  const int vector_size = argc > 2 ? std::atoi(argv[2]) : 1048576;
  const int time_step = argc > 3 ? std::atoi(argv[3]) : 10;
  const int repeat = argc > 4 ? std::atoi(argv[4]) : 1;
  std::vector<float> m(vector_size), v(vector_size), g(vector_size), p(vector_size);
  fill_input(m.data(), v.data(), g.data(), p.data(), vector_size);
  reference<float, float>(repeat, p.data(), m.data(), v.data(), g.data(), 0.9f, 0.999f, 1e-8f,
                          256.f, 1e-3f, time_step, vector_size, ADAM_MODE_0, 0.5f);
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(p.data(), sizeof(float), p.size(), f) != p.size()) return 1;
  std::fclose(f);
  return 0;
}
