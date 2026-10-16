// Deterministic input shared by all Adam builds. HeCBench draws m, v, g and p from
// std::uniform_real_distribution over std::mt19937(19937); the generator is fully specified by
// the C++ standard but the distribution is not, so the same draws are mapped to [0, 1) here
// with an explicit 24-bit conversion.
#pragma once

#include <random>

inline float uniform01(std::mt19937& gen) {
  return static_cast<float>(gen() >> 8) * (1.0f / 16777216.0f);
}

inline void fill_input(float* m, float* v, float* g, float* p, int n) {
  std::mt19937 gen(19937);
  for (int i = 0; i < n; i++) {
    m[i] = uniform01(gen);
    v[i] = uniform01(gen);
    g[i] = uniform01(gen);
    p[i] = uniform01(gen);
  }
}
