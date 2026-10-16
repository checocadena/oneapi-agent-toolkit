// Deterministic input shared by all softmax builds (replaces HeCBench's srand(2)/rand() % 13,
// whose sequence depends on the C library). Values are integers in [0, 12], as in HeCBench.
#pragma once

#include <cstdint>

inline void fill_input(float* x, int n) {
  uint32_t s = 2u;
  for (int i = 0; i < n; i++) {
    s = s * 1664525u + 1013904223u;  // Numerical Recipes LCG
    x[i] = static_cast<float>((s >> 16) % 13u);
  }
}
