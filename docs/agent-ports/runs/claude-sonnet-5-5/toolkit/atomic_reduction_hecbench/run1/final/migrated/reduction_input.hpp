// Deterministic input shared by all atomic-reduction builds (replaces HeCBench's rand() % 2,
// whose sequence depends on the C library). Values are 0 or 1, as in HeCBench.
#pragma once

#include <cstdint>

inline void fill_input(int* x, int n) {
  uint32_t s = 7u;
  for (int i = 0; i < n; i++) {
    s = s * 1664525u + 1013904223u;  // Numerical Recipes LCG
    x[i] = static_cast<int>((s >> 16) & 1u);
  }
}
