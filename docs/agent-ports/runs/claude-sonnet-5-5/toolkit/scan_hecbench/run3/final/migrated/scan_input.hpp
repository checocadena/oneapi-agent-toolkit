// Deterministic input shared by all scan builds (replaces HeCBench's srand(123)/rand() % 5 + 1,
// whose sequence depends on the C library). Values are integers in [1, 5], as in HeCBench.
#pragma once

#include <cstdint>

template <typename T>
inline void fill_input(T* x, int64_t n) {
  uint32_t s = 123u;
  for (int64_t i = 0; i < n; i++) {
    s = s * 1664525u + 1013904223u;  // Numerical Recipes LCG
    x[i] = static_cast<T>((s >> 16) % 5u + 1u);
  }
}
