#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "adam_input.hpp"
#include "reference.h"
// usage: ref_main <cand.bin> [n] [steps]
int main(int argc, char** argv) {
  const int n = argc > 2 ? atoi(argv[2]) : 1048576;
  const int steps = argc > 3 ? atoi(argv[3]) : 10;
  std::vector<float> m(n), v(n), g(n), p(n);
  fill_input(m.data(), v.data(), g.data(), p.data(), n);
  reference<float, float>(1, p.data(), m.data(), v.data(), g.data(), 0.9f, 0.999f, 1e-8f,
                          256.f, 1e-3f, steps, n, ADAM_MODE_0, 0.5f);
  std::vector<float> c(n);
  FILE* f = fopen(argv[1], "rb");
  if (!f || fread(c.data(), sizeof(float), n, f) != (size_t)n) { printf("read fail\n"); return 1; }
  fclose(f);
  double worst = 0; size_t wi = 0; size_t bad = 0;
  for (int i = 0; i < n; i++) {
    double d = std::fabs((double)c[i] - p[i]);
    double tol = 1e-5 + 1e-4 * std::fabs(p[i]);
    if (d > tol) bad++;
    if (d > worst) { worst = d; wi = i; }
  }
  printf("max_abs_diff=%.3e at %zu (ref=%g cand=%g) violations=%zu\n", worst, wi, p[wi], c[wi], bad);
  return bad ? 2 : 0;
}
