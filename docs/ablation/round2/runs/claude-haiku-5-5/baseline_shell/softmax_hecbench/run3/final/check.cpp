#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "softmax_input.hpp"
int main(int argc, char** argv) {
  int ns = atoi(argv[1]), ss = atoi(argv[2]);
  size_t n = (size_t)ns * ss;
  std::vector<float> in(n), out(n), ref(n);
  fill_input(in.data(), (int)n);
  FILE* f = fopen(argv[3], "rb");
  if (!f || fread(out.data(), sizeof(float), n, f) != n) { printf("read fail\n"); return 1; }
  for (int i = 0; i < ns; i++) {
    const float* x = &in[(size_t)i * ss];
    float m = x[0];
    for (int j = 0; j < ss; j++) m = std::max(m, x[j]);
    float s = 0;
    for (int j = 0; j < ss; j++) s += expf(x[j] - m);
    for (int j = 0; j < ss; j++) ref[(size_t)i * ss + j] = expf(x[j] - m) / s;
  }
  double maxerr = 0;
  for (size_t k = 0; k < n; k++) maxerr = std::fmax(maxerr, std::fabs(out[k] - ref[k]));
  printf("max_abs_err=%g %s\n", maxerr, maxerr <= 1e-6 ? "PASS" : "FAIL");
  return maxerr <= 1e-6 ? 0 : 2;
}
