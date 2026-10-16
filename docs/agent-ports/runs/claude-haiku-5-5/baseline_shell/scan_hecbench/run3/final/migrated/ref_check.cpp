#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "scan_input.hpp"
int main(int argc, char** argv) {
  const int N = 512;
  int64_t n = argc > 2 ? atol(argv[2]) : 1048576;
  int64_t nb = (n + N - 1) / N, ne = nb * N;
  std::vector<int> in(ne);
  fill_input(in.data(), ne);
  std::vector<double> ref(2 * ne);
  for (int pass = 0; pass < 2; pass++)
    for (int64_t b = 0; b < nb; b++) {
      long long acc = 0;
      for (int i = 0; i < N; i++) {
        ref[pass * ne + b * N + i] = (double)acc;
        acc += in[b * N + i];
      }
    }
  FILE* f = fopen(argv[1], "rb");
  std::vector<double> got(2 * ne);
  size_t r = fread(got.data(), sizeof(double), got.size(), f);
  fclose(f);
  if (r != got.size()) { printf("short read %zu\n", r); return 1; }
  size_t bad = 0;
  for (size_t i = 0; i < got.size(); i++) if (got[i] != ref[i]) { if (bad < 5) printf("mismatch %zu got %f ref %f\n", i, got[i], ref[i]); bad++; }
  printf("%s: %zu mismatches of %zu\n", bad ? "FAIL" : "PASS", bad, got.size());
  return bad != 0;
}
