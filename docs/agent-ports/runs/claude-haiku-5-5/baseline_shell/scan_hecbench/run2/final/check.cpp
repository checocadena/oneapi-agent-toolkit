#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "migrated/scan_input.hpp"

int main(int argc, char** argv) {
  const int N = 512;
  int64_t n = argc > 2 ? atol(argv[2]) : 1048576;
  int64_t nb = (n + N - 1) / N;
  int64_t nelems = nb * N;
  std::vector<int> in(nelems);
  fill_input(in.data(), nelems);
  std::vector<double> ref(2 * nelems);
  for (int pass = 0; pass < 2; pass++)
    for (int64_t b = 0; b < nb; b++) {
      long long acc = 0;
      for (int i = 0; i < N; i++) {
        ref[pass * nelems + b * N + i] = (double)acc;
        acc += in[b * N + i];
      }
    }
  FILE* f = fopen(argv[1], "rb");
  if (!f) { printf("cannot open\n"); return 1; }
  std::vector<double> got(2 * nelems);
  size_t r = fread(got.data(), sizeof(double), got.size(), f);
  fclose(f);
  if (r != got.size()) { printf("size mismatch %zu vs %zu\n", r, got.size()); return 1; }
  int bad = 0;
  for (size_t i = 0; i < got.size(); i++)
    if (got[i] != ref[i]) { if (bad < 5) printf("diff at %zu: got %f ref %f\n", i, got[i], ref[i]); bad++; }
  printf("%s: %d mismatches of %zu\n", bad ? "FAIL" : "PASS", bad, got.size());
  return bad ? 1 : 0;
}
