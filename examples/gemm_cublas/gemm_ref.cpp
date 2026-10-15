// Reference (ground truth): C = alpha * A * B + beta * C, column-major, naive triple loop
// with double accumulation, rounded to float32 once at the end.
// Inputs are the same deterministic LCG as gemm.cu. Usage: gemm_ref <out.bin> [M] [N] [K]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Uniform in [-0.5, 0.5): 24 high bits of a 32-bit LCG.
static void fill(std::vector<float>& v, uint32_t seed) {
  uint32_t s = seed;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<float>(s >> 8) * (1.0f / 16777216.0f) - 0.5f;
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.bin [M] [N] [K]\n", argv[0]);
    return 2;
  }
  const size_t m = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 384;
  const size_t n = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 320;
  const size_t k = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 256;
  const double alpha = 1.5, beta = -0.5;
  const size_t lda = m, ldb = k, ldc = m;

  std::vector<float> a(lda * k), b(ldb * n), c(ldc * n);
  fill(a, 1u);
  fill(b, 2u);
  fill(c, 3u);

  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) {
      double acc = 0.0;
      for (size_t p = 0; p < k; ++p)
        acc += static_cast<double>(a[i + p * lda]) * static_cast<double>(b[p + j * ldb]);
      c[i + j * ldc] = static_cast<float>(alpha * acc + beta * static_cast<double>(c[i + j * ldc]));
    }
  }

  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(c.data(), sizeof(float), c.size(), f) != c.size()) return 1;
  std::fclose(f);
  return 0;
}
