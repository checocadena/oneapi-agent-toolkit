// Reference (ground truth): SAXPY y = a*x + y, plain C++.
// Inputs are generated deterministically; the result is written as raw float32.
// Usage: saxpy_ref <out.bin> [n=1048576]
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s out.bin [n]\n", argv[0]);
    return 2;
  }
  const size_t n = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : (1u << 20);
  const float a = 2.5f;
  std::vector<float> x(n), y(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i % 1000) * 0.001f;
    y[i] = static_cast<float>((i * 7) % 1000) * 0.002f;
  }
  for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(y.data(), sizeof(float), n, f) != n) return 1;
  std::fclose(f);
  return 0;
}
