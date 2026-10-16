// SYCL 2020 port of HeCBench adam-cuda: applies `time_step` Adam updates to each parameter.
// Usage: adam <out.bin> [vector size=1048576] [time steps=10] [repeat=1]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "adam_input.hpp"
#include "reference.h"

int main(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage: %s <out.bin> [vector size] [number of time steps] [repeat]\n", argv[0]);
    return 1;
  }
  const int vector_size = argc > 2 ? atoi(argv[2]) : 1048576;
  const int time_step = argc > 3 ? atoi(argv[3]) : 10;
  const int repeat = argc > 4 ? atoi(argv[4]) : 1;

  std::vector<float> m(vector_size), v(vector_size), g(vector_size), p(vector_size);
  fill_input(m.data(), v.data(), g.data(), p.data(), vector_size);

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;
  const adamMode_t mode = ADAM_MODE_0;

  // Bias-correction denominators 1 - beta^t depend only on t: compute them once on the host.
  std::vector<float> bc1(time_step + 1), bc2(time_step + 1);
  for (int t = 1; t <= time_step; t++) {
    bc1[t] = 1.f - powf(beta1, t);
    bc2[t] = 1.f - powf(beta2, t);
  }

  sycl::queue q{sycl::property::queue::in_order()};

  const size_t n = vector_size;
  float* d_m = sycl::malloc_device<float>(n, q);
  float* d_v = sycl::malloc_device<float>(n, q);
  float* d_g = sycl::malloc_device<float>(n, q);
  float* d_p = sycl::malloc_device<float>(n, q);
  float* d_bc1 = sycl::malloc_device<float>(bc1.size(), q);
  float* d_bc2 = sycl::malloc_device<float>(bc2.size(), q);
  q.copy(m.data(), d_m, n);
  q.copy(v.data(), d_v, n);
  q.copy(g.data(), d_g, n);
  q.copy(p.data(), d_p, n);
  q.copy(bc1.data(), d_bc1, bc1.size());
  q.copy(bc2.data(), d_bc2, bc2.size());
  q.wait();

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
      const size_t j = idx[0];
      for (int t = 1; t <= time_step; t++) {
        float scaled_grad = d_g[j] / grad_scale;
        d_m[j] = beta1 * d_m[j] + (1.f - beta1) * scaled_grad;
        d_v[j] = beta2 * d_v[j] + (1.f - beta2) * scaled_grad * scaled_grad;
        float m_corrected = d_m[j] / d_bc1[t];
        float v_corrected = d_v[j] / d_bc2[t];
        float denom;
        if (mode == ADAM_MODE_0)
          denom = sycl::sqrt(v_corrected + eps);
        else
          denom = sycl::sqrt(v_corrected) + eps;
        float update = (m_corrected / denom) + (decay * d_p[j]);
        d_p[j] -= (step_size * update);
      }
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", time * 1e-6f / repeat);

  q.copy(d_p, p.data(), n).wait();
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(p.data(), sizeof(float), p.size(), f) != p.size()) return 1;
  fclose(f);
  sycl::free(d_p, q);
  sycl::free(d_m, q);
  sycl::free(d_v, q);
  sycl::free(d_g, q);
  sycl::free(d_bc1, q);
  sycl::free(d_bc2, q);
  return 0;
}
