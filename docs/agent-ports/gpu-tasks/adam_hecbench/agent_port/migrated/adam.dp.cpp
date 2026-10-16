// SYCL Adam optimizer step: HeCBench adam-cuda, applies `time_step` Adam updates
// to each parameter.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/adam-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
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

  const size_t n = static_cast<size_t>(vector_size);
  size_t size_bytes = n * sizeof(float);
  std::vector<float> m(n), v(n), g(n), p(n);
  fill_input(m.data(), v.data(), g.data(), p.data(), vector_size);

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;
  const adamMode_t mode = ADAM_MODE_0;

  // Bias-correction denominators 1 - beta^t depend only on t; compute them once on the
  // host with the same libm powf as the reference so the cancellation is bit-identical.
  std::vector<float> bc1(time_step + 1), bc2(time_step + 1);
  for (int t = 1; t <= time_step; t++) {
    bc1[t] = 1.f - powf(beta1, t);
    bc2[t] = 1.f - powf(beta2, t);
  }

  sycl::queue q{sycl::property::queue::in_order()};

  float* d_m = sycl::malloc_device<float>(n, q);
  float* d_v = sycl::malloc_device<float>(n, q);
  float* d_g = sycl::malloc_device<float>(n, q);
  float* d_p = sycl::malloc_device<float>(n, q);
  float* d_bc1 = sycl::malloc_device<float>(bc1.size(), q);
  float* d_bc2 = sycl::malloc_device<float>(bc2.size(), q);
  q.memcpy(d_m, m.data(), size_bytes);
  q.memcpy(d_v, v.data(), size_bytes);
  q.memcpy(d_g, g.data(), size_bytes);
  q.memcpy(d_p, p.data(), size_bytes);
  q.memcpy(d_bc1, bc1.data(), bc1.size() * sizeof(float));
  q.memcpy(d_bc2, bc2.data(), bc2.size() * sizeof(float));
  q.wait();

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
      const size_t j = idx[0];
      float pj = d_p[j], mj = d_m[j], vj = d_v[j];
      const float gj = d_g[j];
      for (int t = 1; t <= time_step; t++) {
        float scaled_grad = gj / grad_scale;
        mj = beta1 * mj + (1.f - beta1) * scaled_grad;
        vj = beta2 * vj + (1.f - beta2) * scaled_grad * scaled_grad;
        float m_corrected = mj / d_bc1[t];
        float v_corrected = vj / d_bc2[t];
        float denom;
        if (mode == ADAM_MODE_0)
          denom = sycl::sqrt(v_corrected + eps);
        else
          denom = sycl::sqrt(v_corrected) + eps;
        float update = (m_corrected / denom) + (decay * pj);
        pj -= (step_size * update);
      }
      d_p[j] = pj;
      d_m[j] = mj;
      d_v[j] = vj;
    });
  }
  q.wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", time * 1e-6f / repeat);

  q.memcpy(p.data(), d_p, size_bytes).wait();
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
