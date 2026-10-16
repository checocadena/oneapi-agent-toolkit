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

  sycl::queue q{sycl::property::queue::in_order()};

  float* d_m = sycl::malloc_device<float>(vector_size, q);
  float* d_v = sycl::malloc_device<float>(vector_size, q);
  float* d_g = sycl::malloc_device<float>(vector_size, q);
  float* d_p = sycl::malloc_device<float>(vector_size, q);
  q.copy(m.data(), d_m, vector_size);
  q.copy(v.data(), d_v, vector_size);
  q.copy(g.data(), d_g, vector_size);
  q.copy(p.data(), d_p, vector_size).wait();

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float b1 = 0.9f;
  const float b2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;
  const adamMode_t mode = ADAM_MODE_0;

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    q.parallel_for(sycl::range<1>(vector_size), [=](sycl::id<1> id) {
      const size_t j = id[0];
      for (int t = 1; t <= time_step; t++) {
        float scaled_grad = d_g[j] / grad_scale;
        d_m[j] = b1 * d_m[j] + (1.f - b1) * scaled_grad;
        d_v[j] = b2 * d_v[j] + (1.f - b2) * scaled_grad * scaled_grad;
        float m_corrected = d_m[j] / (1.f - sycl::pow(b1, static_cast<float>(t)));
        float v_corrected = d_v[j] / (1.f - sycl::pow(b2, static_cast<float>(t)));
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
  q.wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", time * 1e-6f / repeat);

  q.copy(d_p, p.data(), vector_size).wait();
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(p.data(), sizeof(float), p.size(), f) != p.size()) return 1;
  fclose(f);
  sycl::free(d_p, q);
  sycl::free(d_m, q);
  sycl::free(d_v, q);
  sycl::free(d_g, q);
  return 0;
}
