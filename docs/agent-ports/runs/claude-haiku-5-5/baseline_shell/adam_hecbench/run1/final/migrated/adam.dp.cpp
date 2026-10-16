// CUDA Adam optimizer step (the code to be ported): HeCBench adam-cuda, a grid-stride kernel
// that applies `time_step` Adam updates to each parameter.
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/adam-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
// Adaptations: kernel unchanged; input drawn from std::mt19937(19937) with an explicit [0, 1)
// conversion (see adam_input.hpp); the updated parameters are written to a file instead of
// being checked in-process; argument order changed.
// Usage: adam <out.bin> [vector size=1048576] [time steps=10] [repeat=1]
//        (the output depends on time steps and repeat)
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "adam_input.hpp"
#include "reference.h"

template <typename T, typename G>
void adam(
    T* __restrict__ p,
    T* __restrict__ m,
    T* __restrict__ v,
    const G* __restrict__ g,
    const float b1,
    const float b2,
    const float eps,
    const float grad_scale,
    const float step_size,
    const int time_step,
    const size_t vector_size,
    adamMode_t mode,
    const float decay,
    const sycl::nd_item<1>& item)
{
  const size_t i = item.get_global_id(0);
  const size_t totThreads = item.get_global_range(0);

  for (size_t j = i; j < vector_size; j += totThreads) {
    for (int t = 1; t <= time_step; t++) {
      T scaled_grad = g[j] / grad_scale;
      m[j] = b1 * m[j] + (1.f - b1) * scaled_grad;
      v[j] = b2 * v[j] + (1.f - b2) * scaled_grad * scaled_grad;
      float m_corrected = m[j] / (1.f - sycl::pow(b1, static_cast<float>(t)));
      float v_corrected = v[j] / (1.f - sycl::pow(b2, static_cast<float>(t)));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sycl::sqrt(v_corrected + eps);
      else // Mode 1
        denom = sycl::sqrt(v_corrected) + eps;
      float update = (m_corrected / denom) + (decay * p[j]);
      p[j] -= (step_size * update);
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    printf("Usage: %s <out.bin> [vector size] [number of time steps] [repeat]\n", argv[0]);
    return 1;
  }
  const int vector_size = argc > 2 ? atoi(argv[2]) : 1048576;
  const int time_step = argc > 3 ? atoi(argv[3]) : 10;
  const int repeat = argc > 4 ? atoi(argv[4]) : 1;

  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order()};

  const size_t size_bytes = vector_size * sizeof(float);
  std::vector<float> m(vector_size), v(vector_size), g(vector_size), p(vector_size);
  fill_input(m.data(), v.data(), g.data(), p.data(), vector_size);

  float* d_m = sycl::malloc_device<float>(vector_size, q);
  float* d_v = sycl::malloc_device<float>(vector_size, q);
  float* d_g = sycl::malloc_device<float>(vector_size, q);
  float* d_p = sycl::malloc_device<float>(vector_size, q);
  q.memcpy(d_m, m.data(), size_bytes);
  q.memcpy(d_v, v.data(), size_bytes);
  q.memcpy(d_g, g.data(), size_bytes);
  q.memcpy(d_p, p.data(), size_bytes);
  q.wait();

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;

  const size_t threadsPerBlock = 256;
  const size_t numBlocks = (vector_size + threadsPerBlock - 1) / threadsPerBlock;
  const sycl::nd_range<1> range(numBlocks * threadsPerBlock, threadsPerBlock);

  adamMode_t mode = ADAM_MODE_0;

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(range, [=](sycl::nd_item<1> item) {
      adam<float, float>(d_p, d_m, d_v, d_g, beta1, beta2, eps, grad_scale,
                         step_size, time_step, vector_size, mode, decay, item);
    });
  }
  q.wait();
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
  return 0;
}
