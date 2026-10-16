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
#include <dpct/dpct.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "adam_input.hpp"
#include "reference.h"

template <typename T, typename G>

void adam (
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
  const float decay)
{
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  const size_t i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
  const size_t totThreads =
      item_ct1.get_group_range(2) * item_ct1.get_local_range(2);

  for (size_t j = i; j < vector_size; j += totThreads) {
    for (int t = 1; t <= time_step; t++) {
      T scaled_grad = g[j]/grad_scale;
      m[j] = b1*m[j] + (1.f-b1)*scaled_grad;
      v[j] = b2*v[j] + (1.f-b2)*scaled_grad*scaled_grad;
      float m_corrected = m[j] / (1.f - dpct::pow(b1, t));
      float v_corrected = v[j] / (1.f - dpct::pow(b2, t));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sycl::sqrt(v_corrected + eps);
      else // Mode 1
        denom = sycl::sqrt(v_corrected) + eps;
      float update = (m_corrected/denom) + (decay*p[j]);
      p[j] -= (step_size*update);
    }
  }
}

int main(int argc, char* argv[])
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (argc < 2) {
    printf("Usage: %s <out.bin> [vector size] [number of time steps] [repeat]\n", argv[0]);
    return 1;
  }
  const int vector_size = argc > 2 ? atoi(argv[2]) : 1048576;
  const int time_step = argc > 3 ? atoi(argv[3]) : 10;
  const int repeat = argc > 4 ? atoi(argv[4]) : 1;

  size_t size_bytes = vector_size * sizeof(float);
  std::vector<float> m(vector_size), v(vector_size), g(vector_size), p(vector_size);
  fill_input(m.data(), v.data(), g.data(), p.data(), vector_size);

  float *d_m, *d_v, *d_g, *d_p;
  d_m = (float *)sycl::malloc_device(size_bytes, q_ct1);
  q_ct1.memcpy(d_m, m.data(), size_bytes).wait();
  d_v = (float *)sycl::malloc_device(size_bytes, q_ct1);
  q_ct1.memcpy(d_v, v.data(), size_bytes).wait();
  d_g = (float *)sycl::malloc_device(size_bytes, q_ct1);
  q_ct1.memcpy(d_g, g.data(), size_bytes).wait();
  d_p = (float *)sycl::malloc_device(size_bytes, q_ct1);
  q_ct1.memcpy(d_p, p.data(), size_bytes).wait();

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;

  const int threadsPerBlock = 256;
  const dpct::dim3 grids((vector_size + threadsPerBlock - 1) / threadsPerBlock);
  const dpct::dim3 blocks(threadsPerBlock);

  adamMode_t mode = ADAM_MODE_0;

  dev_ct1.queues_wait_and_throw();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q_ct1.parallel_for(sycl::nd_range<3>(grids * blocks, blocks),
                       [=](sycl::nd_item<3> item_ct1) {
                         adam<float, float>(
                             d_p, d_m, d_v, d_g, beta1, beta2, eps, grad_scale,
                             step_size, time_step, vector_size, mode, decay);
                       });
  }
  dev_ct1.queues_wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", time * 1e-6f / repeat);

  q_ct1.memcpy(p.data(), d_p, size_bytes).wait();
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(p.data(), sizeof(float), p.size(), f) != p.size()) return 1;
  fclose(f);
  dpct::dpct_free(d_p, q_ct1);
  dpct::dpct_free(d_m, q_ct1);
  dpct::dpct_free(d_v, q_ct1);
  dpct::dpct_free(d_g, q_ct1);
  return 0;
}
