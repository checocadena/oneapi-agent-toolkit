// Hand-written idiomatic SYCL 2020 port of adam.cu (the target end state; no dpct helpers).
// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/adam-cuda/main.cu,
// commit f0c817aa). Copyright (c) 2020 Argonne National Laboratory, 2020- Oak Ridge National
// Laboratory (Zheming Jin). BSD 3-Clause License; see LICENSE.HeCBench in this directory.
//
// Port notes: the grid-stride kernel maps 1:1 onto an nd_range of the same size; powf and
// sqrtf become sycl::pow and sycl::sqrt. The CUDA default stream is in-order, so the queue is
// too.
// Usage: adam <out.bin> [vector size=1048576] [time steps=10] [repeat=1]
//        (the output depends on time steps and repeat)
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../adam_input.hpp"

enum class AdamMode { kEpsUnderSqrt = 0, kEpsOutsideSqrt = 1 };

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::printf("Usage: %s <out.bin> [vector size] [number of time steps] [repeat]\n", argv[0]);
    return 1;
  }
  const int vector_size = argc > 2 ? std::atoi(argv[2]) : 1048576;
  const int time_step = argc > 3 ? std::atoi(argv[3]) : 10;
  const int repeat = argc > 4 ? std::atoi(argv[4]) : 1;

  std::vector<float> m(vector_size), v(vector_size), g(vector_size), p(vector_size);
  fill_input(m.data(), v.data(), g.data(), p.data(), vector_size);

  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float b1 = 0.9f;
  const float b2 = 0.999f;
  const float eps = 1e-8f;
  const float grad_scale = 256.f;
  const AdamMode mode = AdamMode::kEpsUnderSqrt;
  constexpr size_t kBlockSize = 256;

  try {
    sycl::queue q{sycl::property::queue::in_order()};
    float* d_m = sycl::malloc_device<float>(vector_size, q);
    float* d_v = sycl::malloc_device<float>(vector_size, q);
    float* d_g = sycl::malloc_device<float>(vector_size, q);
    float* d_p = sycl::malloc_device<float>(vector_size, q);
    q.memcpy(d_m, m.data(), sizeof(float) * vector_size);
    q.memcpy(d_v, v.data(), sizeof(float) * vector_size);
    q.memcpy(d_g, g.data(), sizeof(float) * vector_size);
    q.memcpy(d_p, p.data(), sizeof(float) * vector_size).wait();

    const size_t global = (size_t(vector_size) + kBlockSize - 1) / kBlockSize * kBlockSize;
    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) {
      q.parallel_for(sycl::nd_range<1>{global, kBlockSize}, [=](sycl::nd_item<1> it) {
        const size_t total = it.get_global_range(0);
        for (size_t j = it.get_global_id(0); j < size_t(vector_size); j += total) {
          for (int t = 1; t <= time_step; t++) {
            const float scaled_grad = d_g[j] / grad_scale;
            d_m[j] = b1 * d_m[j] + (1.f - b1) * scaled_grad;
            d_v[j] = b2 * d_v[j] + (1.f - b2) * scaled_grad * scaled_grad;
            const float m_corrected = d_m[j] / (1.f - sycl::pow(b1, float(t)));
            const float v_corrected = d_v[j] / (1.f - sycl::pow(b2, float(t)));
            const float denom = mode == AdamMode::kEpsUnderSqrt
                                    ? sycl::sqrt(v_corrected + eps)
                                    : sycl::sqrt(v_corrected) + eps;
            const float update = m_corrected / denom + decay * d_p[j];
            d_p[j] -= step_size * update;
          }
        }
      });
    }
    q.wait_and_throw();
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("kernel_ms=%f\n", (ns * 1e-6) / repeat);

    q.memcpy(p.data(), d_p, sizeof(float) * vector_size).wait();
    sycl::free(d_m, q);
    sycl::free(d_v, q);
    sycl::free(d_g, q);
    sycl::free(d_p, q);
  } catch (const sycl::exception& ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    return 1;
  }
  FILE* f = std::fopen(argv[1], "wb");
  if (!f || std::fwrite(p.data(), sizeof(float), p.size(), f) != p.size()) return 1;
  std::fclose(f);
  return 0;
}
