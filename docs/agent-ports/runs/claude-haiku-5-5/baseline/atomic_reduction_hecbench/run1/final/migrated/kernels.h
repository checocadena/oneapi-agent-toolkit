#pragma once
#include <sycl/sycl.hpp>
#include <cstddef>

// Grid-stride reduction: each work-item sums V consecutive elements per step
// (V = 1, 2, 4, 8, 16) and then issues a single atomic add to *out.
// arrayLength is a multiple of V, so in[i + V - 1] is always in bounds.
template <int V>
struct atomic_reduction_kernel {
  const int* in;
  int* out;
  std::size_t n;

  void operator()(sycl::nd_item<1> it) const {
    std::size_t stride = it.get_global_range(0) * V;
    int sum = 0;
    for (std::size_t i = it.get_global_id(0) * V; i < n; i += stride) {
      for (int j = 0; j < V; ++j) sum += in[i + j];
    }
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(*out);
    ref.fetch_add(sum);
  }
};
