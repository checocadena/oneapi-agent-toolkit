// Grid-stride reduction kernels (SYCL 2020). Each work-item accumulates V consecutive
// loads per iteration and finishes with one atomic add into *out.
// V = 1, 2, 4, 8, 16 corresponds to atomic_reduction, _v2, _v4, _v8, _v16 of the CUDA original.
#pragma once

#include <sycl/sycl.hpp>

template <int V>
struct atomic_reduction_kernel {
  const int* in;
  int* out;
  int n;

  void operator()(sycl::nd_item<1> item) const {
    const size_t stride = item.get_global_range(0) * V;
    size_t i = item.get_global_id(0) * V;
    int sum = 0;
    for (; i < static_cast<size_t>(n); i += stride) {
      for (int j = 0; j < V; ++j) sum += in[i + j];
    }
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(*out);
    ref.fetch_add(sum);
  }
};
