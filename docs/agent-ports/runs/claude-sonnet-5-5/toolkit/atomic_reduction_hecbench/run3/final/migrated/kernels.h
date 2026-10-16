#pragma once
#include <sycl/sycl.hpp>

// Grid-stride sum with V consecutive loads per iteration, one atomic add per work-item.
template <int V>
struct AtomicReduction {
  const int* in;
  int* out;
  int arrayLength;

  void operator()(sycl::nd_item<1> item) const {
    int sum = 0;
    const int stride = static_cast<int>(item.get_global_range(0)) * V;
    for (int i = static_cast<int>(item.get_global_id(0)) * V; i < arrayLength; i += stride) {
#pragma unroll
      for (int j = 0; j < V; j++) sum += in[i + j];
    }
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(*out);
    ref.fetch_add(sum);
  }
};
