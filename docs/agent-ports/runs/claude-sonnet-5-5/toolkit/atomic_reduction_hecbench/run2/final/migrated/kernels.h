#pragma once
#include <sycl/sycl.hpp>

// Grid-stride sum where each thread loads V consecutive ints per iteration and
// finishes with one atomic add into *out.
template <int V>
struct AtomicReduction {
  const int *in;
  int *out;
  int arrayLength;

  void operator()(sycl::nd_item<1> item) const {
    int sum = 0;
    const int stride = static_cast<int>(item.get_global_range(0)) * V;
    for (int i = static_cast<int>(item.get_global_id(0)) * V; i < arrayLength; i += stride) {
      #pragma unroll
      for (int j = 0; j < V; j++) sum += in[i + j];
    }
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*out)
        .fetch_add(sum);
  }
};
