#pragma once
#include <sycl/sycl.hpp>

// Grid-stride reduction: each work-item sums V consecutive ints per iteration and
// finishes with a single atomic add into *out. arrayLength is a multiple of 16 and
// V divides 16, so in[i + k] (k < V) is always in range.
template <int V>
inline void grid_stride_reduce(sycl::nd_item<1> it, const int* in, int* out,
                               int arrayLength) {
  const size_t stride = it.get_global_range(0) * V;
  int sum = 0;
  for (size_t i = it.get_global_id(0) * V; i < static_cast<size_t>(arrayLength);
       i += stride) {
    for (int k = 0; k < V; k++) sum += in[i + k];
  }
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(*out)
      .fetch_add(sum);
}
