#pragma once
#include <sycl/sycl.hpp>

// Grid-stride reduction with Loads consecutive loads per iteration (1, 2, 4, 8, 16).
// Each work-item accumulates locally and finishes with one atomic add to *out.
// Integer addition wraps, so the result is order-independent and matches the CUDA sums.
template <int Loads>
void atomic_reduction(sycl::nd_item<1> item, const int* in, int* out, int arrayLength) {
  const size_t stride = item.get_global_range(0) * Loads;
  int sum = 0;
  for (size_t i = item.get_global_linear_id() * Loads; i < static_cast<size_t>(arrayLength);
       i += stride) {
    for (int j = 0; j < Loads; ++j) sum += in[i + j];
  }
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(*out)
      .fetch_add(sum);
}
