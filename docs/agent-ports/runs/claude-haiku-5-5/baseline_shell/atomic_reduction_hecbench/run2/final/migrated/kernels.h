#pragma once
#include <sycl/sycl.hpp>

// Grid-stride integer sum; each thread loads LOADS consecutive ints per iteration and
// finishes with one atomic add into *out. Integer addition is exact and order-independent,
// so every variant yields the same sum as the CUDA original.
template <int LOADS>
void atomic_reduction_impl(sycl::nd_item<1> item, const int* in, int* out, int arrayLength) {
  const std::size_t stride = item.get_global_range(0) * LOADS;
  int sum = 0;
  for (std::size_t i = item.get_global_id(0) * LOADS; i < static_cast<std::size_t>(arrayLength);
       i += stride) {
    for (int j = 0; j < LOADS; j++) sum += in[i + j];
  }
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref(*out);
  ref.fetch_add(sum);
}
