#pragma once
#include <sycl/sycl.hpp>

// Grid-stride reduction; each thread loads VEC consecutive ints per iteration and
// finishes with one atomic add into *out.
template <int VEC>
struct atomic_reduction_kernel {
  const int* in;
  int* out;
  int arrayLength;

  void operator()(sycl::nd_item<1> item) const {
    int sum = 0;
    const int stride = static_cast<int>(item.get_global_range(0)) * VEC;
    for (int i = static_cast<int>(item.get_global_id(0)) * VEC; i < arrayLength; i += stride) {
#pragma unroll
      for (int j = 0; j < VEC; j++) sum += in[i + j];
    }
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(*out);
    ref.fetch_add(sum);
  }
};
