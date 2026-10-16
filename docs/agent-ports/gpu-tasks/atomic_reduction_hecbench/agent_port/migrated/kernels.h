#pragma once
#include <sycl/sycl.hpp>

// Grid-stride reduction: each work-item sums V consecutive ints per iteration and
// finishes with a single atomic add into *out. arrayLength must be a multiple of V.
template <int V>
struct atomic_reduction_kernel {
  const int* in;
  int* out;
  int arrayLength;

  void operator()(sycl::nd_item<1> item) const {
    const int stride = static_cast<int>(item.get_global_range(0)) * V;
    int sum = 0;
    for (int i = static_cast<int>(item.get_global_id(0)) * V; i < arrayLength; i += stride) {
      for (int j = 0; j < V; j++) sum += in[i + j];
    }
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(*out);
    ref.fetch_add(sum);
  }
};

template <int V>
void atomic_reduction(sycl::queue& q, const int* in, int* out, int arrayLength,
                      int groups, int block_size) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(groups) * block_size),
                                   sycl::range<1>(block_size)),
                 atomic_reduction_kernel<V>{in, out, arrayLength});
}
