#pragma once
#include <sycl/sycl.hpp>

// Grid-stride reduction: each work-item sums VEC consecutive ints per iteration and
// finishes with a single atomic add into *out. VEC = 1, 2, 4, 8 or 16.
template <int VEC>
struct atomic_reduction_kernel {
  const int *in;
  int *out;
  int arrayLength;

  void operator()(sycl::nd_item<1> item) const {
    int sum = 0;
    const int stride = static_cast<int>(item.get_global_range(0)) * VEC;
    for (int i = static_cast<int>(item.get_global_id(0)) * VEC; i < arrayLength;
         i += stride) {
#pragma unroll
      for (int j = 0; j < VEC; j++) sum += in[i + j];
    }
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(*out);
    ref.fetch_add(sum);
  }
};

template <int VEC>
inline void launch_atomic_reduction(sycl::queue &q, const int *in, int *out,
                                    int arrayLength, int grid_size, int block_size) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(grid_size) * block_size),
                                   sycl::range<1>(block_size)),
                 atomic_reduction_kernel<VEC>{in, out, arrayLength});
}
