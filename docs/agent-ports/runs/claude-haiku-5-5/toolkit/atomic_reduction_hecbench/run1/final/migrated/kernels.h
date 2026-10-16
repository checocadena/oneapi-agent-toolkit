#include <sycl/sycl.hpp>

// Grid-stride reduction with V loads per iteration; each work-item finishes with
// one device-scope atomic add into *out.
template <int V>
inline void atomic_reduction_kernel(sycl::nd_item<1> item, const int *in, int *out,
                                    int arrayLength) {
  int sum = 0;
  int idx = static_cast<int>(item.get_global_id(0));
  int stride = static_cast<int>(item.get_global_range(0));
  for (int i = idx * V; i < arrayLength; i += stride * V) {
    for (int j = 0; j < V; j++) {
      sum += in[i + j];
    }
  }
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(out[0])
      .fetch_add(sum);
}
