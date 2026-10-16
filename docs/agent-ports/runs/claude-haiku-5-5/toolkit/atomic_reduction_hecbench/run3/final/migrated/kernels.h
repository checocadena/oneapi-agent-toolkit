#pragma once
#include <sycl/sycl.hpp>

// Grid-stride kernels: each work-item accumulates a private sum over its strided elements
// and finishes with a single atomic add into *out. Work-items are indexed in 1-D.

inline void atomic_reduction(const int* in, int* out, int arrayLength, sycl::nd_item<1> item) {
  int sum = 0;
  const size_t idx = item.get_global_id(0);
  const size_t stride = item.get_global_range(0);
  for (size_t i = idx; i < static_cast<size_t>(arrayLength); i += stride) {
    sum += in[i];
  }
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(*out).fetch_add(sum);
}

inline void atomic_reduction_v2(const int* in, int* out, int arrayLength, sycl::nd_item<1> item) {
  int sum = 0;
  const size_t idx = item.get_global_id(0);
  const size_t stride = item.get_global_range(0) * 2;
  for (size_t i = idx * 2; i < static_cast<size_t>(arrayLength); i += stride) {
    sum += in[i] + in[i + 1];
  }
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(*out).fetch_add(sum);
}

inline void atomic_reduction_v4(const int* in, int* out, int arrayLength, sycl::nd_item<1> item) {
  int sum = 0;
  const size_t idx = item.get_global_id(0);
  const size_t stride = item.get_global_range(0) * 4;
  for (size_t i = idx * 4; i < static_cast<size_t>(arrayLength); i += stride) {
    sum += in[i] + in[i + 1] + in[i + 2] + in[i + 3];
  }
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(*out).fetch_add(sum);
}

inline void atomic_reduction_v8(const int* in, int* out, int arrayLength, sycl::nd_item<1> item) {
  int sum = 0;
  const size_t idx = item.get_global_id(0);
  const size_t stride = item.get_global_range(0) * 8;
  for (size_t i = idx * 8; i < static_cast<size_t>(arrayLength); i += stride) {
    sum += in[i] + in[i + 1] + in[i + 2] + in[i + 3] + in[i + 4] + in[i + 5] + in[i + 6] +
           in[i + 7];
  }
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(*out).fetch_add(sum);
}

inline void atomic_reduction_v16(const int* in, int* out, int arrayLength, sycl::nd_item<1> item) {
  int sum = 0;
  const size_t idx = item.get_global_id(0);
  const size_t stride = item.get_global_range(0) * 16;
  for (size_t i = idx * 16; i < static_cast<size_t>(arrayLength); i += stride) {
    sum += in[i] + in[i + 1] + in[i + 2] + in[i + 3] + in[i + 4] + in[i + 5] + in[i + 6] +
           in[i + 7] + in[i + 8] + in[i + 9] + in[i + 10] + in[i + 11] + in[i + 12] +
           in[i + 13] + in[i + 14] + in[i + 15];
  }
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(*out).fetch_add(sum);
}
