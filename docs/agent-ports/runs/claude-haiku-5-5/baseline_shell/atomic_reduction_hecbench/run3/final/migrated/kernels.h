// Adapted from HeCBench (https://github.com/zjin-lcf/HeCBench, src/atomicReduction-cuda,
// commit f0c817aa). Copyright (c) 2015-2016 Advanced Micro Devices, Inc. MIT License; see
// LICENSE.AMD in this directory.
// SYCL 2020 port: grid-stride reduction, V loads per iteration, one device-scope atomic
// add per work-item.
#pragma once

#include <sycl/sycl.hpp>

template <int V>
void atomic_reduction_v(const int* in, int* out, int n, sycl::nd_item<1> it)
{
  const size_t gid = it.get_global_id(0);
  const size_t stride = it.get_global_range(0) * V;
  int sum = 0;
  for (size_t i = gid * V; i < static_cast<size_t>(n); i += stride) {
    for (int k = 0; k < V; k++) sum += in[i + k];
  }
  sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(out[0])
      .fetch_add(sum);
}
