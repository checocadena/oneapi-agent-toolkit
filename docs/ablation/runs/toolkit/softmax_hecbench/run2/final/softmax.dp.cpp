// SYCL softmax: the two kernels of HeCBench softmax-cuda, ported to SYCL 2020.
// Usage: softmax <out.bin> [numSlice=8192] [sliceSize=256] [kernel=1] [repeat=1]
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "softmax_input.hpp"

constexpr int BLOCK_SIZE = 256;

// One work-item per slice.
static void softMax(sycl::nd_item<1> it, const int numSlice, const int sliceSize,
                    const float* src, float* dest)
{
  const int i = static_cast<int>(it.get_global_id(0));
  if (i >= numSlice) return;
  const float* in = src + static_cast<size_t>(i) * sliceSize;
  float* out = dest + static_cast<size_t>(i) * sliceSize;
  float max_ = in[0];
  for (int j = 0; j < sliceSize; j++) max_ = sycl::max(max_, in[j]);
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) sum += sycl::exp(in[j] - max_);
  for (int j = 0; j < sliceSize; j++) out[j] = sycl::exp(in[j] - max_) / sum;
}

// One sub-group per slice; works for any sub-group size.
static void softMax2(sycl::nd_item<1> it, const int numSlice, const int sliceSize,
                     const float* src, float* dest)
{
  sycl::sub_group sg = it.get_sub_group();
  const int sgSize = static_cast<int>(sg.get_local_range()[0]);
  const int lane = static_cast<int>(sg.get_local_id()[0]);
  const int i = static_cast<int>(it.get_group(0)) *
                    static_cast<int>(sg.get_group_range()[0]) +
                static_cast<int>(sg.get_group_id()[0]);
  if (i >= numSlice) return;  // uniform across the sub-group
  const float* in = src + static_cast<size_t>(i) * sliceSize;
  float* out = dest + static_cast<size_t>(i) * sliceSize;
  float max_ = in[0];
  for (int j = lane; j < sliceSize; j += sgSize) max_ = sycl::max(max_, in[j]);
  max_ = sycl::reduce_over_group(sg, max_, sycl::maximum<float>{});
  float sum = 0;
  for (int j = lane; j < sliceSize; j += sgSize) sum += sycl::exp(in[j] - max_);
  sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});
  for (int j = lane; j < sliceSize; j += sgSize) out[j] = sycl::exp(in[j] - max_) / sum;
}

int main(int argc, char *argv[]) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (argc < 2) {
    printf("Usage: %s <out.bin> [number of slices] [slice size] [kernel 0|1] [repeat]\n", argv[0]);
    return 1;
  }
  int numSlice = argc > 2 ? atoi(argv[2]) : 8192;
  int sliceSize = argc > 3 ? atoi(argv[3]) : 256;
  int kernel = argc > 4 ? atoi(argv[4]) : 1;
  int repeat = argc > 5 ? atoi(argv[5]) : 1;
  int numElem = numSlice * sliceSize;

  std::vector<float> input(numElem), output_gpu(numElem);
  fill_input(input.data(), numElem);

  float* d_input = sycl::malloc_device<float>(numElem, q);
  float* d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  // The largest supported sub-group size gives the fewest sub-groups per work-group,
  // so it bounds the number of work-groups needed (one sub-group per slice).
  size_t maxSg = 1;
  for (size_t s : q.get_device().get_info<sycl::info::device::sub_group_sizes>())
    maxSg = std::max(maxSg, s);

  q.wait();
  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    size_t subGroupsPerGroup = std::max<size_t>(1, BLOCK_SIZE / maxSg);
    size_t groups = (numSlice + subGroupsPerGroup - 1) / subGroupsPerGroup;
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(sycl::nd_range<1>(groups * BLOCK_SIZE, BLOCK_SIZE),
                     [=](sycl::nd_item<1> it) {
                       softMax2(it, numSlice, sliceSize, d_input, d_output);
                     });
    }
  } else {
    size_t groups = (numSlice + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(sycl::nd_range<1>(groups * BLOCK_SIZE, BLOCK_SIZE),
                     [=](sycl::nd_item<1> it) {
                       softMax(it, numSlice, sliceSize, d_input, d_output);
                     });
    }
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(output_gpu.data(), d_output, sizeof(float) * numElem).wait();
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(output_gpu.data(), sizeof(float), numElem, f) != size_t(numElem)) return 1;
  fclose(f);
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  return 0;
}
