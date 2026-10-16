// SYCL 2020 softmax: port of the two kernels of HeCBench softmax-cuda.
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
                    const float* src, float* dest) {
  int i = it.get_global_id(0);
  if (i >= numSlice) return;
  const float* s = src + size_t(i) * sliceSize;
  float* d = dest + size_t(i) * sliceSize;
  float max_ = s[0];
  for (int j = 0; j < sliceSize; j++) max_ = sycl::max(max_, s[j]);
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) sum += sycl::exp(s[j] - max_);
  for (int j = 0; j < sliceSize; j++) d[j] = sycl::exp(s[j] - max_) / sum;
}

// One sub-group per slice; works for any sub-group size.
static void softMax2(sycl::nd_item<1> it, const int numSlice, const int sliceSize,
                     const float* src, float* dest) {
  sycl::sub_group sg = it.get_sub_group();
  int i = int(it.get_group(0) * sg.get_group_range()[0] + sg.get_group_id()[0]);
  if (i >= numSlice) return;  // uniform across the sub-group
  const int lane = int(sg.get_local_id()[0]);
  const int width = int(sg.get_local_range()[0]);
  const float* s = src + size_t(i) * sliceSize;
  float* d = dest + size_t(i) * sliceSize;
  float max_ = s[0];
  for (int j = lane; j < sliceSize; j += width) max_ = sycl::max(max_, s[j]);
  max_ = sycl::reduce_over_group(sg, max_, sycl::maximum<float>{});
  float sum = 0;
  for (int j = lane; j < sliceSize; j += width) sum += sycl::exp(s[j] - max_);
  sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});
  for (int j = lane; j < sliceSize; j += width) d[j] = sycl::exp(s[j] - max_) / sum;
}

int main(int argc, char* argv[]) {
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

  sycl::queue q{sycl::property::queue::in_order()};
  float* d_input = sycl::malloc_device<float>(numElem, q);
  float* d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  // A work-group holds BLOCK_SIZE / sub_group_size slices (one per sub-group). The runtime
  // picks the sub-group size, so size the grid for the largest supported one (fewest slices
  // per group); surplus sub-groups exit immediately.
  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  size_t max_sg = sg_sizes.empty() ? 1 : size_t(*std::max_element(sg_sizes.begin(), sg_sizes.end()));
  size_t slices_per_group = std::max<size_t>(1, BLOCK_SIZE / max_sg);

  q.wait();
  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    size_t groups = (numSlice + slices_per_group - 1) / slices_per_group;
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
