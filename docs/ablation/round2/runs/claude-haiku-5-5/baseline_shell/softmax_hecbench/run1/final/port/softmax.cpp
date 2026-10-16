// SYCL 2020 softmax: port of the two kernels of HeCBench softmax-cuda.
// Usage: softmax <out.bin> [numSlice=8192] [sliceSize=256] [kernel=1] [repeat=1]
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "softmax_input.hpp"

constexpr int BLOCK_SIZE = 256;

// One work-item per slice.
void softmax_per_item(sycl::nd_item<1> it, int numSlice, int sliceSize,
                      const float* src, float* dest) {
  int i = static_cast<int>(it.get_global_id(0));
  if (i >= numSlice) return;
  const float* in = src + static_cast<size_t>(i) * sliceSize;
  float* out = dest + static_cast<size_t>(i) * sliceSize;
  float max_ = in[0];
  for (int j = 0; j < sliceSize; j++) max_ = sycl::max(max_, in[j]);
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) sum += sycl::exp(in[j] - max_);
  for (int j = 0; j < sliceSize; j++) out[j] = sycl::exp(in[j] - max_) / sum;
}

// One sub-group per slice; works for any sub-group size. Sub-groups grid-stride
// over the slices so the launch does not need to know the sub-group size.
void softmax_per_subgroup(sycl::nd_item<1> it, int numSlice, int sliceSize,
                          const float* src, float* dest) {
  sycl::sub_group sg = it.get_sub_group();
  const size_t sgs_per_wg = sg.get_group_range().size();
  const size_t total_sgs = it.get_group_range(0) * sgs_per_wg;
  const size_t first = it.get_group(0) * sgs_per_wg + sg.get_group_linear_id();
  const size_t lid = sg.get_local_linear_id();
  const size_t lrange = sg.get_local_linear_range();

  for (size_t i = first; i < static_cast<size_t>(numSlice); i += total_sgs) {
    const float* in = src + i * sliceSize;
    float* out = dest + i * sliceSize;

    float max_ = in[0];
    for (size_t j = lid; j < static_cast<size_t>(sliceSize); j += lrange)
      max_ = sycl::max(max_, in[j]);
    max_ = sycl::reduce_over_group(sg, max_, sycl::maximum<float>{});

    float sum = 0;
    for (size_t j = lid; j < static_cast<size_t>(sliceSize); j += lrange)
      sum += sycl::exp(in[j] - max_);
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});

    for (size_t j = lid; j < static_cast<size_t>(sliceSize); j += lrange)
      out[j] = sycl::exp(in[j] - max_) / sum;
  }
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
  size_t numElem = static_cast<size_t>(numSlice) * sliceSize;

  std::vector<float> input(numElem), output_gpu(numElem);
  fill_input(input.data(), static_cast<int>(numElem));

  sycl::queue q{sycl::property::queue::in_order{}};
  float* d_input = sycl::malloc_device<float>(numElem, q);
  float* d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  q.wait();
  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    // Work-group of BLOCK_SIZE items = BLOCK_SIZE/sg sub-groups; enough groups
    // so that every sub-group of the grid can take a slice (grid-stride covers rest).
    size_t groups = (static_cast<size_t>(numSlice) + 7) / 8;
    sycl::nd_range<1> range{sycl::range<1>(groups * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)};
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) {
        softmax_per_subgroup(it, numSlice, sliceSize, d_input, d_output);
      });
    }
  } else {
    size_t groups = (static_cast<size_t>(numSlice) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sycl::nd_range<1> range{sycl::range<1>(groups * BLOCK_SIZE), sycl::range<1>(BLOCK_SIZE)};
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(range, [=](sycl::nd_item<1> it) {
        softmax_per_item(it, numSlice, sliceSize, d_input, d_output);
      });
    }
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(output_gpu.data(), d_output, sizeof(float) * numElem).wait();
  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(output_gpu.data(), sizeof(float), numElem, f) != numElem) return 1;
  fclose(f);
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  return 0;
}
