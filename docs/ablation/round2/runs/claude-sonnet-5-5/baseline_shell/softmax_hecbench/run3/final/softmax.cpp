// SYCL 2020 softmax: port of the two kernels of HeCBench softmax-cuda.
// Usage: softmax <out.bin> [numSlice=8192] [sliceSize=256] [kernel=1] [repeat=1]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "softmax_input.hpp"

constexpr size_t BLOCK_SIZE = 256;

// One work-item per slice.
struct SoftMaxKernel {
  int numSlice, sliceSize;
  const float* src;
  float* dest;
  void operator()(sycl::nd_item<1> it) const {
    int i = static_cast<int>(it.get_global_id(0));
    if (i >= numSlice) return;
    const float* in = src + size_t(i) * sliceSize;
    float* out = dest + size_t(i) * sliceSize;
    float max_ = in[0];
    for (int j = 0; j < sliceSize; j++) max_ = sycl::max(max_, in[j]);
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) sum += sycl::exp(in[j] - max_);
    for (int j = 0; j < sliceSize; j++) out[j] = sycl::exp(in[j] - max_) / sum;
  }
};

// One sub-group per slice; works for any sub-group size.
struct SoftMax2Kernel {
  int numSlice, sliceSize;
  const float* src;
  float* dest;
  void operator()(sycl::nd_item<1> it) const {
    sycl::sub_group sg = it.get_sub_group();
    int i = static_cast<int>(it.get_group(0) * sg.get_group_linear_range() +
                             sg.get_group_linear_id());
    if (i >= numSlice) return;  // uniform across the sub-group
    const float* in = src + size_t(i) * sliceSize;
    float* out = dest + size_t(i) * sliceSize;
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const int width = static_cast<int>(sg.get_local_linear_range());
    float max_ = in[0];
    for (int j = lane; j < sliceSize; j += width) max_ = sycl::max(max_, in[j]);
    max_ = sycl::reduce_over_group(sg, max_, sycl::maximum<float>{});
    float sum = 0;
    for (int j = lane; j < sliceSize; j += width) sum += sycl::exp(in[j] - max_);
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});
    for (int j = lane; j < sliceSize; j += width) out[j] = sycl::exp(in[j] - max_) / sum;
  }
};

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <out.bin> [number of slices] [slice size] [kernel 0|1] [repeat]\n", argv[0]);
    return 1;
  }
  int numSlice = argc > 2 ? atoi(argv[2]) : 8192;
  int sliceSize = argc > 3 ? atoi(argv[3]) : 256;
  int kernel = argc > 4 ? atoi(argv[4]) : 1;
  int repeat = argc > 5 ? atoi(argv[5]) : 1;
  size_t numElem = size_t(numSlice) * sliceSize;

  std::vector<float> input(numElem), output_gpu(numElem);
  fill_input(input.data(), numElem);

  sycl::queue q{sycl::property::queue::in_order()};
  float* d_input = sycl::malloc_device<float>(numElem, q);
  float* d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  // Sub-group size the compiler/runtime will use for the sub-group kernel.
  size_t sgSize = 32;
  if (kernel == 1) {
    auto ctx = q.get_context();
    auto id = sycl::get_kernel_id<SoftMax2Kernel>();
    auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {q.get_device()}, {id});
    auto k = kb.get_kernel(id);
    sgSize = k.get_info<sycl::info::kernel_device_specific::max_sub_group_size>(
        q.get_device());
    if (sgSize == 0 || sgSize > BLOCK_SIZE) sgSize = 1;
  }

  q.wait();
  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    size_t slicesPerGroup = BLOCK_SIZE / sgSize;
    size_t groups = (numSlice + slicesPerGroup - 1) / slicesPerGroup;
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(sycl::nd_range<1>(groups * BLOCK_SIZE, BLOCK_SIZE),
                     SoftMax2Kernel{numSlice, sliceSize, d_input, d_output});
    }
  } else {
    size_t groups = (numSlice + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(sycl::nd_range<1>(groups * BLOCK_SIZE, BLOCK_SIZE),
                     SoftMaxKernel{numSlice, sliceSize, d_input, d_output});
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
