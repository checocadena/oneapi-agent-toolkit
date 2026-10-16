// SYCL 2020 softmax: two kernels ported from HeCBench softmax-cuda.
// Usage: softmax <out.bin> [numSlice=8192] [sliceSize=256] [kernel=1] [repeat=1]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "softmax_input.hpp"

constexpr int BLOCK_SIZE = 256;

int main(int argc, char *argv[]) {
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
  float *d_input = sycl::malloc_device<float>(numElem, q);
  float *d_output = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input.data(), sizeof(float) * numElem).wait();

  auto start = std::chrono::steady_clock::now();
  if (kernel == 1) {
    // One sub-group per slice; sub-groups stride over slices, so any
    // sub-group size works.
    const size_t groups = (numSlice + 7) / 8;
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(
          sycl::nd_range<1>(groups * BLOCK_SIZE, BLOCK_SIZE),
          [=](sycl::nd_item<1> it) {
            sycl::sub_group sg = it.get_sub_group();
            const int lane = sg.get_local_linear_id();
            const int width = sg.get_local_linear_range();
            const int sgPerGroup = sg.get_group_linear_range();
            const int totalSg = it.get_group_range(0) * sgPerGroup;
            for (int i = it.get_group(0) * sgPerGroup + sg.get_group_linear_id();
                 i < numSlice; i += totalSg) {
              const float *s = d_input + (size_t)i * sliceSize;
              float *d = d_output + (size_t)i * sliceSize;
              float m = s[0];
              for (int j = lane; j < sliceSize; j += width)
                m = sycl::max(m, s[j]);
              m = sycl::reduce_over_group(sg, m, sycl::maximum<float>{});
              float sum = 0;
              for (int j = lane; j < sliceSize; j += width)
                sum += sycl::exp(s[j] - m);
              sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>{});
              for (int j = lane; j < sliceSize; j += width)
                d[j] = sycl::exp(s[j] - m) / sum;
            }
          });
    }
  } else {
    const size_t global = ((numSlice + BLOCK_SIZE - 1) / BLOCK_SIZE) * (size_t)BLOCK_SIZE;
    for (int n = 0; n < repeat; n++) {
      q.parallel_for(sycl::nd_range<1>(global, BLOCK_SIZE),
                     [=](sycl::nd_item<1> it) {
        int i = it.get_global_id(0);
        if (i >= numSlice) return;
        const float *s = d_input + (size_t)i * sliceSize;
        float *d = d_output + (size_t)i * sliceSize;
        float m = s[0];
        for (int j = 0; j < sliceSize; j++) m = sycl::max(m, s[j]);
        float sum = 0;
        for (int j = 0; j < sliceSize; j++) sum += sycl::exp(s[j] - m);
        for (int j = 0; j < sliceSize; j++) d[j] = sycl::exp(s[j] - m) / sum;
      });
    }
  }
  q.wait_and_throw();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("kernel_ms=%f\n", (time * 1e-6f) / repeat);

  q.memcpy(output_gpu.data(), d_output, sizeof(float) * numElem).wait();
  FILE *f = fopen(argv[1], "wb");
  if (!f || fwrite(output_gpu.data(), sizeof(float), numElem, f) != size_t(numElem)) return 1;
  fclose(f);
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  return 0;
}
