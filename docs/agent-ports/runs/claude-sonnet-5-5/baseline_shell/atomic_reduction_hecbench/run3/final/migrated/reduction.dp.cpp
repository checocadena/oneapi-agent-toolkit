// SYCL atomic reduction: HeCBench atomicReduction, five grid-stride kernels (1, 2, 4, 8
// or 16 loads per iteration) that each finish with one atomic add per work-item, run for
// four work-group sizes.
// Usage: reduction <out.bin> [arrayLength=16777216, a multiple of 16] [repeat=10]
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kernels.h"
#include "reduction_input.hpp"

int main(int argc, char** argv)
{
  if (argc < 2) {
    printf("Usage: %s <out.bin> [array length, a multiple of 16] [repeat]\n", argv[0]);
    return 1;
  }
  const int arrayLength = argc > 2 ? atoi(argv[2]) : 16777216;
  const int N = argc > 3 ? atoi(argv[3]) : 10;
  if (arrayLength % 16 != 0) return 1;
  const int block_sizes[] = {128, 256, 512, 1024};

  std::vector<int> array(arrayLength);
  fill_input(array.data(), arrayLength);

  sycl::queue q{sycl::property::queue::in_order()};
  int *in = sycl::malloc_device<int>(arrayLength, q);
  int *out = sycl::malloc_device<int>(1, q);
  q.memcpy(in, array.data(), sizeof(int) * arrayLength).wait();

  std::vector<double> sums;
  double total_ns = 0;
  int launches = 0;

  auto benchmark = [&]<int VEC>(int grid_size, int block_size) {
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
      q.memset(out, 0, sizeof(int));
      if (grid_size > 0) {
        q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(size_t(grid_size) * block_size),
                              sycl::range<1>(block_size)),
            atomic_reduction_kernel<VEC>{in, out, arrayLength});
      }
    }
    q.wait_and_throw();
    auto t2 = std::chrono::steady_clock::now();
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    launches += N;
    int sum = 0;
    q.memcpy(&sum, out, sizeof(int)).wait();
    sums.push_back(sum);
  };

  for (int block_size : block_sizes) {
    int blocks = std::min((arrayLength + block_size - 1) / block_size, 2048);
    benchmark.operator()<1>(blocks, block_size);
    benchmark.operator()<2>(blocks / 2, block_size);
    benchmark.operator()<4>(blocks / 4, block_size);
    benchmark.operator()<8>(blocks / 8, block_size);
    benchmark.operator()<16>(blocks / 16, block_size);
  }
  printf("kernel_ms=%f\n", total_ns * 1e-6 / launches);

  FILE* f = fopen(argv[1], "wb");
  if (!f || fwrite(sums.data(), sizeof(double), sums.size(), f) != sums.size()) return 1;
  fclose(f);
  sycl::free(in, q);
  sycl::free(out, q);
  return 0;
}
