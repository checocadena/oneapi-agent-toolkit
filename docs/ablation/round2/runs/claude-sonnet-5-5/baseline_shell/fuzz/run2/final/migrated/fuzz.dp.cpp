// CUDA version of the tanh soft-clip fuzz (the code to be ported).
// Usage: fuzz <in.wav> <out.wav> [drive=8] [gain=0.8]
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "wav.hpp"

/*
DPCT1009:0: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    dpct::err0 err_ = (call);                                                  \
                                                                               \
  } while (0)

void fuzz_kernel(const float* __restrict__ in, float* __restrict__ out, int n,
                            float drive, float norm) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2);
  if (i < n) out[i] = norm * sycl::tanh(drive * in[i]);
}

int main(int argc, char **argv) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s in.wav out.wav [drive] [gain]\n", argv[0]);
    return 2;
  }
  const float drive = argc > 3 ? std::strtof(argv[3], nullptr) : 8.0f;
  const float gain = argc > 4 ? std::strtof(argv[4], nullptr) : 0.8f;
  WavData wav;
  if (!wav_read(argv[1], wav)) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  const int n = static_cast<int>(wav.samples.size());
  const float norm = gain / std::tanh(drive);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(DPCT_CHECK_ERROR(d_in = sycl::malloc_device<float>(n, q_ct1)));
  CUDA_CHECK(DPCT_CHECK_ERROR(d_out = sycl::malloc_device<float>(n, q_ct1)));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      q_ct1.memcpy(d_in, wav.samples.data(), n * sizeof(float)).wait()));

  dpct::event_ptr start, stop;
  CUDA_CHECK(DPCT_CHECK_ERROR(start = new sycl::event()));
  CUDA_CHECK(DPCT_CHECK_ERROR(stop = new sycl::event()));
  const int block = 256;
  const int grid = (n + block - 1) / block;
  /*
  DPCT1024:1: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::sync_barrier(start)));
  q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, grid) *
                                           sycl::range<3>(1, 1, block),
                                       sycl::range<3>(1, 1, block)),
                     [=](sycl::nd_item<3> item_ct1) {
                       fuzz_kernel(d_in, d_out, n, drive, norm);
                     });
  /*
  DPCT1010:2: SYCL uses exceptions to report errors and does not use the error
  codes. The cudaGetLastError function call was replaced with 0. You need to
  rewrite this code.
  */
  CUDA_CHECK(0);
  /*
  DPCT1024:3: The original code returned the error code that was further
  consumed by the program logic. This original code was replaced with 0. You may
  need to rewrite the program logic consuming the error code.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::sync_barrier(stop)));
  CUDA_CHECK(DPCT_CHECK_ERROR(stop->wait_and_throw()));
  float ms = 0.f;
  CUDA_CHECK(DPCT_CHECK_ERROR(
      ms = (stop->get_profiling_info<
                sycl::info::event_profiling::command_end>() -
            start->get_profiling_info<
                sycl::info::event_profiling::command_start>()) /
           1000000.0f));
  std::printf("kernel_ms=%f\n", ms);

  std::vector<float> out(n);
  CUDA_CHECK(DPCT_CHECK_ERROR(
      q_ct1.memcpy(out.data(), d_out, n * sizeof(float)).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(d_in, q_ct1)));
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(d_out, q_ct1)));
  if (!wav_write_f32(argv[2], out.data(), out.size(), wav.channels, wav.sample_rate)) return 1;
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
