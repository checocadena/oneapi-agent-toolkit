// CUDA version of the 64-tap FIR (the code to be ported).
// Coefficients live in __constant__ memory; each thread computes one output sample of
// one channel; the input tile + history is staged through shared memory.
// Usage: fir <in.wav> <out.wav> [cutoff_hz=2000]
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdlib>

#include "fir_coeffs.hpp"
#include "wav.hpp"

#define BLOCK 256

/*
DPCT1009:1: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    dpct::err0 err_ = (call);                                                  \
                                                                               \
  } while (0)

inline dpct::constant_memory<float, 1> c_taps(kTaps);

// x and y are one de-interleaved channel of length n.
void fir_kernel(const float* __restrict__ x, float* __restrict__ y, int n,
                float const *c_taps, float *tile) {
  auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

  const int base = item_ct1.get_group(2) * item_ct1.get_local_range(2);
  const int t = item_ct1.get_local_id(2);
  // Load history (kTaps-1 samples before the block) and the block itself.
  for (int i = t; i < BLOCK + kTaps - 1; i += item_ct1.get_local_range(2)) {
    const int src = base - (kTaps - 1) + i;
    tile[i] = (src >= 0 && src < n) ? x[src] : 0.0f;
  }
  /*
  DPCT1113:0: Consider replacing
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) with
  sycl::nd_item::barrier() if function "fir_kernel" is called in a
  multidimensional kernel.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);
  const int idx = base + t;
  if (idx >= n) return;
  float acc = 0.0f;
  for (int k = 0; k < kTaps; ++k) acc += c_taps[k] * tile[t + (kTaps - 1) - k];
  y[idx] = acc;
}

int main(int argc, char **argv) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s in.wav out.wav [cutoff_hz]\n", argv[0]);
    return 2;
  }
  const double cutoff = argc > 3 ? std::strtod(argv[3], nullptr) : 2000.0;
  WavData wav;
  if (!wav_read(argv[1], wav)) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  float h[kTaps];
  make_lowpass(h, cutoff, wav.sample_rate);
  CUDA_CHECK(
      DPCT_CHECK_ERROR(q_ct1.memcpy(c_taps.get_ptr(), h, sizeof(h)).wait()));

  const int ch = wav.channels;
  const int n = static_cast<int>(wav.frames());
  std::vector<float> chan(n), result(n), out(wav.samples.size());
  float *d_x = nullptr, *d_y = nullptr;
  CUDA_CHECK(DPCT_CHECK_ERROR(d_x = sycl::malloc_device<float>(n, q_ct1)));
  CUDA_CHECK(DPCT_CHECK_ERROR(d_y = sycl::malloc_device<float>(n, q_ct1)));

  for (int c = 0; c < ch; ++c) {
    for (int i = 0; i < n; ++i) chan[i] = wav.samples[i * ch + c];
    CUDA_CHECK(DPCT_CHECK_ERROR(
        q_ct1.memcpy(d_x, chan.data(), n * sizeof(float)).wait()));
    {
      c_taps.init();

      q_ct1.submit([&](sycl::handler &cgh) {
        auto c_taps_ptr_ct1 = c_taps.get_ptr();

        /*
        DPCT1101:3: 'BLOCK + kTaps - 1' expression was replaced with a value.
        Modify the code to use the original expression, provided in comments,
        if it is correct.
        */
        sycl::local_accessor<float, 1> tile_acc_ct1(
            sycl::range<1>(319 /*BLOCK + kTaps - 1*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, (n + BLOCK - 1) / BLOCK) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
              fir_kernel(
                  d_x, d_y, n, c_taps_ptr_ct1,
                  tile_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                      .get());
            });
      });
    }
    /*
    DPCT1010:2: SYCL uses exceptions to report errors and does not use the error
    codes. The cudaGetLastError function call was replaced with 0. You need to
    rewrite this code.
    */
    CUDA_CHECK(0);
    CUDA_CHECK(DPCT_CHECK_ERROR(
        q_ct1.memcpy(result.data(), d_y, n * sizeof(float)).wait()));
    for (int i = 0; i < n; ++i) out[i * ch + c] = result[i];
  }
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(d_x, q_ct1)));
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(d_y, q_ct1)));
  if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate)) return 1;
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
