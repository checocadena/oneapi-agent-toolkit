// SYCL 2020 version of the 64-tap FIR.
// Taps are copied once to device memory; each work-item computes one output sample of
// one channel; the input tile + history is staged through work-group local memory.
// Usage: fir <in.wav> <out.wav> [cutoff_hz=2000]
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "fir_coeffs.hpp"
#include "wav.hpp"

constexpr int kBlock = 256;
constexpr int kTile = kBlock + kTaps - 1;

int main(int argc, char **argv) try {
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

  sycl::queue q{sycl::property::queue::in_order()};
  float *d_taps = sycl::malloc_device<float>(kTaps, q);
  q.memcpy(d_taps, h, sizeof(h)).wait();

  const int ch = wav.channels;
  const int n = static_cast<int>(wav.frames());
  std::vector<float> chan(n), result(n), out(wav.samples.size());
  float *d_x = sycl::malloc_device<float>(n, q);
  float *d_y = sycl::malloc_device<float>(n, q);

  const size_t groups = (static_cast<size_t>(n) + kBlock - 1) / kBlock;

  for (int c = 0; c < ch; ++c) {
    for (int i = 0; i < n; ++i) chan[i] = wav.samples[i * ch + c];
    q.memcpy(d_x, chan.data(), n * sizeof(float));
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> tile(sycl::range<1>(kTile), cgh);
      const float *x = d_x;
      float *y = d_y;
      const float *taps = d_taps;
      cgh.parallel_for(
          sycl::nd_range<1>(groups * kBlock, kBlock),
          [=](sycl::nd_item<1> it) {
            const int base = static_cast<int>(it.get_group(0)) * kBlock;
            const int t = static_cast<int>(it.get_local_id(0));
            for (int i = t; i < kTile; i += kBlock) {
              const int src = base - (kTaps - 1) + i;
              tile[i] = (src >= 0 && src < n) ? x[src] : 0.0f;
            }
            sycl::group_barrier(it.get_group());
            const int idx = base + t;
            if (idx >= n) return;
            float acc = 0.0f;
            for (int k = 0; k < kTaps; ++k)
              acc += taps[k] * tile[t + (kTaps - 1) - k];
            y[idx] = acc;
          });
    });
    q.memcpy(result.data(), d_y, n * sizeof(float)).wait();
    for (int i = 0; i < n; ++i) out[i * ch + c] = result[i];
  }
  sycl::free(d_x, q);
  sycl::free(d_y, q);
  sycl::free(d_taps, q);
  if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate)) return 1;
  return 0;
} catch (sycl::exception const &e) {
  std::cerr << "SYCL exception: " << e.what() << std::endl;
  return 1;
}
