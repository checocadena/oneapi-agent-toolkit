// SYCL 2020 version of the 64-tap FIR.
// Taps are captured by value in the kernel (read-only, uniform across work-items);
// each work-item computes one output sample, staged through a work-group local tile.
// Usage: fir <in.wav> <out.wav> [cutoff_hz=2000]
#include <sycl/sycl.hpp>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "fir_coeffs.hpp"
#include "wav.hpp"

constexpr int BLOCK = 256;

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
  std::array<float, kTaps> taps;
  make_lowpass(taps.data(), cutoff, wav.sample_rate);

  sycl::queue q{sycl::property::queue::in_order()};

  const int ch = wav.channels;
  const int n = static_cast<int>(wav.frames());
  std::vector<float> chan(n), result(n), out(wav.samples.size());
  float *d_x = sycl::malloc_device<float>(n, q);
  float *d_y = sycl::malloc_device<float>(n, q);

  const size_t groups = (static_cast<size_t>(n) + BLOCK - 1) / BLOCK;

  for (int c = 0; c < ch; ++c) {
    for (int i = 0; i < n; ++i) chan[i] = wav.samples[i * ch + c];
    q.memcpy(d_x, chan.data(), n * sizeof(float));
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> tile(sycl::range<1>(BLOCK + kTaps - 1), cgh);
      const auto h = taps;
      cgh.parallel_for(
          sycl::nd_range<1>(groups * BLOCK, BLOCK),
          [=](sycl::nd_item<1> it) {
            const int base = static_cast<int>(it.get_group(0)) * BLOCK;
            const int t = static_cast<int>(it.get_local_id(0));
            for (int i = t; i < BLOCK + kTaps - 1; i += BLOCK) {
              const int src = base - (kTaps - 1) + i;
              tile[i] = (src >= 0 && src < n) ? d_x[src] : 0.0f;
            }
            sycl::group_barrier(it.get_group());
            const int idx = base + t;
            if (idx >= n) return;
            float acc = 0.0f;
            for (int k = 0; k < kTaps; ++k)
              acc += h[k] * tile[t + (kTaps - 1) - k];
            d_y[idx] = acc;
          });
    });
    q.memcpy(result.data(), d_y, n * sizeof(float)).wait();
    for (int i = 0; i < n; ++i) out[i * ch + c] = result[i];
  }
  sycl::free(d_x, q);
  sycl::free(d_y, q);
  if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate)) return 1;
  return 0;
} catch (sycl::exception const &e) {
  std::cerr << "SYCL exception: " << e.what() << std::endl;
  return 1;
}
