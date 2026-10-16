// SYCL 2020 port of the 64-tap FIR.
// Coefficients are passed to the kernel by value (replaces CUDA __constant__ memory);
// each work-item computes one output sample of one channel; the input tile + history
// is staged through work-group local memory.
// Usage: fir <in.wav> <out.wav> [cutoff_hz=2000]
#include <sycl/sycl.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "fir_coeffs.hpp"
#include "wav.hpp"

constexpr std::size_t BLOCK = 256;
constexpr std::size_t HALO = kTaps - 1;

using taps_t = std::array<float, kTaps>;

int main(int argc, char **argv) {
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

  try {
    float h[kTaps];
    make_lowpass(h, cutoff, wav.sample_rate);
    taps_t taps;
    for (std::size_t k = 0; k < kTaps; ++k) taps[k] = h[k];

    sycl::queue q{sycl::default_selector_v,
                  [](sycl::exception_list el) {
                    for (auto &e : el) std::rethrow_exception(e);
                  }};

    const int ch = wav.channels;
    const std::size_t n = static_cast<std::size_t>(wav.frames());
    std::vector<float> chan(n), result(n), out(wav.samples.size());

    float *d_x = sycl::malloc_device<float>(n, q);
    float *d_y = sycl::malloc_device<float>(n, q);
    if (!d_x || !d_y) {
      std::fprintf(stderr, "device allocation failed\n");
      return 1;
    }

    const std::size_t groups = (n + BLOCK - 1) / BLOCK;
    const sycl::nd_range<1> range{sycl::range<1>{groups * BLOCK},
                                  sycl::range<1>{BLOCK}};

    for (int c = 0; c < ch; ++c) {
      for (std::size_t i = 0; i < n; ++i) chan[i] = wav.samples[i * ch + c];
      q.memcpy(d_x, chan.data(), n * sizeof(float)).wait();

      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> tile(sycl::range<1>{BLOCK + HALO}, cgh);
        cgh.parallel_for(range, [=](sycl::nd_item<1> it) {
          const std::size_t base = it.get_group(0) * BLOCK;
          const std::size_t t = it.get_local_id(0);
          // Load history (kTaps-1 samples before the block) and the block itself.
          for (std::size_t i = t; i < BLOCK + HALO; i += BLOCK) {
            const long long src = static_cast<long long>(base) -
                                  static_cast<long long>(HALO) +
                                  static_cast<long long>(i);
            tile[i] = (src >= 0 && src < static_cast<long long>(n))
                          ? d_x[src]
                          : 0.0f;
          }
          sycl::group_barrier(it.get_group());
          const std::size_t idx = base + t;
          if (idx >= n) return;
          float acc = 0.0f;
          for (std::size_t k = 0; k < kTaps; ++k)
            acc += taps[k] * tile[t + HALO - k];
          d_y[idx] = acc;
        });
      });
      q.wait();

      q.memcpy(result.data(), d_y, n * sizeof(float)).wait();
      for (std::size_t i = 0; i < n; ++i) out[i * ch + c] = result[i];
    }

    sycl::free(d_x, q);
    sycl::free(d_y, q);
    if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate))
      return 1;
  } catch (const sycl::exception &e) {
    std::cerr << "SYCL error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
