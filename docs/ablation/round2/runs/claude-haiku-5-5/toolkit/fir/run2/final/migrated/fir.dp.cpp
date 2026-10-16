// SYCL 2020 port of the 64-tap FIR.
// Taps are captured by value in the kernel (the SYCL analogue of __constant__ memory);
// each work-item computes one output sample of one channel; the input tile + history
// is staged through work-group local memory.
// Usage: fir <in.wav> <out.wav> [cutoff_hz=2000]
#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "fir_coeffs.hpp"
#include "wav.hpp"

constexpr std::size_t kBlock = 256;
constexpr std::size_t kHalo = kTaps - 1;

int main(int argc, char** argv) {
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
  std::array<float, kTaps> taps{};
  for (int k = 0; k < kTaps; ++k) taps[k] = h[k];

  const int ch = wav.channels;
  const std::size_t n = static_cast<std::size_t>(wav.frames());
  std::vector<float> out(wav.samples.size());

  try {
    sycl::queue q{sycl::property::queue::in_order{}};

    std::vector<float> chan(n), result(n);
    float* d_x = sycl::malloc_device<float>(n, q);
    float* d_y = sycl::malloc_device<float>(n, q);
    const std::size_t global = ((n + kBlock - 1) / kBlock) * kBlock;

    for (int c = 0; c < ch; ++c) {
      for (std::size_t i = 0; i < n; ++i) chan[i] = wav.samples[i * ch + c];
      q.memcpy(d_x, chan.data(), n * sizeof(float));

      q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> tile(sycl::range<1>(kBlock + kHalo), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kBlock)),
            [=](sycl::nd_item<1> it) {
              const std::size_t base = it.get_group(0) * kBlock;
              const std::size_t t = it.get_local_id(0);
              // Load history (kHalo samples before the block) and the block itself.
              for (std::size_t i = t; i < kBlock + kHalo; i += kBlock) {
                const std::ptrdiff_t src =
                    static_cast<std::ptrdiff_t>(base) - static_cast<std::ptrdiff_t>(kHalo) +
                    static_cast<std::ptrdiff_t>(i);
                tile[i] = (src >= 0 && static_cast<std::size_t>(src) < n)
                              ? d_x[src]
                              : 0.0f;
              }
              sycl::group_barrier(it.get_group());
              const std::size_t idx = base + t;
              if (idx >= n) return;
              float acc = 0.0f;
              for (std::size_t k = 0; k < kTaps; ++k) acc += taps[k] * tile[t + kHalo - k];
              d_y[idx] = acc;
            });
      });
      q.memcpy(result.data(), d_y, n * sizeof(float)).wait();
      for (std::size_t i = 0; i < n; ++i) out[i * ch + c] = result[i];
    }
    sycl::free(d_x, q);
    sycl::free(d_y, q);
  } catch (sycl::exception const& exc) {
    std::cerr << exc.what() << " at " << __FILE__ << ":" << __LINE__ << std::endl;
    return 1;
  }

  if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate)) return 1;
  return 0;
}
