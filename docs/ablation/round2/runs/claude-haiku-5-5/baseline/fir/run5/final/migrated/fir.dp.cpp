// SYCL 2020 port of the 64-tap FIR.
// Coefficients are passed to the kernel by value (small, read-only, uniform across work-items);
// each work-item computes one output sample of one channel; the input tile + history is staged
// through work-group local memory.
// Usage: fir <in.wav> <out.wav> [cutoff_hz=2000]
#include <sycl/sycl.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fir_coeffs.hpp"
#include "wav.hpp"

constexpr std::size_t kBlock = 256;
constexpr std::size_t kTileLen = kBlock + kTaps - 1;

int main(int argc, char** argv) try {
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
  std::array<float, kTaps> taps{};
  make_lowpass(taps.data(), cutoff, wav.sample_rate);

  const int ch = wav.channels;
  const std::size_t n = static_cast<std::size_t>(wav.frames());
  std::vector<float> chan(n), result(n), out(wav.samples.size());

  sycl::queue q{sycl::property::queue::in_order()};
  float* d_x = sycl::malloc_device<float>(n, q);
  float* d_y = sycl::malloc_device<float>(n, q);
  const std::size_t groups = (n + kBlock - 1) / kBlock;

  for (int c = 0; c < ch; ++c) {
    for (std::size_t i = 0; i < n; ++i) chan[i] = wav.samples[i * ch + c];
    q.memcpy(d_x, chan.data(), n * sizeof(float));
    if (groups > 0) {
      q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> tile(sycl::range<1>(kTileLen), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * kBlock), sycl::range<1>(kBlock)),
            [=](sycl::nd_item<1> it) {
              const std::size_t base = it.get_group(0) * kBlock;
              const std::size_t t = it.get_local_id(0);
              const long long nn = static_cast<long long>(n);
              // Load history (kTaps-1 samples before the block) and the block itself.
              for (std::size_t i = t; i < kTileLen; i += kBlock) {
                const long long src = static_cast<long long>(base) - (kTaps - 1) +
                                      static_cast<long long>(i);
                tile[i] = (src >= 0 && src < nn) ? d_x[src] : 0.0f;
              }
              sycl::group_barrier(it.get_group());
              const std::size_t idx = base + t;
              if (idx >= n) return;
              float acc = 0.0f;
              for (int k = 0; k < kTaps; ++k) acc += taps[k] * tile[t + (kTaps - 1) - k];
              d_y[idx] = acc;
            });
      });
    }
    q.memcpy(result.data(), d_y, n * sizeof(float)).wait();
    for (std::size_t i = 0; i < n; ++i) out[i * ch + c] = result[i];
  }
  sycl::free(d_x, q);
  sycl::free(d_y, q);
  if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate)) return 1;
  return 0;
} catch (const sycl::exception& e) {
  std::fprintf(stderr, "SYCL error: %s\n", e.what());
  return 1;
}
