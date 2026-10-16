// SYCL 2020 port of the 64-tap FIR.
// Taps live in a device USM buffer (read-only in the kernel); each work-item computes
// one output sample of one channel; the input tile + history is staged in local memory.
// Usage: fir <in.wav> <out.wav> [cutoff_hz=2000]
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fir_coeffs.hpp"
#include "wav.hpp"

namespace {

constexpr std::size_t kBlock = 256;
constexpr std::size_t kHistory = kTaps - 1;
constexpr std::size_t kTile = kBlock + kHistory;

// x and y are one de-interleaved channel of length n; taps is a device array of kTaps floats.
void run_fir(sycl::queue &q, const float *d_x, float *d_y, const float *d_taps, int n) {
  const std::size_t groups = (static_cast<std::size_t>(n) + kBlock - 1) / kBlock;
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> tile(sycl::range<1>(kTile), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kBlock), sycl::range<1>(kBlock)),
        [=](sycl::nd_item<1> it) {
          const int base = static_cast<int>(it.get_group(0) * kBlock);
          const std::size_t t = it.get_local_id(0);
          // Load history (kHistory samples before the block) and the block itself.
          for (std::size_t i = t; i < kTile; i += kBlock) {
            const int src = base - static_cast<int>(kHistory) + static_cast<int>(i);
            tile[i] = (src >= 0 && src < n) ? d_x[src] : 0.0f;
          }
          sycl::group_barrier(it.get_group());
          const int idx = base + static_cast<int>(t);
          if (idx >= n) return;
          float acc = 0.0f;
          for (std::size_t k = 0; k < kTaps; ++k)
            acc += d_taps[k] * tile[t + kHistory - k];
          d_y[idx] = acc;
        });
  });
}

}  // namespace

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
  float h[kTaps];
  make_lowpass(h, cutoff, wav.sample_rate);

  try {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};

    float *d_taps = sycl::malloc_device<float>(kTaps, q);
    q.memcpy(d_taps, h, sizeof(h)).wait();

    const int ch = wav.channels;
    const int n = static_cast<int>(wav.frames());
    std::vector<float> chan(n), result(n), out(wav.samples.size());
    if (n > 0) {
      float *d_x = sycl::malloc_device<float>(n, q);
      float *d_y = sycl::malloc_device<float>(n, q);
      for (int c = 0; c < ch; ++c) {
        for (int i = 0; i < n; ++i) chan[i] = wav.samples[i * ch + c];
        q.memcpy(d_x, chan.data(), n * sizeof(float));
        run_fir(q, d_x, d_y, d_taps, n);
        q.memcpy(result.data(), d_y, n * sizeof(float)).wait();
        for (int i = 0; i < n; ++i) out[i * ch + c] = result[i];
      }
      sycl::free(d_x, q);
      sycl::free(d_y, q);
    }
    sycl::free(d_taps, q);

    if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate)) return 1;
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  }
  return 0;
}
