// Hand-written idiomatic SYCL 2020 port of fir.cu (the target end state; no dpct helpers).
// Taps are a kernel argument (USM), the tile is a sycl::local_accessor, 1-D nd_range.
// Usage: fir <in.wav> <out.wav> [cutoff_hz=2000]
#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fir_coeffs.hpp"
#include "wav.hpp"

constexpr size_t kBlock = 256;

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

  const int ch = wav.channels;
  const size_t n = wav.frames();
  std::vector<float> chan(n), result(n), out(wav.samples.size());
  try {
    sycl::queue q;
    float* d_taps = sycl::malloc_device<float>(kTaps, q);
    float* d_x = sycl::malloc_device<float>(n, q);
    float* d_y = sycl::malloc_device<float>(n, q);
    q.memcpy(d_taps, h, sizeof(h)).wait();
    const size_t global = (n + kBlock - 1) / kBlock * kBlock;

    for (int c = 0; c < ch; ++c) {
      for (size_t i = 0; i < n; ++i) chan[i] = wav.samples[i * ch + c];
      q.memcpy(d_x, chan.data(), n * sizeof(float)).wait();
      q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> tile{sycl::range<1>{kBlock + kTaps - 1}, cgh};
        cgh.parallel_for(sycl::nd_range<1>{global, kBlock}, [=](sycl::nd_item<1> it) {
          const long base = static_cast<long>(it.get_group(0) * kBlock);
          const size_t t = it.get_local_id(0);
          for (size_t i = t; i < kBlock + kTaps - 1; i += kBlock) {
            const long src = base - (kTaps - 1) + static_cast<long>(i);
            tile[i] = (src >= 0 && src < static_cast<long>(n)) ? d_x[src] : 0.0f;
          }
          sycl::group_barrier(it.get_group());
          const size_t idx = it.get_global_id(0);
          if (idx >= n) return;
          float acc = 0.0f;
          for (int k = 0; k < kTaps; ++k) acc += d_taps[k] * tile[t + (kTaps - 1) - k];
          d_y[idx] = acc;
        });
      }).wait_and_throw();
      q.memcpy(result.data(), d_y, n * sizeof(float)).wait();
      for (size_t i = 0; i < n; ++i) out[i * ch + c] = result[i];
    }
    sycl::free(d_taps, q);
    sycl::free(d_x, q);
    sycl::free(d_y, q);
  } catch (const sycl::exception& ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    return 1;
  }
  if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate)) return 1;
  return 0;
}
