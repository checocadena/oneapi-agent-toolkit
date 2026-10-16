// SYCL 2020 port of the 64-tap FIR (originally fir.cu).
// Taps are passed by value to the kernel (replacing __constant__ memory); each work-item
// computes one output sample of one channel; the input tile + history is staged through
// work-group local memory.
// Usage: fir <in.wav> <out.wav> [cutoff_hz=2000]
#include <sycl/sycl.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

#include "fir_coeffs.hpp"
#include "wav.hpp"

namespace {

constexpr std::size_t kBlock = 256;
constexpr std::size_t kTileLen = kBlock + kTaps - 1;

using TapArray = std::array<float, kTaps>;

// x and y are one de-interleaved channel of length n.
void run_fir(sycl::queue& q, const float* x, float* y, int n, const TapArray& taps) {
  const std::size_t global = ((static_cast<std::size_t>(n) + kBlock - 1) / kBlock) * kBlock;

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> tile(sycl::range<1>(kTileLen), cgh);

    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(kBlock)),
                     [=](sycl::nd_item<1> it) {
                       const int lid = static_cast<int>(it.get_local_id(0));
                       const int base = static_cast<int>(it.get_group(0) * kBlock);
                       const int lsz = static_cast<int>(kBlock);

                       // Load history (kTaps-1 samples before the block) and the block itself.
                       for (int i = lid; i < static_cast<int>(kTileLen); i += lsz) {
                         const int src = base - (kTaps - 1) + i;
                         tile[i] = (src >= 0 && src < n) ? x[src] : 0.0f;
                       }
                       sycl::group_barrier(it.get_group());

                       const int idx = base + lid;
                       if (idx >= n) return;
                       float acc = 0.0f;
                       for (int k = 0; k < kTaps; ++k)
                         acc += taps[k] * tile[lid + (kTaps - 1) - k];
                       y[idx] = acc;
                     });
  });
  q.wait_and_throw();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s in.wav out.wav [cutoff_hz]\n", argv[0]);
    return 2;
  }
  try {
    const double cutoff = argc > 3 ? std::strtod(argv[3], nullptr) : 2000.0;
    WavData wav;
    if (!wav_read(argv[1], wav)) {
      std::fprintf(stderr, "cannot read %s\n", argv[1]);
      return 1;
    }
    float h[kTaps];
    make_lowpass(h, cutoff, wav.sample_rate);
    TapArray taps{};
    for (int k = 0; k < kTaps; ++k) taps[k] = h[k];

    const int ch = wav.channels;
    const int n = static_cast<int>(wav.frames());
    std::vector<float> chan(n), result(n), out(wav.samples.size());

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    float* d_x = sycl::malloc_device<float>(n, q);
    float* d_y = sycl::malloc_device<float>(n, q);
    if (!d_x || !d_y) {
      std::fprintf(stderr, "device allocation failed\n");
      return 1;
    }

    for (int c = 0; c < ch; ++c) {
      for (int i = 0; i < n; ++i) chan[i] = wav.samples[i * ch + c];
      q.memcpy(d_x, chan.data(), n * sizeof(float)).wait();
      run_fir(q, d_x, d_y, n, taps);
      q.memcpy(result.data(), d_y, n * sizeof(float)).wait();
      for (int i = 0; i < n; ++i) out[i * ch + c] = result[i];
    }
    sycl::free(d_x, q);
    sycl::free(d_y, q);
    if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate)) return 1;
    return 0;
  } catch (const sycl::exception& e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
