// SYCL 2020 port of the tanh soft-clip fuzz.
// Usage: fuzz <in.wav> <out.wav> [drive=8] [gain=0.8]
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "wav.hpp"

int main(int argc, char** argv) {
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
  const std::size_t n = wav.samples.size();
  const float norm = gain / std::tanh(drive);

  try {
    sycl::queue q{sycl::default_selector_v,
                  {sycl::property::queue::enable_profiling{}}};

    std::vector<float> out(n);
    {
      float* d_in = sycl::malloc_device<float>(n, q);
      float* d_out = sycl::malloc_device<float>(n, q);
      q.memcpy(d_in, wav.samples.data(), n * sizeof(float)).wait();

      const std::size_t block = 256;
      const std::size_t grid = (n + block - 1) / block;
      sycl::event ev = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>{grid * block, block},
                       [=](sycl::nd_item<1> it) {
                         const std::size_t i = it.get_global_id(0);
                         if (i < n) d_out[i] = norm * sycl::tanh(drive * d_in[i]);
                       });
      });
      ev.wait();
      const float ms = (ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                        ev.get_profiling_info<sycl::info::event_profiling::command_start>()) /
                       1.0e6f;
      std::printf("kernel_ms=%f\n", ms);

      q.memcpy(out.data(), d_out, n * sizeof(float)).wait();
      sycl::free(d_in, q);
      sycl::free(d_out, q);
    }
    if (!wav_write_f32(argv[2], out.data(), out.size(), wav.channels, wav.sample_rate)) return 1;
  } catch (const sycl::exception& e) {
    std::fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  }
  return 0;
}
