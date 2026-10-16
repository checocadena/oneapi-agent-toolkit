// SYCL version of the tanh soft-clip fuzz.
// Usage: fuzz <in.wav> <out.wav> [drive=8] [gain=0.8]
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "wav.hpp"

int main(int argc, char** argv) try {
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
  const size_t n = wav.samples.size();
  const float norm = gain / std::tanh(drive);

  sycl::queue q{sycl::property_list{sycl::property::queue::in_order{},
                                    sycl::property::queue::enable_profiling{}}};

  float* d_in = sycl::malloc_device<float>(n, q);
  float* d_out = sycl::malloc_device<float>(n, q);
  q.memcpy(d_in, wav.samples.data(), n * sizeof(float)).wait();

  sycl::event ev = q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
    const size_t i = idx[0];
    d_out[i] = norm * sycl::tanh(drive * d_in[i]);
  });
  ev.wait_and_throw();
  const double ms =
      (ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
       ev.get_profiling_info<sycl::info::event_profiling::command_start>()) /
      1e6;
  std::printf("kernel_ms=%f\n", ms);

  std::vector<float> out(n);
  q.memcpy(out.data(), d_out, n * sizeof(float)).wait();
  sycl::free(d_in, q);
  sycl::free(d_out, q);
  if (!wav_write_f32(argv[2], out.data(), out.size(), wav.channels, wav.sample_rate)) return 1;
  return 0;
} catch (const sycl::exception& e) {
  std::cerr << "SYCL error: " << e.what() << std::endl;
  return 1;
}
