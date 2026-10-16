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

  sycl::queue q{sycl::property::queue::enable_profiling{}};
  std::vector<float> out(n);
  {
    sycl::buffer<float, 1> in_buf(wav.samples.data(), sycl::range<1>(n));
    sycl::buffer<float, 1> out_buf(out.data(), sycl::range<1>(n));
    sycl::event ev = q.submit([&](sycl::handler& h) {
      sycl::accessor in(in_buf, h, sycl::read_only);
      sycl::accessor o(out_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        o[i] = norm * sycl::tanh(drive * in[i]);
      });
    });
    ev.wait_and_throw();
    const double ms = (ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                       ev.get_profiling_info<sycl::info::event_profiling::command_start>()) /
                      1.0e6;
    std::printf("kernel_ms=%f\n", ms);
  }  // buffer destruction copies results back into `out`

  if (!wav_write_f32(argv[2], out.data(), out.size(), wav.channels, wav.sample_rate)) return 1;
  return 0;
} catch (sycl::exception const& e) {
  std::cerr << "SYCL error: " << e.what() << std::endl;
  return 1;
}
