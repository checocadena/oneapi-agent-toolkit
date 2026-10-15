// Reference (ground truth): tanh soft-clip "fuzz" waveshaper, plain C++.
//   y = gain * tanh(drive * x) / tanh(drive)      (normalised so |y| <= gain)
// Usage: fuzz_ref <in.wav> <out.wav> [drive=8] [gain=0.8]
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "wav.hpp"

static void fuzz_cpu(const float* in, float* out, size_t n, float drive, float gain) {
  const float norm = gain / std::tanh(drive);
  for (size_t i = 0; i < n; ++i) out[i] = norm * std::tanh(drive * in[i]);
}

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
  std::vector<float> out(wav.samples.size());
  fuzz_cpu(wav.samples.data(), out.data(), out.size(), drive, gain);
  if (!wav_write_f32(argv[2], out.data(), out.size(), wav.channels, wav.sample_rate)) return 1;
  return 0;
}
