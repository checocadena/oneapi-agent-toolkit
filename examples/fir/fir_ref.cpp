// Reference (ground truth): 64-tap windowed-sinc low-pass FIR, plain C++.
//   y[n] = sum_{k=0}^{63} h[k] * x[n-k]   (x[m] = 0 for m < 0), per channel.
// Usage: fir_ref <in.wav> <out.wav> [cutoff_hz=2000]
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "fir_coeffs.hpp"
#include "wav.hpp"

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
  const long frames = static_cast<long>(wav.frames());
  std::vector<float> out(wav.samples.size());
  for (long n = 0; n < frames; ++n) {
    for (int c = 0; c < ch; ++c) {
      float acc = 0.0f;
      for (int k = 0; k < kTaps; ++k) {
        const long m = n - k;
        if (m >= 0) acc += h[k] * wav.samples[m * ch + c];
      }
      out[n * ch + c] = acc;
    }
  }
  if (!wav_write_f32(argv[2], out.data(), out.size(), ch, wav.sample_rate)) return 1;
  return 0;
}
