// Host reference: same math as the CUDA kernel (tanhf(drive*x) * norm), plus input generator.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "wav.hpp"
int main(int argc, char** argv) {
  if (std::string(argv[1]) == "gen") {
    std::vector<float> s(48000 * 2);
    for (size_t i = 0; i < s.size(); ++i) s[i] = 0.9f * std::sin(0.01f * i) + 0.3f * std::sin(0.137f * i);
    wav_write_f32(argv[2], s.data(), s.size(), 1, 48000);
    return 0;
  }
  WavData w; if (!wav_read(argv[1], w)) return 1;
  float drive = std::strtof(argv[3], nullptr), gain = std::strtof(argv[4], nullptr);
  float norm = gain / std::tanh(drive);
  std::vector<float> o(w.samples.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = norm * std::tanh(drive * w.samples[i]);
  wav_write_f32(argv[2], o.data(), o.size(), w.channels, w.sample_rate);
  return 0;
}
