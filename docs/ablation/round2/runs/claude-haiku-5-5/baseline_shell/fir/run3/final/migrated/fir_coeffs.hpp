// Shared host-side coefficient design for the FIR example (Hann-windowed sinc low-pass).
// Computed in double and rounded once to float so every build gets identical taps.
#pragma once

#include <cmath>

constexpr int kTaps = 64;

inline void make_lowpass(float* h, double cutoff_hz, int sample_rate) {
  const double pi = 3.14159265358979323846;
  const double fc = cutoff_hz / sample_rate;  // normalised cutoff (cycles/sample)
  const double mid = (kTaps - 1) / 2.0;
  double taps[kTaps];
  double sum = 0.0;
  for (int k = 0; k < kTaps; ++k) {
    const double t = k - mid;
    const double sinc = (t == 0.0) ? 2.0 * fc : std::sin(2.0 * pi * fc * t) / (pi * t);
    const double window = 0.5 - 0.5 * std::cos(2.0 * pi * k / (kTaps - 1));
    taps[k] = sinc * window;
    sum += taps[k];
  }
  for (int k = 0; k < kTaps; ++k) h[k] = static_cast<float>(taps[k] / sum);  // unity DC gain
}
