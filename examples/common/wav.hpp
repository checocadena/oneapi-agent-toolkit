// Minimal WAV I/O shared by the audio examples (reference C++, CUDA and SYCL builds).
// Reads PCM16 or IEEE float32 (mono/multichannel, interleaved); writes IEEE float32.
// Host-only code: no CUDA or SYCL constructs, so c2s leaves it untouched.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct WavData {
  std::vector<float> samples;  // interleaved
  int channels = 0;
  int sample_rate = 0;
  size_t frames() const { return channels ? samples.size() / channels : 0; }
};

inline bool wav_read(const std::string& path, WavData& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::vector<unsigned char> buf;
  unsigned char tmp[65536];
  size_t n;
  while ((n = std::fread(tmp, 1, sizeof(tmp), f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
  std::fclose(f);
  if (buf.size() < 12 || std::memcmp(buf.data(), "RIFF", 4) || std::memcmp(buf.data() + 8, "WAVE", 4))
    return false;
  auto u16 = [&](size_t p) { return uint16_t(buf[p] | (buf[p + 1] << 8)); };
  auto u32 = [&](size_t p) {
    return uint32_t(buf[p]) | (uint32_t(buf[p + 1]) << 8) | (uint32_t(buf[p + 2]) << 16) |
           (uint32_t(buf[p + 3]) << 24);
  };
  int fmt_tag = 0, bits = 0;
  size_t pos = 12, data_pos = 0, data_len = 0;
  while (pos + 8 <= buf.size()) {
    uint32_t len = u32(pos + 4);
    if (!std::memcmp(buf.data() + pos, "fmt ", 4)) {
      fmt_tag = u16(pos + 8);
      out.channels = u16(pos + 10);
      out.sample_rate = int(u32(pos + 12));
      bits = u16(pos + 22);
      if (fmt_tag == 0xFFFE && len >= 26) fmt_tag = u16(pos + 32);
    } else if (!std::memcmp(buf.data() + pos, "data", 4)) {
      data_pos = pos + 8;
      data_len = len;
    }
    pos += 8 + len + (len & 1);
  }
  if (!data_pos || out.channels <= 0) return false;
  if (data_pos + data_len > buf.size()) data_len = buf.size() - data_pos;
  if (fmt_tag == 3 && bits == 32) {
    out.samples.resize(data_len / 4);
    std::memcpy(out.samples.data(), buf.data() + data_pos, out.samples.size() * 4);
  } else if (fmt_tag == 1 && bits == 16) {
    out.samples.resize(data_len / 2);
    for (size_t i = 0; i < out.samples.size(); ++i)
      out.samples[i] = float(int16_t(u16(data_pos + 2 * i))) / 32768.0f;
  } else {
    return false;
  }
  return true;
}

inline bool wav_write_f32(const std::string& path, const float* samples, size_t count,
                          int channels, int sample_rate) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
  auto w16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
  uint32_t data_bytes = uint32_t(count * 4);
  std::fwrite("RIFF", 1, 4, f);
  w32(4 + 8 + 16 + 8 + data_bytes);
  std::fwrite("WAVEfmt ", 1, 8, f);
  w32(16);
  w16(3);  // IEEE float
  w16(uint16_t(channels));
  w32(uint32_t(sample_rate));
  w32(uint32_t(sample_rate * channels * 4));
  w16(uint16_t(channels * 4));
  w16(32);
  std::fwrite("data", 1, 4, f);
  w32(data_bytes);
  std::fwrite(samples, 4, count, f);
  return std::fclose(f) == 0;
}
