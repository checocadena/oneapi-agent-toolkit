"""Minimal, dependency-free (numpy only) WAV I/O and dBFS helpers.

Supports PCM 16/24/32-bit and IEEE float 32/64-bit, mono or multichannel, including
WAVE_FORMAT_EXTENSIBLE. Samples are returned as float64 in [-1, 1] with shape
(frames, channels).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


@dataclass
class Wav:
    samples: np.ndarray  # float64, shape (frames, channels)
    sample_rate: int
    bits: int
    format_tag: int  # 1 = PCM, 3 = float

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])


def is_wav_bytes(head: bytes) -> bool:
    return len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE"


def read_wav(path: str | Path) -> Wav:
    data = Path(path).read_bytes()
    if not is_wav_bytes(data):
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos = 12
    fmt = None
    raw = None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, channels, rate, _byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if tag == WAVE_FORMAT_EXTENSIBLE and len(fmt) >= 26:
        tag = struct.unpack_from("<H", fmt, 24)[0]  # first 2 bytes of SubFormat GUID
    if channels == 0 or block_align == 0:
        raise ValueError(f"{path}: invalid header")
    frames = len(raw) // block_align
    raw = raw[: frames * block_align]

    if tag == WAVE_FORMAT_IEEE_FLOAT and bits in (32, 64):
        arr = np.frombuffer(raw, dtype="<f4" if bits == 32 else "<f8").astype(np.float64)
    elif tag == WAVE_FORMAT_PCM and bits == 16:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif tag == WAVE_FORMAT_PCM and bits == 32:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif tag == WAVE_FORMAT_PCM and bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        arr = ints.astype(np.float64) / 8388608.0
    elif tag == WAVE_FORMAT_PCM and bits == 8:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: unsupported WAV format tag={tag} bits={bits}")
    return Wav(arr.reshape(frames, channels), rate, bits, tag)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int, *, bits: int = 32,
              float_format: bool = True) -> None:
    """Write float32 (default) or PCM16 WAV. ``samples``: (frames,) or (frames, channels)."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    channels = x.shape[1]
    if float_format:
        if bits != 32:
            raise ValueError("float WAV output supports 32-bit only")
        payload = x.astype("<f4").tobytes()
        tag = WAVE_FORMAT_IEEE_FLOAT
    else:
        if bits != 16:
            raise ValueError("PCM WAV output supports 16-bit only")
        payload = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2").tobytes()
        tag = WAVE_FORMAT_PCM
    block_align = channels * bits // 8
    fmt = struct.pack("<HHIIHH", tag, channels, sample_rate, sample_rate * block_align,
                      block_align, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        body += b"\x00"
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def dbfs(value: float) -> float | None:
    """Amplitude (full scale = 1.0) to dBFS; None for exact digital silence."""
    if value <= 0.0 or not math.isfinite(value):
        return None if value == 0.0 else float("nan")
    return 20.0 * math.log10(value)


def sine_sweep(sample_rate: int, seconds: float, f0: float = 20.0, f1: float = 20000.0,
               amplitude: float = 0.5) -> np.ndarray:
    """Exponential (log) sine sweep, the standard measurement signal."""
    n = int(round(sample_rate * seconds))
    t = np.arange(n) / sample_rate
    f1 = min(f1, sample_rate / 2 * 0.95)
    k = math.log(f1 / f0)
    phase = 2 * math.pi * f0 * seconds / k * (np.exp(t / seconds * k) - 1.0)
    return amplitude * np.sin(phase)


def make_test_signal(sample_rate: int = 48000, seconds: float = 2.0, seed: int = 1234,
                noise_db: float = -30.0) -> np.ndarray:
    """Deterministic sweep + white noise test signal (mono)."""
    rng = np.random.default_rng(seed)
    sweep = sine_sweep(sample_rate, seconds)
    noise = rng.standard_normal(sweep.shape[0]) * (10 ** (noise_db / 20.0))
    return np.clip(sweep + noise, -1.0, 1.0)
