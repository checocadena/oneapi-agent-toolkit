#!/usr/bin/env python3
"""Generate the example audio inputs procedurally (no audio files are committed).

Writes examples/data/sweep_noise_48k.wav (mono, float32, 2 s) and
examples/data/sweep_noise_stereo_44k.wav (stereo PCM16, 1 s): a log sine sweep
20 Hz..20 kHz at -6 dBFS plus white noise at -30 dBFS, from fixed seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oneapi_agent.audio import make_test_signal, write_wav  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    mono = make_test_signal(48000, 2.0, seed=1234)
    write_wav(DATA / "sweep_noise_48k.wav", mono, 48000)
    left = make_test_signal(44100, 1.0, seed=7)
    right = make_test_signal(44100, 1.0, seed=8)[::-1].copy()  # reversed sweep on R
    write_wav(DATA / "sweep_noise_stereo_44k.wav", np.stack([left, right], axis=1), 44100,
              bits=16, float_format=False)
    for p in sorted(DATA.glob("*.wav")):
        print(f"wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
