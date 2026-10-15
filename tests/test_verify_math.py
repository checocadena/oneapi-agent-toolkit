"""Comparison math, WAV I/O and output parsing: no external tools involved."""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from oneapi_agent.audio import dbfs, make_test_signal, read_wav, sine_sweep, write_wav
from oneapi_agent.compare import audio_null_test, compare_arrays, ulp_distance_f32
from oneapi_agent.numio import load_output, parse_numbers

# ------------------------------------------------------------------ arrays


def test_identical_arrays_pass_exactly():
    a = np.linspace(-1, 1, 101)
    s = compare_arrays(a, a.copy())
    assert s["passed"] and s["max_abs_err"] == 0.0 and s["mismatches"] == 0
    assert s["exact_matches"] == 101 and s["max_ulp_f32"] == 0


def test_within_and_outside_tolerance():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    cand = ref + np.array([0.0, 5e-6, -2e-5, 0.0])
    s = compare_arrays(ref, cand, atol=1e-5)
    assert not s["passed"]
    assert s["mismatches"] == 1
    assert s["first_mismatches"][0]["index"] == 2
    assert s["max_abs_err"] == pytest.approx(2e-5)
    assert s["rms_err"] == pytest.approx(math.sqrt((5e-6**2 + 2e-5**2) / 4))
    assert compare_arrays(ref, cand, atol=1e-4)["passed"]


def test_rtol_scales_with_magnitude():
    ref = np.array([1e6, 1.0])
    cand = np.array([1e6 + 0.5, 1.0])
    assert not compare_arrays(ref, cand, atol=1e-6)["passed"]
    assert compare_arrays(ref, cand, atol=1e-6, rtol=1e-6)["passed"]


def test_size_mismatch_and_empty():
    s = compare_arrays(np.zeros(3), np.zeros(4))
    assert not s["passed"] and "size mismatch" in s["reason"]
    assert not compare_arrays(np.zeros(0), np.zeros(0))["passed"]


def test_nan_and_inf_handling():
    ref = np.array([np.nan, np.inf, 1.0])
    assert compare_arrays(ref, ref.copy())["passed"]
    s = compare_arrays(ref, np.array([0.0, np.inf, 1.0]))
    assert not s["passed"] and s["nan_mismatches"] == 1
    assert not compare_arrays(ref, np.array([np.nan, -np.inf, 1.0]))["passed"]
    json.dumps(s)  # evidence stays valid JSON (no NaN literals in stats)


def test_ulp_distance_float32():
    one = np.array([1.0], dtype=np.float32)
    nxt = np.nextafter(one, np.float32(2.0))
    assert ulp_distance_f32(one, nxt)[0] == 1
    assert ulp_distance_f32(np.array([-0.0]), np.array([0.0]))[0] == 0
    # across zero: smallest negative and positive subnormals are 2 ulps apart
    tiny = np.float32(1e-45)
    assert ulp_distance_f32(np.array([-tiny]), np.array([tiny]))[0] == 2


def test_max_ulp_gate():
    ref = np.array([1.0], dtype=np.float32)
    cand = ref.copy()
    for _ in range(3):
        cand = np.nextafter(cand, np.float32(2.0))
    s = compare_arrays(ref, cand, atol=1.0, max_ulp=2)
    assert s["max_ulp_f32"] == 3 and not s["passed"]
    assert compare_arrays(ref, cand, atol=1.0, max_ulp=3)["passed"]


# ------------------------------------------------------------------ audio null test


def test_audio_null_bit_exact():
    x = make_test_signal(8000, 0.25)
    s = audio_null_test(x, x.copy())
    assert s["passed"] and s["bit_exact"]
    assert s["residual_peak_dbfs"] is None  # digital silence, reported without -inf
    json.dumps(s)


def test_audio_null_float_rounding_passes():
    x = make_test_signal(48000, 0.5)
    y = x.astype(np.float32).astype(np.float64)  # float32 rounding: ~ -150 dBFS residual
    s = audio_null_test(x, y)
    assert s["passed"] and not s["bit_exact"]
    assert s["residual_peak_dbfs"] < -140


@pytest.mark.parametrize("level_db,should_pass", [(-120.0, True), (-97.0, True), (-95.0, False),
                                                  (-60.0, False)])
def test_audio_null_threshold(level_db, should_pass):
    x = sine_sweep(16000, 0.5)
    err = np.zeros_like(x)
    err[1234] = 10 ** (level_db / 20)  # one sample error of exactly level_db dBFS
    s = audio_null_test(x, x + err, threshold_dbfs=-96.0)
    assert s["residual_peak_dbfs"] == pytest.approx(level_db, abs=1e-9)
    assert s["passed"] is should_pass
    assert s["per_channel"][0]["peak_frame"] == 1234


def test_audio_null_shape_and_rate_mismatch():
    x = np.zeros((100, 2))
    assert not audio_null_test(x, np.zeros((100, 1)))["passed"]
    assert not audio_null_test(x, np.zeros((99, 2)))["passed"]
    s = audio_null_test(x, x, ref_rate=48000, cand_rate=44100)
    assert not s["passed"] and "sample-rate" in s["reason"]


def test_audio_null_rejects_nan():
    x = np.zeros(10)
    y = x.copy()
    y[3] = np.nan
    assert not audio_null_test(x, y)["passed"]


def test_dbfs():
    assert dbfs(1.0) == 0.0
    assert dbfs(0.5) == pytest.approx(-6.0206, abs=1e-4)
    assert dbfs(0.0) is None


# ------------------------------------------------------------------ WAV I/O


def test_wav_float32_roundtrip(tmp_path):
    x = np.stack([make_test_signal(44100, 0.1, seed=1), make_test_signal(44100, 0.1, seed=2)], 1)
    p = tmp_path / "s.wav"
    write_wav(p, x, 44100)
    w = read_wav(p)
    assert (w.sample_rate, w.channels, w.bits, w.format_tag) == (44100, 2, 32, 3)
    assert np.array_equal(w.samples, x.astype(np.float32).astype(np.float64))


def test_wav_pcm16_roundtrip(tmp_path):
    x = sine_sweep(8000, 0.1)
    p = tmp_path / "s16.wav"
    write_wav(p, x, 8000, bits=16, float_format=False)
    w = read_wav(p)
    assert w.bits == 16 and w.format_tag == 1
    assert np.max(np.abs(w.samples[:, 0] - x)) <= 1 / 32767 + 1e-12


def _raw_wav(fmt_tag: int, bits: int, channels: int, payload: bytes, extensible: bool = False) -> bytes:
    block = channels * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHH", 0xFFFE, channels, 8000, 8000 * block, block, bits)
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", fmt_tag) + b"\x00" * 14
    else:
        fmt = struct.pack("<HHIIHH", fmt_tag, channels, 8000, 8000 * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"LIST" + struct.pack("<I", 4) + b"INFO"  # an unrelated chunk to skip
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_wav_pcm24_and_extensible(tmp_path):
    # 24-bit samples: +0.5 (0x400000) and -0.5 (0xC00000)
    payload = bytes([0x00, 0x00, 0x40, 0x00, 0x00, 0xC0])
    p = tmp_path / "s24.wav"
    p.write_bytes(_raw_wav(1, 24, 1, payload, extensible=True))
    w = read_wav(p)
    assert w.bits == 24 and w.format_tag == 1
    assert w.samples[:, 0].tolist() == [0.5, -0.5]


def test_wav_rejects_garbage(tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"not a wav at all")
    with pytest.raises(ValueError):
        read_wav(p)


# ------------------------------------------------------------------ output parsing


def test_parse_numbers_formats():
    text = "kernel_ms=0.25\nvalues: 1, -2.5e-3, +3., .5, 1E2 nan inf -inf\ninformation 7"
    vals = parse_numbers(text)
    assert vals[:6].tolist() == [0.25, 1.0, -2.5e-3, 3.0, 0.5, 100.0]
    assert math.isnan(vals[6])
    assert vals[7] == math.inf and vals[8] == -math.inf
    assert vals[9] == 7.0 and len(vals) == 10  # "information" is not "inf"


def test_load_output_sniffing(tmp_path):
    f32 = tmp_path / "a.bin"
    np.array([1.5, -2.0], dtype="<f4").tofile(f32)
    arr, fmt, _ = load_output(f32)
    assert fmt == "f32" and arr.tolist() == [1.5, -2.0]

    npy = tmp_path / "a.npy"
    np.save(npy, np.array([[1.0, 2.0]]))
    arr, fmt, _ = load_output(npy)
    assert fmt == "npy" and arr.tolist() == [[1.0, 2.0]]

    txt = tmp_path / "a.txt"
    txt.write_text("1 2 3\n")
    arr, fmt, _ = load_output(txt)
    assert fmt == "text" and arr.tolist() == [1.0, 2.0, 3.0]

    wav = tmp_path / "a.wav"
    write_wav(wav, np.zeros(10), 8000)
    arr, fmt, meta = load_output(wav)
    assert fmt == "wav" and meta["sample_rate"] == 8000 and arr.shape == (10, 1)

    f64 = tmp_path / "b.bin"
    np.array([0.1], dtype="<f8").tofile(f64)
    assert load_output(f64, "f64")[0].tolist() == [0.1]
