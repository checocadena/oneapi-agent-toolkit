"""Reading program outputs as numeric arrays for comparison."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .audio import is_wav_bytes, read_wav

_NUM_RE = re.compile(
    r"[-+]?(?:(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    r"|(?<![A-Za-z])(?:inf(?:inity)?|nan)(?![A-Za-z]))",
    re.IGNORECASE,
)


def parse_numbers(text: str) -> np.ndarray:
    """Every number in ``text`` (whitespace/comma/any-separated), in order, as float64."""
    vals = [float(tok) for tok in _NUM_RE.findall(text)]
    return np.asarray(vals, dtype=np.float64)


def load_output(path: str | Path, fmt: str = "auto") -> tuple[np.ndarray, str, dict]:
    """Load an output file. Returns (array, detected_format, metadata).

    Formats: ``wav``, ``npy``, ``f32``/``f64`` (raw little-endian), ``text``, or ``auto``
    (RIFF -> wav, numpy magic -> npy, decodable text with numbers -> text, else f32).
    """
    p = Path(path)
    data = p.read_bytes()
    meta: dict = {"bytes": len(data)}
    if fmt == "auto":
        if is_wav_bytes(data[:12]):
            fmt = "wav"
        elif data[:6] == b"\x93NUMPY":
            fmt = "npy"
        else:
            try:
                text = data.decode("utf-8")
                fmt = "text" if _NUM_RE.search(text) or not text.strip() else "f32"
            except UnicodeDecodeError:
                fmt = "f32"
                meta["note"] = "binary output interpreted as raw little-endian float32"
    if fmt == "wav":
        w = read_wav(p)
        meta.update(sample_rate=w.sample_rate, channels=w.channels, bits=w.bits,
                    format_tag=w.format_tag)
        return w.samples, "wav", meta
    if fmt == "npy":
        return np.load(p, allow_pickle=False).astype(np.float64), "npy", meta
    if fmt == "f32":
        return np.frombuffer(data[: len(data) // 4 * 4], dtype="<f4").astype(np.float64), "f32", meta
    if fmt == "f64":
        return np.frombuffer(data[: len(data) // 8 * 8], dtype="<f8").astype(np.float64), "f64", meta
    if fmt == "text":
        return parse_numbers(data.decode("utf-8", errors="replace")), "text", meta
    raise ValueError(f"unknown output format {fmt!r}")
