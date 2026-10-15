"""Pure comparison math used by verify (unit-tested independently of any toolchain)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .audio import dbfs


def _ordered_int32(x: np.ndarray) -> np.ndarray:
    """Map float32 bit patterns onto a monotonic integer line (for ULP distance)."""
    i = x.astype(np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, np.int64(-(2**31)) - i, i)


def ulp_distance_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ULP distance after rounding both arrays to float32 (\"ULP-ish\" for float64 data)."""
    return np.abs(_ordered_int32(a) - _ordered_int32(b))


def _fnum(x: float) -> float | None:
    """JSON-safe float: None for NaN/inf so evidence stays valid JSON."""
    return float(x) if math.isfinite(x) else None


def compare_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float = 1e-5,
    rtol: float = 0.0,
    max_ulp: int | None = None,
    max_report: int = 10,
) -> dict[str, Any]:
    """Elementwise comparison. Element i passes iff |c-r| <= atol + rtol*|r|
    (NaN must match NaN; +/-inf must match exactly)."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cand = np.asarray(candidate, dtype=np.float64).ravel()
    out: dict[str, Any] = {
        "n_reference": int(ref.size),
        "n_candidate": int(cand.size),
        "atol": atol,
        "rtol": rtol,
        "max_ulp_allowed": max_ulp,
    }
    if ref.size != cand.size:
        out.update(passed=False, reason=f"size mismatch: reference {ref.size} vs candidate {cand.size}")
        return out
    if ref.size == 0:
        out.update(passed=False, reason="no numbers to compare (empty outputs)")
        return out

    ref_nan, cand_nan = np.isnan(ref), np.isnan(cand)
    nan_mismatch = ref_nan != cand_nan
    both_finite = np.isfinite(ref) & np.isfinite(cand)
    inf_mismatch = ~(ref_nan | cand_nan) & ~both_finite & (ref != cand)

    diff = np.zeros_like(ref)
    diff[both_finite] = np.abs(cand[both_finite] - ref[both_finite])
    bound = atol + rtol * np.abs(ref)
    bad = nan_mismatch | inf_mismatch | (both_finite & (diff > bound))

    ulps = ulp_distance_f32(ref[both_finite], cand[both_finite])
    if max_ulp is not None:
        ulp_bad = np.zeros_like(bad)
        ulp_bad[both_finite] = ulps > max_ulp
        bad |= ulp_bad

    fin_diff = diff[both_finite]
    fin_ref = np.abs(ref[both_finite])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(fin_ref > 0, fin_diff / fin_ref, np.where(fin_diff > 0, np.inf, 0.0))
    bad_idx = np.flatnonzero(bad)
    out.update(
        passed=bool(bad_idx.size == 0),
        max_abs_err=_fnum(float(fin_diff.max()) if fin_diff.size else 0.0),
        rms_err=_fnum(float(np.sqrt(np.mean(fin_diff**2))) if fin_diff.size else 0.0),
        mean_abs_err=_fnum(float(fin_diff.mean()) if fin_diff.size else 0.0),
        max_rel_err=_fnum(float(rel.max()) if rel.size else 0.0),
        max_ulp_f32=int(ulps.max()) if ulps.size else 0,
        median_ulp_f32=float(np.median(ulps)) if ulps.size else 0.0,
        exact_matches=int(np.sum(cand == ref) + np.sum(ref_nan & cand_nan)),
        mismatches=int(bad_idx.size),
        nan_mismatches=int(nan_mismatch.sum()),
        reference_range=[_fnum(float(np.nanmin(ref))), _fnum(float(np.nanmax(ref)))]
        if not ref_nan.all() else None,
        first_mismatches=[
            {"index": int(i), "reference": _fnum(float(ref[i])), "candidate": _fnum(float(cand[i])),
             "abs_err": _fnum(float(abs(cand[i] - ref[i])))}
            for i in bad_idx[:max_report]
        ],
    )
    if bad_idx.size:
        out["reason"] = f"{bad_idx.size} of {ref.size} element(s) outside tolerance"
    return out


def audio_null_test(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    threshold_dbfs: float = -96.0,
    ref_rate: int | None = None,
    cand_rate: int | None = None,
) -> dict[str, Any]:
    """Subtract candidate from reference (a "null test"). Pass iff residual peak <= threshold.

    Arrays are (frames, channels) or (frames,), full scale = 1.0.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref[:, None]
    if cand.ndim == 1:
        cand = cand[:, None]
    out: dict[str, Any] = {
        "threshold_dbfs": threshold_dbfs,
        "reference_frames": int(ref.shape[0]),
        "candidate_frames": int(cand.shape[0]),
        "reference_channels": int(ref.shape[1]),
        "candidate_channels": int(cand.shape[1]),
        "reference_sample_rate": ref_rate,
        "candidate_sample_rate": cand_rate,
    }
    problems = []
    if ref.shape != cand.shape:
        problems.append(f"shape mismatch {ref.shape} vs {cand.shape}")
    if ref_rate is not None and cand_rate is not None and ref_rate != cand_rate:
        problems.append(f"sample-rate mismatch {ref_rate} vs {cand_rate}")
    if problems:
        out.update(passed=False, reason="; ".join(problems))
        return out
    if not (np.isfinite(ref).all() and np.isfinite(cand).all()):
        out.update(passed=False, reason="non-finite samples (NaN/inf) present")
        return out

    residual = ref - cand
    peak = float(np.max(np.abs(residual))) if residual.size else 0.0
    rms = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0
    sig_rms = float(np.sqrt(np.mean(ref**2))) if ref.size else 0.0
    peak_db, rms_db = dbfs(peak), dbfs(rms)
    per_channel = []
    for ch in range(residual.shape[1]):
        r = residual[:, ch]
        pk = float(np.max(np.abs(r))) if r.size else 0.0
        per_channel.append({
            "channel": ch,
            "residual_peak_dbfs": dbfs(pk),
            "residual_rms_dbfs": dbfs(float(np.sqrt(np.mean(r**2))) if r.size else 0.0),
            "peak_frame": int(np.argmax(np.abs(r))) if r.size else None,
        })
    passed = peak_db is None or peak_db <= threshold_dbfs
    out.update(
        passed=bool(passed),
        bit_exact=peak == 0.0,
        residual_peak=peak,
        residual_peak_dbfs=peak_db,
        residual_rms_dbfs=rms_db,
        reference_rms_dbfs=dbfs(sig_rms),
        null_depth_db=(dbfs(sig_rms) - rms_db) if (rms_db is not None and sig_rms > 0) else None,
        per_channel=per_channel,
    )
    if not passed:
        out["reason"] = f"residual peak {peak_db:.2f} dBFS exceeds threshold {threshold_dbfs} dBFS"
    return out
