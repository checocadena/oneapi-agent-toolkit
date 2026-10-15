"""benchmark: time a command N times on each SYCL device (selected via ONEAPI_DEVICE_SELECTOR).

Only measured wall-clock times are reported. Nothing is estimated or extrapolated.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..parsers.sycl_ls import parse_sycl_ls
from ..result import Status, ToolResult
from ..toolchain import Runner, find_tool, run_command, split_command

NO_SELECTOR = "none"  # run without ONEAPI_DEVICE_SELECTOR (e.g. a plain C++ baseline)


def _stats(samples: list[float]) -> dict[str, Any]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(min(samples)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95, method="linear")),
        "max": float(max(samples)),
        "mean": float(arr.mean()),
        "stdev": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
    }


def list_devices(runner: Runner | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (devices, evidence) from sycl-ls. Devices empty if sycl-ls is unavailable."""
    sycl_ls = find_tool("sycl-ls")
    if sycl_ls is None and runner is None:
        return [], {"sycl_ls": None, "failed_command": "sycl-ls", "reason": "sycl-ls not found"}
    res = (runner or run_command)([sycl_ls or "sycl-ls"], timeout=120)
    ev = {"sycl_ls": res.evidence(limit=4000)}
    if res.returncode != 0:
        return [], ev
    return [d.to_dict() for d in parse_sycl_ls(res.stdout)], ev


def benchmark(
    cmd: str | Sequence[str],
    runs: int = 10,
    devices: Sequence[str] | None = None,
    *,
    warmup: int = 1,
    cwd: str | None = None,
    timeout: float = 600.0,
    report_regex: str | None = None,
    runner: Runner | None = None,
) -> ToolResult:
    """Time ``cmd`` ``runs`` times per device.

    ``devices``: ONEAPI_DEVICE_SELECTOR values (e.g. ``["opencl:cpu", "level_zero:0"]``),
    ``["none"]`` for no selector, or None to use every device sycl-ls reports.
    ``report_regex``: optional regex with one group capturing a program-reported time
    (e.g. ``"kernel_ms=([0-9.eE+-]+)"``); it is summarised alongside wall time.
    """
    if runs < 1:
        return ToolResult("benchmark", "error", "runs must be >= 1")
    run = runner or run_command
    argv = split_command(cmd)
    limitations = [
        "Wall-clock time of the whole process (includes start-up, JIT compilation of SYCL "
        "kernels on first use, I/O). Use report_regex for in-program kernel timings.",
        "Device selection uses ONEAPI_DEVICE_SELECTOR; it only takes effect if the program "
        "uses a default/selector-based sycl::queue.",
    ]
    evidence: dict[str, Any] = {"command": " ".join(argv), "runs": runs, "warmup": warmup}

    device_info: dict[str, dict[str, Any]] = {}
    if devices is None:
        found, ls_ev = list_devices(runner)
        evidence["device_discovery"] = ls_ev
        if found:
            selectors = [d["selector"] for d in found]
            device_info = {d["selector"]: d for d in found}
        else:
            selectors = [NO_SELECTOR]
            limitations.append(
                "No SYCL devices discovered (sycl-ls missing or empty); ran without a device "
                "selector, so the device used is whatever the program/runtime defaults to."
            )
    else:
        selectors = list(devices) or [NO_SELECTOR]

    pattern = re.compile(report_regex) if report_regex else None
    results: list[dict[str, Any]] = []
    any_fail = False
    for sel in selectors:
        env = None if sel == NO_SELECTOR else {"ONEAPI_DEVICE_SELECTOR": sel}
        entry: dict[str, Any] = {"selector": sel}
        if sel in device_info:
            entry["device"] = device_info[sel]
        times: list[float] = []
        reported: list[float] = []
        failure = None
        for i in range(warmup + runs):
            res = run(argv, cwd=cwd, env=env, timeout=timeout)
            if res.not_found:
                return ToolResult("benchmark", "error", f"command not found: {argv[0]}",
                                  {**evidence, "failed_command": res.command}, limitations)
            if res.timed_out or res.returncode != 0:
                failure = res.evidence(limit=3000)
                failure["run_index"] = i
                break
            if i >= warmup:
                times.append(res.duration_s)
                if pattern:
                    m = pattern.search(res.stdout)
                    if m:
                        reported.append(float(m.group(1)))
        if failure:
            entry.update(status="fail", failure=failure)
            any_fail = True
        else:
            entry.update(status="ok", wall_time_s=_stats(times), samples_s=[round(t, 6) for t in times])
            if pattern:
                entry["reported"] = _stats(reported) if reported else None
                entry["reported_samples"] = reported
                if len(reported) < len(times):
                    entry["reported_note"] = f"regex matched {len(reported)}/{len(times)} runs"
        results.append(entry)

    evidence["results"] = results
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return ToolResult("benchmark", "fail", f"command failed on all {len(results)} device(s)",
                          evidence, limitations)
    parts = []
    for r in ok:
        name = r.get("device", {}).get("name", r["selector"])
        parts.append(f"{name}: median {r['wall_time_s']['median'] * 1e3:.2f} ms, "
                     f"p95 {r['wall_time_s']['p95'] * 1e3:.2f} ms")
    status: Status = "fail" if any_fail else "ok"
    prefix = "partial: " if any_fail else ""
    return ToolResult("benchmark", status, prefix + f"{runs} run(s) each; " + "; ".join(parts),
                      evidence, limitations)
