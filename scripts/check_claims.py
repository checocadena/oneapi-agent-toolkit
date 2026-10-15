#!/usr/bin/env python3
"""Fail loudly when the project's documented claims stop holding on the real toolchain.

Reads the JSON written by ``examples/run_pipeline.py`` (``examples/results/<name>.json``) and,
optionally, a pytest JUnit XML report, and asserts:

* every example under ``examples/*/task.json`` has a fresh result file (no stale committed JSON),
  and its reference build succeeded;
* every hand-written idiomatic port compiles, verifies ``ok`` and has 0 dpct helpers;
* the raw SYCLomatic migration of ``softmax_hecbench`` still maps ``expf`` to reduced-precision
  ``sycl::native::exp`` with no DPCT warning (``native_math`` advisory from ``scan_helpers``), and
  compiles. If a SYCLomatic update changes that, this check fails on purpose so the docs get updated;
* that raw migration FAILS ``verify`` (atol 1e-6) -- asserted when the host CPU has AVX-512.
  The OpenCL CPU runtime's ``native::exp`` precision depends on the ISA it generates code for. On
  the same binary, AVX-512 code generation gives 22,156 mismatches (max err 4.1e-6), while AVX/SSE
  code generation stays within 1e-6 (max err ~5e-7). So on AVX2-only hosts (e.g. AMD EPYC GitHub
  runners) the raw migration passes ``verify``; there the outcome is only reported
  (override with ``--native-exp-verify``);
* (with ``--junit``) pytest ran the toolchain-marked tests and nothing was skipped.

Usage:
  python scripts/check_claims.py [--results DIR] [--not-before ISO8601] [--junit report.xml]
                                 [--native-exp-verify auto|fail|report]
Exit status 0 when every claim holds, 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
# Example whose raw c2s output carries the documented native-math finding.
NATIVE_EXP_EXAMPLE = "softmax_hecbench"
TOOLCHAIN_TEST_MODULE = "test_toolchain_real"


def _utc(text: str) -> dt.datetime:
    """Parse ISO-8601; a naive timestamp is taken as UTC."""
    t = dt.datetime.fromisoformat(text)
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


def host_has_avx512() -> bool:
    """True when this Linux host's CPU reports avx512f (decides native::exp precision on CPU)."""
    try:
        return " avx512f" in Path("/proc/cpuinfo").read_text()
    except OSError:
        return False


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def check(self, cond: bool, what: str, detail: str = "") -> None:
        if cond:
            self.passes.append(what)
            print(f"  PASS  {what}")
        else:
            msg = f"{what}" + (f" -- {detail}" if detail else "")
            self.failures.append(msg)
            print(f"  FAIL  {msg}")

    def note(self, msg: str) -> None:
        print(f"  NOTE  {msg}")


def _step(steps: dict[str, Any], key: str) -> dict[str, Any]:
    v = steps.get(key)
    return v if isinstance(v, dict) else {}


def _desc(s: dict[str, Any]) -> str:
    if not s:
        return "step missing"
    return f"status={s.get('status')!r}: {s.get('summary', '')}"


def check_example(c: Checker, name: str, data: dict[str, Any], expect_native_fail: bool) -> None:
    steps = data.get("steps", {})
    ref = _step(steps, "reference_build")
    c.check(ref.get("returncode") == 0 and "error" not in data,
            f"{name}: reference builds", data.get("error", f"rc={ref.get('returncode')}"))

    comp = _step(steps, "compile_idiomatic")
    c.check(comp.get("status") == "ok", f"{name}: idiomatic port compiles", _desc(comp))
    ver = _step(steps, "verify_idiomatic")
    c.check(ver.get("status") == "ok", f"{name}: idiomatic port verifies ok", _desc(ver))
    scan = _step(steps, "scan_idiomatic")
    total = scan.get("evidence", {}).get("total")
    c.check(scan.get("status") == "ok" and total == 0,
            f"{name}: idiomatic port has 0 dpct helpers", f"total={total!r}; {_desc(scan)}")

    if name != NATIVE_EXP_EXAMPLE:
        return
    mig = _step(steps, "migrate")
    c.check(mig.get("status") == "ok", f"{name}: c2s migration runs", _desc(mig))
    warns = mig.get("evidence", {}).get("warning_counts", {})
    sm = _step(steps, "scan_migrated")
    native = [a for a in sm.get("evidence", {}).get("advisories", []) if a.get("kind") == "native_math"]
    c.check(len(native) > 0 and not warns,
            f"{name}: c2s emits sycl::native:: math with no DPCT warning (documented finding)",
            f"native_math advisories={len(native)}, DPCT warnings={warns}. "
            "If SYCLomatic changed, update examples/README.md and this check.")
    cm = _step(steps, "compile_migrated")
    c.check(cm.get("status") == "ok", f"{name}: raw migrated code compiles", _desc(cm))
    vm = _step(steps, "verify_migrated")
    if expect_native_fail:
        c.check(vm.get("status") == "fail",
                f"{name}: raw migrated code FAILS verify (documented finding)",
                "expected status 'fail'; got " + _desc(vm))
    else:
        c.note(f"{name}: raw migrated verify not asserted here (native::exp precision is "
               f"ISA-dependent; see --native-exp-verify): {_desc(vm)}")


def check_results(c: Checker, results: Path, not_before: dt.datetime | None,
                  expect_native_fail: bool) -> None:
    names = sorted(p.parent.name for p in EXAMPLES.glob("*/task.json"))
    c.check(bool(names), "examples found", f"no */task.json under {EXAMPLES}")
    for name in names:
        print(f"== {name}")
        f = results / f"{name}.json"
        if not f.is_file():
            c.check(False, f"{name}: result file exists", str(f))
            continue
        data = json.loads(f.read_text())
        if not_before is not None:
            stamp = _utc(data.get("env", {}).get("date_utc", "1970-01-01T00:00:00"))
            c.check(stamp >= not_before, f"{name}: result is fresh",
                    f"generated {stamp.isoformat()} < {not_before.isoformat()}")
        check_example(c, name, data, expect_native_fail)


def check_junit(c: Checker, junit: Path) -> None:
    print(f"== pytest ({junit.name})")
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tot = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    cases = list(root.iter("testcase"))
    tc_cases = [t for t in cases if TOOLCHAIN_TEST_MODULE in (t.get("classname") or "")]
    tc_skipped = [t.get("name") for t in tc_cases if t.find("skipped") is not None]
    c.check(tot["failures"] == 0 and tot["errors"] == 0, "pytest: no failures/errors", str(tot))
    c.check(len(tc_cases) > 0, "pytest: toolchain-marked tests were collected",
            f"no testcase from {TOOLCHAIN_TEST_MODULE}")
    c.check(not tc_skipped, f"pytest: all {len(tc_cases)} toolchain tests ran (none skipped)",
            f"skipped: {tc_skipped}")
    c.check(tot["skipped"] == 0, "pytest: nothing skipped at all", f"{tot['skipped']} skipped")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=EXAMPLES / "results",
                    help="directory with <name>.json from run_pipeline.py")
    ap.add_argument("--not-before", type=_utc, default=None,
                    help="reject result files generated before this ISO-8601 time (stale JSON)")
    ap.add_argument("--junit", type=Path, default=None, help="pytest --junitxml report to check")
    ap.add_argument("--native-exp-verify", choices=["auto", "fail", "report"], default="auto",
                    help="raw softmax migration verify: 'fail' asserts it fails, 'report' only prints "
                         "it, 'auto' (default) asserts on AVX-512 hosts and reports elsewhere")
    args = ap.parse_args(argv)

    avx512 = host_has_avx512()
    expect = args.native_exp_verify == "fail" or (args.native_exp_verify == "auto" and avx512)
    print(f"host AVX-512: {avx512}; raw {NATIVE_EXP_EXAMPLE} verify failure asserted: {expect}")
    c = Checker()
    check_results(c, args.results, args.not_before, expect)
    if args.junit is not None:
        check_junit(c, args.junit)
    print(f"\n{len(c.passes)} claim(s) hold, {len(c.failures)} broken")
    for f in c.failures:
        print(f"BROKEN: {f}")
    return 1 if c.failures else 0


if __name__ == "__main__":
    sys.exit(main())
