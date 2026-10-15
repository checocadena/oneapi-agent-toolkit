#!/usr/bin/env python3
"""Fail loudly when the project's documented claims stop holding on the real toolchain.

Reads the JSON written by ``examples/run_pipeline.py`` (``examples/results/<name>.json``) and,
optionally, a pytest JUnit XML report, and asserts:

* every example under ``examples/*/task.json`` has a fresh result file (no stale committed JSON),
  and its reference build succeeded;
* every hand-written idiomatic port compiles, verifies ``ok`` and has 0 dpct helpers;
* the raw SYCLomatic migration of ``softmax_hecbench`` compiles but FAILS ``verify`` -- the
  documented c2s finding (``expf`` -> ``sycl::native::exp``). If a SYCLomatic update fixes it,
  this check fails on purpose so the docs get updated;
* (with ``--junit``) pytest ran the toolchain-marked tests and nothing was skipped.

Usage:
  python scripts/check_claims.py [--results DIR] [--not-before ISO8601] [--junit report.xml]
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
# Raw-migration outcomes that are documented as findings and asserted here.
EXPECTED_MIGRATED_VERIFY_FAIL = {"softmax_hecbench"}
TOOLCHAIN_TEST_MODULE = "test_toolchain_real"


def _utc(text: str) -> dt.datetime:
    """Parse ISO-8601; a naive timestamp is taken as UTC."""
    t = dt.datetime.fromisoformat(text)
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


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


def _step(steps: dict[str, Any], key: str) -> dict[str, Any]:
    v = steps.get(key)
    return v if isinstance(v, dict) else {}


def _desc(s: dict[str, Any]) -> str:
    if not s:
        return "step missing"
    return f"status={s.get('status')!r}: {s.get('summary', '')}"


def check_example(c: Checker, name: str, data: dict[str, Any]) -> None:
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

    if name in EXPECTED_MIGRATED_VERIFY_FAIL:
        mig = _step(steps, "migrate")
        c.check(mig.get("status") == "ok", f"{name}: c2s migration runs", _desc(mig))
        cm = _step(steps, "compile_migrated")
        c.check(cm.get("status") == "ok", f"{name}: raw migrated code compiles", _desc(cm))
        vm = _step(steps, "verify_migrated")
        c.check(vm.get("status") == "fail",
                f"{name}: raw migrated code FAILS verify (documented SYCLomatic finding)",
                "expected status 'fail'; got " + _desc(vm)
                + ". If SYCLomatic changed, update examples/README.md and this check.")


def check_results(c: Checker, results: Path, not_before: dt.datetime | None) -> None:
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
        check_example(c, name, data)


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
    args = ap.parse_args(argv)

    c = Checker()
    check_results(c, args.results, args.not_before)
    if args.junit is not None:
        check_junit(c, args.junit)
    print(f"\n{len(c.passes)} claim(s) hold, {len(c.failures)} broken")
    for f in c.failures:
        print(f"BROKEN: {f}")
    return 1 if c.failures else 0


if __name__ == "__main__":
    sys.exit(main())
