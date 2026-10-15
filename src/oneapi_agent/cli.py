"""``oneapi-agent`` command line: the same functions the MCP server exposes.

Prints the ToolResult JSON to stdout. Exit code: 0 = ok, 1 = fail, 2 = anything else.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Sequence

from .result import ToolResult
from .toolchain import toolchain_report
from .tools import benchmark, compile_sycl, migrate, scan_helpers, verify

EXIT = {"ok": 0, "fail": 1}


def _emit(result: ToolResult, summary_only: bool) -> int:
    if summary_only:
        print(f"[{result.tool}] {result.status}: {result.summary}")
    else:
        print(result.to_json())
    return EXIT.get(result.status, 2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oneapi-agent", description=__doc__)
    p.add_argument("--summary", action="store_true", help="print one line instead of full JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="show which toolchain pieces were found")

    m = sub.add_parser("migrate", help="CUDA -> SYCL with SYCLomatic (c2s)")
    m.add_argument("cuda_path")
    m.add_argument("--out", "-o", required=True, dest="out_dir")
    m.add_argument("--in-root")
    m.add_argument("--extra-arg", action="append", default=[],
                   help="passed to c2s as --extra-arg=... (repeatable)")

    c = sub.add_parser("compile", help="icpx -fsycl")
    c.add_argument("sources", nargs="+")
    c.add_argument("--flags", default="", help='e.g. "-O2 -Iinclude"')
    c.add_argument("--output", "-o")
    c.add_argument("--no-dpct-include", action="store_true")

    s = sub.add_parser("scan-helpers", aliases=["scan_helpers"], help="inventory dpct:: helpers")
    s.add_argument("path")

    v = sub.add_parser("verify", help="compare reference vs candidate outputs")
    v.add_argument("--ref", required=True, help="reference command (string)")
    v.add_argument("--cand", required=True, help="candidate command (string)")
    v.add_argument("--input", action="append", default=[], dest="inputs")
    v.add_argument("--tolerance", "--atol", type=float, default=1e-5)
    v.add_argument("--rtol", type=float, default=0.0)
    v.add_argument("--max-ulp", type=int)
    v.add_argument("--mode", choices=["auto", "array", "audio"], default="auto")
    v.add_argument("--threshold-dbfs", type=float, default=-96.0)
    v.add_argument("--work-dir")

    b = sub.add_parser("benchmark", help="time a command per SYCL device")
    b.add_argument("command", help="command string, e.g. './app in.wav out.wav'")
    b.add_argument("--runs", type=int, default=10)
    b.add_argument("--warmup", type=int, default=1)
    b.add_argument("--device", action="append", dest="devices",
                   help="ONEAPI_DEVICE_SELECTOR value (repeatable); 'none' = no selector")
    b.add_argument("--report-regex")

    sub.add_parser("serve", help="run the MCP server on stdio")

    lp = sub.add_parser("loop", help="run the standalone agent loop")
    lp.add_argument("--config", help="JSON task file (see examples/*/task.json)")
    lp.add_argument("--dry-run", action="store_true",
                    help="scripted fake model; no API key needed")
    lp.add_argument("--mock-tools", action="store_true",
                    help="with --dry-run: fake tool results too (no toolchain needed)")
    lp.add_argument("--scenario", default="success", choices=["success", "stuck", "max_iterations"],
                    help="dry-run script to play")
    lp.add_argument("--max-iterations", type=int)
    lp.add_argument("--runs-dir", default="runs")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "doctor":
        rep = toolchain_report()
        missing = [k for k in ("icpx", "c2s", "sycl-ls") if not rep[k]]
        res = ToolResult("doctor", "ok" if not missing else "toolchain_missing",
                         "all tools found" if not missing else f"missing: {', '.join(missing)}",
                         rep)
        return _emit(res, args.summary)
    if args.cmd == "migrate":
        return _emit(migrate(args.cuda_path, args.out_dir, in_root=args.in_root,
                             extra_args=args.extra_arg), args.summary)
    if args.cmd == "compile":
        return _emit(compile_sycl(args.sources, shlex.split(args.flags), output=args.output,
                                  add_dpct_include=not args.no_dpct_include), args.summary)
    if args.cmd in ("scan-helpers", "scan_helpers"):
        return _emit(scan_helpers(args.path), args.summary)
    if args.cmd == "verify":
        return _emit(verify(args.ref, args.cand, args.inputs, args.tolerance, rtol=args.rtol,
                            max_ulp=args.max_ulp, mode=args.mode,
                            threshold_dbfs=args.threshold_dbfs, work_dir=args.work_dir),
                     args.summary)
    if args.cmd == "benchmark":
        return _emit(benchmark(args.command, args.runs, args.devices, warmup=args.warmup,
                               report_regex=args.report_regex), args.summary)
    if args.cmd == "serve":
        from .server import main as serve_main

        serve_main()
        return 0
    if args.cmd == "loop":
        from .loop import cli_entry

        outcome = cli_entry(args)
        print(json.dumps(outcome, indent=2, default=str))
        return 0 if outcome.get("stop_reason") == "success" else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
