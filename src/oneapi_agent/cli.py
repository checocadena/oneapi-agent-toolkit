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
from .tools import benchmark, compile_sycl, gpu_test, migrate, scan_helpers, verify

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

    g = sub.add_parser("gpu-test", help="verify a task's SYCL port on this machine's GPU "
                       "(NVIDIA, AMD or Intel), against the C++ reference and the original CUDA")
    g.add_argument("task", help="task.json (examples/*/task.json format)")
    g.add_argument("--out", default="gputest-out", help="output dir (results/, SUMMARY.md)")
    g.add_argument("--vendor", choices=["nvidia", "amd", "intel"], help="only use this vendor's GPU")
    g.add_argument("--device", help="exact ONEAPI_DEVICE_SELECTOR value, e.g. level_zero:0")
    g.add_argument("--allow-cpu", action="store_true", help="fall back to a CPU device (testing)")
    g.add_argument("--no-native", action="store_true", help="skip building the original CUDA")
    g.add_argument("--runs", type=int, default=10)

    bu = sub.add_parser("bundle", help="pack a task + this toolkit into one .sh to run gpu-test "
                        "on a machine this tool cannot reach (SSH box, Jupyter, CI)")
    bu.add_argument("task")
    bu.add_argument("-o", "--output", help="bundle path (default gputest-<name>.sh)")

    ig = sub.add_parser("ingest", help="import a gpu-test results archive from a bundle run")
    ig.add_argument("archive")
    ig.add_argument("--into", default="docs/gpu-run")

    rt = sub.add_parser("remote-test", help="rent a Runpod GPU (NVIDIA or AMD), run gpu-test "
                        "there, import the results, delete the pod (needs RUNPOD_API_KEY)")
    rt.add_argument("task")
    rt.add_argument("--provider", choices=["runpod", "vast", "vultr", "digitalocean"],
                    default="runpod",
                    help="GPU rental service; its API key comes from RUNPOD_API_KEY, "
                         "VAST_API_KEY, VULTR_API_KEY or OCEAN_API_KEY (or DIGITALOCEAN_TOKEN)")
    rt.add_argument("--vendor", choices=["nvidia", "amd"], default="nvidia")
    rt.add_argument("--gpu", dest="gpu_type", help="Runpod GPU type id (default per vendor)")
    rt.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="SECURE")
    rt.add_argument("--image", help="container image (default per vendor)")
    rt.add_argument("--sycl-url", help="SYCL compiler tarball to unpack to /opt/sycl on the "
                    "machine (default for AMD: the HIP-enabled intel/llvm build published by "
                    "this repo's sycl-hip-toolchain workflow)")
    rt.add_argument("--sycl-sha256", help="expected sha256 of --sycl-url")
    rt.add_argument("--max-minutes", type=float, default=45)
    rt.add_argument("--max-usd", type=float, default=2.0)
    rt.add_argument("--into", default="docs/gpu-run")
    rt.add_argument("--keep-pod", action="store_true", help="do not delete the pod (debugging)")
    rt.add_argument("--boot-minutes", type=float, default=15,
                    help="replace a machine whose agent has not answered after this long")

    sub.add_parser("serve", help="run the MCP server on stdio")

    st = sub.add_parser("setup", help="register the MCP server (and skill) with coding agents")
    st.add_argument("--agent", action="append", dest="agents",
                    choices=["claude", "codex", "cursor", "gemini"],
                    help="agent to configure (repeatable); default: every agent detected")
    st.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    st.add_argument("--no-skill", action="store_true", help="do not install the oneapi-port skill")

    lp = sub.add_parser("loop", help="run the standalone agent loop")
    lp.add_argument("--config", help="JSON task file (see examples/*/task.json)")
    lp.add_argument("--dry-run", action="store_true",
                    help="scripted fake model; no API key needed")
    lp.add_argument("--mock-tools", action="store_true",
                    help="with --dry-run: fake tool results too (no toolchain needed)")
    lp.add_argument("--scenario", default="success", choices=["success", "stuck", "max_iterations"],
                    help="dry-run script to play")
    lp.add_argument("--max-iterations", type=int)
    lp.add_argument("--baseline", action="store_true",
                    help="ablation: no scan_helpers/verify/benchmark; run ends when the model "
                         "calls finish")
    lp.add_argument("--shell", action="store_true",
                    help="also give the model a bash shell in the work dir (`run` tool)")
    lp.add_argument("--evaluate", action="store_true",
                    help="grade the final sources with a hidden compile + verify + scan")
    lp.add_argument("--runs-dir", default="runs")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:  # remote-test has already deleted its machine in a finally block
        print("[oneapi-agent] interrupted", file=sys.stderr)
        return 130


def _main(argv: Sequence[str] | None = None) -> int:
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
    if args.cmd == "gpu-test":
        return _emit(gpu_test(args.task, args.out, vendor=args.vendor, device=args.device,
                              allow_cpu=args.allow_cpu, native=not args.no_native,
                              runs=args.runs), args.summary)
    if args.cmd in ("bundle", "ingest"):
        from .bundle import ingest_results, make_bundle

        return _emit(make_bundle(args.task, args.output) if args.cmd == "bundle"
                     else ingest_results(args.archive, args.into), args.summary)
    if args.cmd == "remote-test":
        from .remote import remote_test

        return _emit(remote_test(args.task, vendor=args.vendor, gpu_type=args.gpu_type,
                                 provider=args.provider,
                                 cloud=args.cloud, image=args.image, sycl_url=args.sycl_url,
                                 sycl_sha256=args.sycl_sha256, max_minutes=args.max_minutes,
                                 max_usd=args.max_usd, into=args.into, keep_pod=args.keep_pod,
                                 boot_minutes=args.boot_minutes,
                                 log=lambda m: print(m, file=sys.stderr)), args.summary)
    if args.cmd == "setup":
        from .setup import setup

        return _emit(setup(args.agents, dry_run=args.dry_run, skill=not args.no_skill),
                     args.summary)
    if args.cmd == "serve":
        from .server import main as serve_main

        serve_main()
        return 0
    if args.cmd == "loop":
        from .loop import cli_entry

        outcome = cli_entry(args)
        print(json.dumps(outcome, indent=2, default=str))
        if args.baseline:
            return 0 if outcome.get("stop_reason") == "model_finished" else 1
        return 0 if outcome.get("stop_reason") == "success" else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
