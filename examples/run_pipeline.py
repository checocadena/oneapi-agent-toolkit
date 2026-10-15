#!/usr/bin/env python3
"""Run the full evidence pipeline on every example with the real tools.

For each examples/<name>/task.json:
  1. build the plain C++ reference (g++)
  2. migrate the CUDA source with c2s
  3. compile the raw migrated (dpct) code, verify it, scan its helpers
  4. compile the hand-written idiomatic port in <name>/sycl/ WITHOUT dpct headers,
     verify it, scan it, benchmark it (and the reference, for context)

Every ToolResult is saved to examples/results/<name>.json and summarised in
examples/RESULTS.md. Nothing is edited by hand: if a step fails, the failure is recorded.
Usage: python examples/run_pipeline.py [name ...]
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oneapi_agent.result import ToolResult  # noqa: E402
from oneapi_agent.toolchain import run_command, toolchain_report  # noqa: E402
from oneapi_agent.tools import (  # noqa: E402
    benchmark,
    compile_sycl,
    list_devices,
    migrate,
    scan_helpers,
    verify,
)

EX = Path(__file__).resolve().parent
RESULTS = EX / "results"
WORK = RESULTS / "work"


def run_task(task_file: Path) -> dict:
    task = json.loads(task_file.read_text())
    tdir = task_file.parent
    name = task["name"]
    work = WORK / name
    if work.exists():
        shutil.rmtree(work)
    (work / "build").mkdir(parents=True)
    includes = [str((tdir / d).resolve()) for d in task.get("include_dirs", [])]
    inputs = [str((tdir / i).resolve()) for i in task.get("inputs", [])]
    steps: dict[str, ToolResult | dict] = {}

    ref_argv = [a.replace("{task_dir}", str(tdir)) for a in task["reference_build"]]
    ref = run_command(ref_argv, cwd=str(work))
    steps["reference_build"] = ref.evidence()
    if ref.returncode != 0:
        return {"name": name, "steps": steps, "error": "reference build failed"}

    def cand_cmd(binary: str) -> str:
        return task["candidate_cmd"].replace("build/candidate", binary)

    vkw = dict(tolerance=task.get("tolerance", 1e-5), rtol=task.get("rtol", 0.0),
               mode=task.get("mode", "auto"), threshold_dbfs=task.get("threshold_dbfs", -96.0),
               cwd=str(work))

    # --- raw SYCLomatic output
    mig_dir = work / "migrated"
    steps["migrate"] = migrate(str(tdir / task["cuda_path"]), str(mig_dir),
                               extra_args=[f"-I{d}" for d in includes])
    main_src = mig_dir / (Path(task["cuda_path"]).stem + ".dp.cpp")
    # base_flags: flags for the idiomatic port (and the agent loop); migrated_flags: for the raw
    # c2s output when it needs different libraries (e.g. oneMKL instead of oneMath).
    # ${VARS} are expanded from the environment (see scripts/oneapi-env.sh).
    base = [os.path.expandvars(f) for f in task.get("base_flags", ["-O2"])]
    mig_base = [os.path.expandvars(f) for f in task.get("migrated_flags", base)]
    inc_flags = [f"-I{d}" for d in includes]
    flags = [*base, *inc_flags]
    if main_src.exists():
        steps["compile_migrated"] = compile_sycl([str(main_src)], [*mig_base, *inc_flags],
                                                 output=str(work / "build" / "migrated"))
        if steps["compile_migrated"].ok:
            steps["verify_migrated"] = verify(task["reference_cmd"], cand_cmd("build/migrated"),
                                              inputs, work_dir=str(work / "v_migrated"), **vkw)
        steps["scan_migrated"] = scan_helpers(str(mig_dir))

    # --- hand-written idiomatic port
    sycl_src = tdir / "sycl" / (Path(task["cuda_path"]).stem + ".cpp")
    steps["compile_idiomatic"] = compile_sycl([str(sycl_src)], flags,
                                              output=str(work / "build" / "idiomatic"),
                                              add_dpct_include=False)
    if steps["compile_idiomatic"].ok:
        steps["verify_idiomatic"] = verify(task["reference_cmd"], cand_cmd("build/idiomatic"),
                                           inputs, work_dir=str(work / "v_idiomatic"), **vkw)
        first = inputs[0] if inputs else ""
        out = str(work / ("bench_out" + (".wav" if inputs else ".bin")))

        def concrete(cmd: str) -> str:
            return cmd.replace("{input}", first).replace("{output}", out)

        steps["benchmark_idiomatic"] = benchmark(concrete(cand_cmd("build/idiomatic")), runs=5,
                                                 cwd=str(work), report_regex=r"kernel_ms=([0-9.eE+-]+)")
        steps["benchmark_reference"] = benchmark(concrete(task["reference_cmd"]), runs=5,
                                                 devices=["none"], cwd=str(work))
    steps["scan_idiomatic"] = scan_helpers(str(sycl_src))
    return {"name": name, "steps": steps}


def as_dict(v: ToolResult | dict) -> dict:
    return v.to_dict() if isinstance(v, ToolResult) else v


def main() -> int:
    names = sys.argv[1:] or [p.parent.name for p in sorted(EX.glob("*/task.json"))]
    RESULTS.mkdir(exist_ok=True)
    env = {
        "date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "toolchain": toolchain_report(),
        "devices": list_devices()[0],
    }
    icpx_v = run_command([env["toolchain"]["icpx"] or "icpx", "--version"])
    c2s_v = run_command([env["toolchain"]["c2s"] or "c2s", "--version"])
    env["icpx_version"] = icpx_v.stdout.splitlines()[0] if icpx_v.stdout else icpx_v.stderr[:200]
    env["c2s_version"] = (c2s_v.stdout.strip().splitlines() or [c2s_v.stderr[:200]])[0]

    md = ["# Example pipeline results (real tool output)", "",
          f"Generated by `examples/run_pipeline.py` on {env['date_utc']}.",
          f"- Platform: `{env['platform']}`",
          f"- icpx: `{env['icpx_version']}`",
          f"- c2s: `{env['c2s_version']}`",
          "- SYCL devices: " + (", ".join(f"`{d['selector']}` {d['name']}" for d in env["devices"])
                                or "none found"),
          "",
          "Each row is one tool call; the status and summary come straight from the ToolResult. "
          "Full evidence: `examples/results/<name>.json`.",
          "The *idiomatic* ports in `<name>/sycl/` were hand-written (they are the target end "
          "state, not output of the agent loop).", ""]
    for n in names:
        print(f"== {n}", flush=True)
        res = run_task(EX / n / "task.json")
        data = {"env": env, "name": n, **{k: v for k, v in res.items() if k != "steps"},
                "steps": {k: as_dict(v) for k, v in res["steps"].items()}}
        (RESULTS / f"{n}.json").write_text(json.dumps(data, indent=2, default=str))
        md += [f"## {n}", "", "| step | status | summary |", "|---|---|---|"]
        for k, v in res["steps"].items():
            if isinstance(v, ToolResult):
                summ = v.summary.replace("|", "\\|")
                md.append(f"| {k} | `{v.status}` | {summ} |")
                print(f"  {k}: {v.status}: {v.summary}", flush=True)
            else:
                md.append(f"| {k} | rc={v.get('returncode')} | `{v.get('command')}` |")
        md.append("")
    (EX / "RESULTS.md").write_text("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
