#!/usr/bin/env python3
"""GPU phase of scripts/gpu-run.sh: CUDA vs C++ reference, then SYCL-on-NVIDIA vs both.

Usage: gpu_run.py <results_dir>   (env: NVCC, SYCL_CXX; either may be empty)
Writes one JSON file per tool call to <results_dir> and SUMMARY.md next to it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from oneapi_agent.result import ToolResult
from oneapi_agent.toolchain import run_command
from oneapi_agent.tools import benchmark, compile_sycl, verify

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples"
KERNEL_RE = r"kernel_ms=([0-9.eE+-]+)"


def save(results: Path, name: str, r: ToolResult | dict) -> None:
    data = r.to_dict() if isinstance(r, ToolResult) else r
    (results / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))
    line = f"{data.get('status', data.get('returncode'))}: {data.get('summary', data.get('command'))}"
    print(f"[{name}] {line}", flush=True)


def run_example(task_file: Path, results: Path, nvcc: str, sycl_cxx: str) -> list[tuple[str, str, str]]:
    task = json.loads(task_file.read_text())
    name, tdir = task["name"], task_file.parent
    work = results.parent / "work" / name
    (work / "build").mkdir(parents=True, exist_ok=True)
    incs = [f"-I{(tdir / d).resolve()}" for d in task.get("include_dirs", [])]
    inputs = [str((tdir / i).resolve()) for i in task.get("inputs", [])]
    vkw = dict(tolerance=task.get("tolerance", 1e-5), rtol=task.get("rtol", 0.0),
               mode=task.get("mode", "auto"), threshold_dbfs=task.get("threshold_dbfs", -96.0),
               cwd=str(work))
    rows: list[tuple[str, str, str]] = []

    def record(step: str, r: ToolResult | dict) -> ToolResult | dict:
        save(results, f"{name}__{step}", r)
        if isinstance(r, ToolResult):
            rows.append((step, r.status, r.summary))
        else:
            rows.append((step, f"rc={r.get('returncode')}", r.get("command", "")))
        return r

    ref = run_command([a.replace("{task_dir}", str(tdir)) for a in task["reference_build"]],
                      cwd=str(work))
    record("reference_build", ref.evidence())
    if ref.returncode != 0:
        return rows

    def cmd(binary: str) -> str:
        return task["candidate_cmd"].replace("build/candidate", binary)

    first = inputs[0] if inputs else ""
    bench_out = str(work / ("bench_out" + (".wav" if inputs else ".bin")))

    def concrete(c: str) -> str:
        return c.replace("{input}", first).replace("{output}", bench_out)

    cuda_ok = False
    if nvcc:
        cu = run_command([nvcc, "-O2", "-std=c++17", "-arch=native", *incs,
                          str(tdir / task["cuda_path"]), "-o", "build/cuda"], cwd=str(work))
        record("cuda_build", cu.evidence())
        cuda_ok = cu.returncode == 0
    if cuda_ok:
        record("cuda_verify_vs_cpp", verify(task["reference_cmd"], cmd("build/cuda"), inputs,
                                            work_dir=str(work / "v_cuda"), **vkw))
        record("cuda_benchmark", benchmark(concrete(cmd("build/cuda")), runs=10, devices=["none"],
                                           cwd=str(work), report_regex=KERNEL_RE))

    if sycl_cxx:
        src = tdir / "sycl" / (Path(task["cuda_path"]).stem + ".cpp")
        c = record("sycl_gpu_compile", compile_sycl(
            [str(src)], ["-O2", "-fsycl-targets=nvptx64-nvidia-cuda", *incs],
            output=str(work / "build" / "sycl_gpu"), add_dpct_include=False, icpx_path=sycl_cxx))
        if isinstance(c, ToolResult) and c.ok:
            gpu_env = {"ONEAPI_DEVICE_SELECTOR": "cuda:*"}
            record("sycl_verify_vs_cpp", verify(task["reference_cmd"], cmd("build/sycl_gpu"), inputs,
                                                env=gpu_env, work_dir=str(work / "v_sycl"), **vkw))
            if cuda_ok:
                record("sycl_verify_vs_cuda", verify(cmd("build/cuda"), cmd("build/sycl_gpu"), inputs,
                                                     env=gpu_env, work_dir=str(work / "v_sycl_cuda"),
                                                     **vkw))
            record("sycl_benchmark", benchmark(concrete(cmd("build/sycl_gpu")), runs=10,
                                               devices=["cuda:0"], cwd=str(work),
                                               report_regex=KERNEL_RE))
    record("cpp_benchmark", benchmark(concrete(task["reference_cmd"]), runs=10, devices=["none"],
                                      cwd=str(work)))
    return rows


def main() -> int:
    results = Path(sys.argv[1]).resolve()
    results.mkdir(parents=True, exist_ok=True)
    nvcc, sycl_cxx = os.environ.get("NVCC", ""), os.environ.get("SYCL_CXX", "")
    md = ["# GPU run summary (real tool output)", "",
          f"- nvcc: `{nvcc or 'not found'}`", f"- SYCL compiler: `{sycl_cxx or 'not found'}`", ""]
    for task_file in sorted(EX.glob("*/task.json")):
        name = task_file.parent.name
        print(f"\n##### {name}", flush=True)
        try:
            rows = run_example(task_file, results, nvcc, sycl_cxx)
        except Exception as exc:  # keep going: one broken example must not hide the others
            rows = [("exception", "error", f"{type(exc).__name__}: {exc}")]
            print(f"[{name}] EXCEPTION {exc!r}", flush=True)
        md += [f"## {name}", "", "| step | status | summary |", "|---|---|---|"]
        md += [f"| {s} | `{st}` | {str(sm).replace('|', '/')} |" for s, st, sm in rows]
        md.append("")
        (results.parent / "SUMMARY.md").write_text("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
