#!/usr/bin/env python3
"""GPU phase of scripts/{gpu,amd-gpu,intel-gpu}-run.sh: native GPU build vs C++ reference, then
SYCL-on-GPU vs both.

Usage: gpu_run.py <results_dir>
Env (all optional):
  GPU_VENDOR          nvidia (default) | amd | intel
  NVCC                nvcc for the native CUDA build (nvidia)
  HIPCC, HIPIFY       hipcc + hipify-perl: the CUDA source is hipified and built (amd)
  AMD_ARCH            offload arch for hipcc, e.g. gfx942
  SYCL_CXX            SYCL compiler (icpx or intel/llvm clang++)
  SYCL_TARGET_FLAGS   default per vendor (-fsycl-targets=nvptx64-nvidia-cuda, amd_gpu_<arch>, spir64)
  SYCL_DEVICE_SELECTOR default per vendor (cuda:*, hip:*, level_zero:gpu)
  EXAMPLES            space-separated example names (default: every examples/*/task.json except
                      gemm_cublas, whose oneMath port is built for the MKL CPU backend)
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
VENDOR = os.environ.get("GPU_VENDOR", "nvidia")
AMD_ARCH = os.environ.get("AMD_ARCH", "gfx942")
SYCL_TARGET_FLAGS = os.environ.get("SYCL_TARGET_FLAGS") or {
    "nvidia": "-fsycl-targets=nvptx64-nvidia-cuda",
    "amd": f"-fsycl-targets=amd_gpu_{AMD_ARCH}",
    "intel": "-fsycl-targets=spir64",
}[VENDOR]
SELECTOR = os.environ.get("SYCL_DEVICE_SELECTOR") or {
    "nvidia": "cuda:*", "amd": "hip:*", "intel": "level_zero:gpu"}[VENDOR]
NATIVE = {"nvidia": "cuda", "amd": "hip", "intel": ""}[VENDOR]  # step-name prefix


def save(results: Path, name: str, r: ToolResult | dict) -> None:
    data = r.to_dict() if isinstance(r, ToolResult) else r
    (results / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))
    line = f"{data.get('status', data.get('returncode'))}: {data.get('summary', data.get('command'))}"
    print(f"[{name}] {line}", flush=True)


def native_build(task: dict, tdir: Path, work: Path, incs: list[str], nvcc: str):
    """Build the original CUDA source for this vendor's GPU: nvcc, or hipify + hipcc on AMD."""
    src = tdir / task["cuda_path"]
    if VENDOR == "nvidia" and nvcc:
        return run_command([nvcc, "-O2", "-std=c++17", "-arch=native", *incs, str(src),
                            "-o", "build/native"], cwd=str(work))
    hipcc, hipify = os.environ.get("HIPCC", ""), os.environ.get("HIPIFY", "")
    if VENDOR == "amd" and hipcc and hipify:
        hip_src = work / (src.stem + ".hip.cpp")
        conv = run_command([hipify, str(src)], cwd=str(work))
        if conv.returncode != 0:
            return conv
        hip_src.write_text(conv.stdout)
        return run_command([hipcc, "-O2", "-std=c++17", f"--offload-arch={AMD_ARCH}", *incs,
                            str(hip_src), "-o", "build/native"], cwd=str(work))
    return None


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
    nb = native_build(task, tdir, work, incs, nvcc) if NATIVE else None
    if nb is not None:
        record(f"{NATIVE}_build", nb.evidence())
        cuda_ok = nb.returncode == 0
    if cuda_ok:
        record(f"{NATIVE}_verify_vs_cpp", verify(task["reference_cmd"], cmd("build/native"),
                                                 inputs, work_dir=str(work / "v_native"), **vkw))
        record(f"{NATIVE}_benchmark", benchmark(concrete(cmd("build/native")), runs=10,
                                                devices=["none"], cwd=str(work),
                                                report_regex=KERNEL_RE))

    if sycl_cxx:
        src = tdir / "sycl" / (Path(task["cuda_path"]).stem + ".cpp")
        c = record("sycl_gpu_compile", compile_sycl(
            [str(src)], ["-O2", *SYCL_TARGET_FLAGS.split(), *incs],
            output=str(work / "build" / "sycl_gpu"), add_dpct_include=False, icpx_path=sycl_cxx))
        if isinstance(c, ToolResult) and c.ok:
            gpu_env = {"ONEAPI_DEVICE_SELECTOR": SELECTOR}
            record("sycl_verify_vs_cpp", verify(task["reference_cmd"], cmd("build/sycl_gpu"), inputs,
                                                env=gpu_env, work_dir=str(work / "v_sycl"), **vkw))
            if cuda_ok:
                record(f"sycl_verify_vs_{NATIVE}", verify(cmd("build/native"), cmd("build/sycl_gpu"),
                                                          inputs, env=gpu_env,
                                                          work_dir=str(work / "v_sycl_native"),
                                                          **vkw))
            record("sycl_benchmark", benchmark(concrete(cmd("build/sycl_gpu")), runs=10,
                                               devices=[SELECTOR.replace("*", "0")], cwd=str(work),
                                               report_regex=KERNEL_RE))
    record("cpp_benchmark", benchmark(concrete(task["reference_cmd"]), runs=10, devices=["none"],
                                      cwd=str(work)))
    return rows


def main() -> int:
    results = Path(sys.argv[1]).resolve()
    results.mkdir(parents=True, exist_ok=True)
    nvcc, sycl_cxx = os.environ.get("NVCC", ""), os.environ.get("SYCL_CXX", "")
    md = ["# GPU run summary (real tool output)", "",
          f"- vendor: `{VENDOR}`; SYCL flags: `{SYCL_TARGET_FLAGS}`; selector: `{SELECTOR}`",
          f"- nvcc: `{nvcc or 'not found'}`; hipcc: `{os.environ.get('HIPCC') or 'not used'}`",
          f"- SYCL compiler: `{sycl_cxx or 'not found'}`", ""]
    wanted = os.environ.get("EXAMPLES", "").split()
    tasks = [t for t in sorted(EX.glob("*/task.json"))
             if (t.parent.name in wanted if wanted else t.parent.name != "gemm_cublas")]
    for task_file in tasks:
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
