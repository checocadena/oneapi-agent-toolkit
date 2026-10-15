"""``gpu_test``: verify a SYCL port on whatever GPU this machine has (NVIDIA, AMD or Intel).

Given a task file (the ``examples/*/task.json`` format), it:

1. finds a GPU with ``sycl-ls`` and works out the vendor from its backend
   (``cuda`` -> NVIDIA, ``hip`` -> AMD, ``level_zero`` / ``opencl`` GPU -> Intel);
2. builds the task's C++ reference (the ground truth);
3. builds the ORIGINAL CUDA for this GPU when it can: ``nvcc`` on NVIDIA, ``hipify-perl`` +
   ``hipcc`` on AMD (Intel has no CUDA path, so the reference stands alone there);
4. compiles the SYCL port ahead of time for the vendor's target, verifies it against the
   reference AND the native CUDA/HIP build, and benchmarks it on that one device.

Every step is a real tool call; results go to ``<out_dir>/results/<step>.json``, a
``SUMMARY.md`` and ``summary.json`` (the returned ToolResult), so a run done on another
machine can be brought back with ``ingest_results``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..result import ToolResult
from ..toolchain import Runner, run_command
from .benchmark import benchmark, list_devices
from .compile import compile_sycl
from .verify import verify

KERNEL_RE = r"kernel_ms=([0-9.eE+-]+)"
VENDOR_OF_BACKEND = {"cuda": "nvidia", "hip": "amd", "level_zero": "intel", "opencl": "intel"}
_BACKEND_PREFERENCE = ("cuda", "hip", "level_zero", "opencl")


def pick_device(devices: Sequence[dict[str, Any]], *, vendor: str | None = None,
                selector: str | None = None, allow_cpu: bool = False) -> dict[str, Any] | None:
    """Choose one device: an explicit selector, else the first GPU (optionally of a vendor)."""
    if selector:
        return next((d for d in devices if d["selector"] == selector), None)
    gpus = [d for d in devices if d["device_type"] == "gpu"
            and (vendor is None or VENDOR_OF_BACKEND.get(d["backend"]) == vendor)]
    gpus.sort(key=lambda d: (_BACKEND_PREFERENCE.index(d["backend"])
                             if d["backend"] in _BACKEND_PREFERENCE else 99, d["index"]))
    if gpus:
        return gpus[0]
    if allow_cpu:
        return next((d for d in devices if d["device_type"] == "cpu"), None)
    return None


def target_flags(device: dict[str, Any], amd_arch: str | None = None,
                 cuda_arch: str | None = None) -> list[str]:
    """Ahead-of-time target for the device (JIT-only SPIR-V for CPUs and Intel GPUs).

    On NVIDIA, with the GPU's compute capability known, compile real machine code for exactly
    that GPU (``nvidia_gpu_sm_XY``). Plain ``nvptx64-nvidia-cuda`` ships PTX that the driver
    must JIT, which fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION when the SYCL compiler is
    newer than the driver (seen on a Vultr A16).

    On AMD, use the generic triple plus ``--offload-arch``: the ``amd_gpu_gfxXXX`` shorthand
    stops at gfx942 in intel/llvm, while ``--offload-arch`` takes any arch LLVM knows,
    including gfx950 (MI350X/MI355X).
    """
    if device["device_type"] != "gpu":
        return []
    backend = device["backend"]
    if backend == "cuda":
        return [f"-fsycl-targets=nvidia_gpu_sm_{cuda_arch}" if cuda_arch
                else "-fsycl-targets=nvptx64-nvidia-cuda"]
    if backend == "hip":
        return (["-fsycl-targets=amdgcn-amd-amdhsa", "-Xsycl-target-backend=amdgcn-amd-amdhsa",
                 f"--offload-arch={amd_arch}"] if amd_arch else [])
    return ["-fsycl-targets=spir64"]


def detect_cuda_arch(runner: Runner | None = None, index: int = 0) -> str | None:
    """Compute capability of NVIDIA GPU ``index`` as "86" (from nvidia-smi), or None."""
    if os.environ.get("CUDA_ARCH"):
        return os.environ["CUDA_ARCH"].removeprefix("sm_")
    smi = shutil.which("nvidia-smi")
    if smi is None and runner is None:
        return None
    res = (runner or run_command)([smi or "nvidia-smi", "--query-gpu=compute_cap",
                                   "--format=csv,noheader"], timeout=60)
    caps = re.findall(r"^\s*(\d+)\.(\d+)\s*$", res.stdout, re.M) if res.returncode == 0 else []
    if len(caps) <= index:
        return None
    return caps[index][0] + caps[index][1]


def detect_amd_arch(runner: Runner | None = None) -> str | None:
    if os.environ.get("AMD_ARCH"):
        return os.environ["AMD_ARCH"]
    rocminfo = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    res = (runner or run_command)([rocminfo], timeout=60)
    archs = [a for a in re.findall(r"\bgfx[0-9a-f]+\b", res.stdout) if a != "gfx000"]
    return archs[0] if archs else None


def load_task(task_file: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a task file and resolve its paths against the task's directory."""
    p = Path(task_file).resolve()
    task = json.loads(p.read_text())
    tdir = p.parent
    task["task_dir"] = str(tdir)
    task["name"] = task.get("name", tdir.name)
    task["cuda_path_abs"] = str((tdir / task["cuda_path"]).resolve())
    sycl = task.get("sycl_sources") or [f"sycl/{Path(task['cuda_path']).stem}.cpp"]
    task["sycl_sources_abs"] = [str((tdir / s).resolve()) for s in sycl]
    task["include_dirs_abs"] = [str((tdir / d).resolve()) for d in task.get("include_dirs", [])]
    task["inputs_abs"] = [str((tdir / i).resolve()) for i in task.get("inputs", [])]
    task["base_flags_expanded"] = [os.path.expandvars(f) for f in task.get("base_flags", ["-O2"])]
    return task


def find_hip_include(hipcc: str, roots: tuple[str, ...] = ("/opt/rocm",)) -> Path | None:
    """The include dir holding ``hip/hip_runtime.h`` for this ``hipcc``, or None.

    Looks in hipcc's own ROCm root first (``<root>/bin/hipcc``), then ``$HIP_PATH``,
    ``$ROCM_PATH`` and ``roots`` (each root and its ``core-*`` installs). ROCm 7.14 "core"
    installs without the runtime dev package have hipcc but no headers anywhere.
    """
    cands = [Path(os.path.realpath(hipcc)).parent.parent]
    cands += [Path(os.environ[v]) for v in ("HIP_PATH", "ROCM_PATH") if os.environ.get(v)]
    for r in roots:
        cands += [Path(r), *sorted(Path(r).glob("core-*"), reverse=True)]
    for c in cands:
        if (c / "include" / "hip" / "hip_runtime.h").is_file():
            return c / "include"
    return None


def _native_build(task: dict[str, Any], vendor: str, work: Path, incs: list[str],
                  amd_arch: str | None, runner: Runner | None):
    """Build the original CUDA for this GPU, or return (None, reason)."""
    run = runner or run_command
    src = Path(task["cuda_path_abs"])
    if vendor == "nvidia":
        nvcc = os.environ.get("NVCC") or shutil.which("nvcc") or (
            "/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)
        if not nvcc:
            return None, "nvcc not found"
        return run([nvcc, "-O2", "-std=c++17", "-arch=native", *incs, str(src),
                    "-o", "build/native"], cwd=str(work), timeout=900), None
    if vendor == "amd":
        hipcc = os.environ.get("HIPCC") or shutil.which("hipcc")
        hipify = os.environ.get("HIPIFY") or shutil.which("hipify-perl")
        if not (hipcc and hipify and amd_arch):
            return None, "hipcc, hipify-perl or the gfx arch not found"
        hip_inc = find_hip_include(hipcc)
        if hip_inc is None:
            return None, ("no HIP headers on this host (hip/hip_runtime.h not found next to "
                          f"{hipcc} or under /opt/rocm; ROCm 7.14+ ships them in "
                          "amdrocm-runtime-dev<X.Y>)")
        # hipcc finds headers in its own ROCm root; name any other location explicitly.
        if hip_inc != Path(os.path.realpath(hipcc)).parent.parent / "include":
            incs = [*incs, f"-I{hip_inc}"]
        conv = run([hipify, str(src)], cwd=str(work), timeout=300)
        if conv.returncode != 0:
            return conv, None
        hip_src = work / f"{src.stem}.hip.cpp"
        hip_src.write_text(conv.stdout)
        return run([hipcc, "-O2", "-std=c++17", f"--offload-arch={amd_arch}", *incs,
                    str(hip_src), "-o", "build/native"], cwd=str(work), timeout=900), None
    return None, "no CUDA path on this vendor; the C++ reference is the ground truth"


def gpu_test(task_file: str, out_dir: str, *, vendor: str | None = None,
             device: str | None = None, allow_cpu: bool = False, native: bool = True,
             runs: int = 10, sycl_cxx: str | None = None,
             runner: Runner | None = None) -> ToolResult:
    """Verify a task's SYCL port on this machine's GPU. See the module docstring."""
    out = Path(out_dir).resolve()
    results = out / "results"
    work = out / "work"
    (work / "build").mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    task = load_task(task_file)
    steps: list[dict[str, Any]] = []

    def record(step: str, r: ToolResult | dict[str, Any]) -> None:
        data = r.to_dict() if isinstance(r, ToolResult) else r
        (results / f"{step}.json").write_text(json.dumps(data, indent=2, default=str))
        if isinstance(r, ToolResult):
            steps.append({"step": step, "status": r.status, "summary": r.summary})
        else:
            rc = data.get("returncode")
            steps.append({"step": step, "status": "ok" if rc == 0 else "fail",
                          "summary": f"exit {rc}: {data.get('command', '')}"[:300]})

    devices, dev_ev = list_devices(runner)
    dev = pick_device(devices, vendor=vendor, selector=device, allow_cpu=allow_cpu)
    if dev is None:
        res = ToolResult("gpu_test", "toolchain_missing",
                         "no matching SYCL GPU device found by sycl-ls"
                         + (f" (vendor={vendor})" if vendor else "")
                         + (f" (selector={device})" if device else ""),
                         {"devices": devices, **dev_ev, "task": task["name"]},
                         ["Run `sycl-ls`: a GPU must be listed (cuda:gpu, hip:gpu or "
                          "level_zero:gpu). On a shared system you may be on a login node."])
        return _finish(res, out, steps)
    dev_vendor = VENDOR_OF_BACKEND.get(dev["backend"], "unknown") if dev["device_type"] == "gpu" \
        else "cpu"
    amd_arch = detect_amd_arch(runner) if dev["backend"] == "hip" else None
    cuda_arch = detect_cuda_arch(runner, dev["index"]) if dev["backend"] == "cuda" else None
    selector = dev["selector"]
    gpu_env = {"ONEAPI_DEVICE_SELECTOR": selector}
    incs = [f"-I{d}" for d in task["include_dirs_abs"]]
    inputs = task["inputs_abs"]
    vkw: dict[str, Any] = dict(tolerance=task.get("tolerance", 1e-5), rtol=task.get("rtol", 0.0),
                               mode=task.get("mode", "auto"),
                               threshold_dbfs=task.get("threshold_dbfs", -96.0),
                               cwd=str(work), runner=runner)

    def cmd(binary: str) -> str:
        return task["candidate_cmd"].replace("build/candidate", binary)

    bench_out = str(work / ("bench_out" + (".wav" if inputs else ".bin")))

    def concrete(c: str) -> str:
        return c.replace("{input}", inputs[0] if inputs else "").replace("{output}", bench_out)

    run = runner or run_command
    ref = run([a.replace("{task_dir}", task["task_dir"]) for a in task["reference_build"]],
              cwd=str(work), timeout=900)
    record("reference_build", ref.evidence())
    if ref.returncode != 0:
        res = ToolResult("gpu_test", "error", f"reference build failed: {ref.command}",
                         {"device": dev, "vendor": dev_vendor, "steps": steps})
        return _finish(res, out, steps)

    native_ok, native_note = False, None
    if native and dev["device_type"] == "gpu":
        nb, native_note = _native_build(task, dev_vendor, work, incs, amd_arch, runner)
        if nb is not None:
            record("native_build", nb.evidence())
            native_ok = nb.returncode == 0
        if native_ok:
            record("native_verify_vs_reference",
                   verify(task["reference_cmd"], cmd("build/native"), inputs,
                          work_dir=str(work / "v_native"), **vkw))

    tf = target_flags(dev, amd_arch, cuda_arch)
    comp = compile_sycl(task["sycl_sources_abs"], [*task["base_flags_expanded"], *tf, *incs],
                        output=str(work / "build" / "sycl"), add_dpct_include=False,
                        icpx_path=sycl_cxx or os.environ.get("SYCL_CXX"), runner=runner)
    record("sycl_compile", comp)
    sycl_vs_ref: ToolResult | None = None
    sycl_vs_native: ToolResult | None = None
    if comp.ok:
        sycl_vs_ref = verify(task["reference_cmd"], cmd("build/sycl"), inputs, env=gpu_env,
                             work_dir=str(work / "v_sycl"), **vkw)
        record("sycl_verify_vs_reference", sycl_vs_ref)
        if native_ok:
            sycl_vs_native = verify(cmd("build/native"), cmd("build/sycl"), inputs, env=gpu_env,
                                    work_dir=str(work / "v_sycl_native"), **vkw)
            record("sycl_verify_vs_native", sycl_vs_native)
        record("sycl_benchmark", benchmark(concrete(cmd("build/sycl")), runs=runs,
                                           devices=[selector], cwd=str(work),
                                           report_regex=KERNEL_RE, runner=runner))
        if native_ok:
            record("native_benchmark", benchmark(concrete(cmd("build/native")), runs=runs,
                                                 devices=["none"], cwd=str(work),
                                                 report_regex=KERNEL_RE, runner=runner))

    checks = [r for r in (sycl_vs_ref, sycl_vs_native) if r is not None]
    passed = comp.ok and bool(checks) and all(r.ok for r in checks)
    where = f"{dev['name']} ({selector}, {dev_vendor})"
    if not comp.ok:
        summary = f"SYCL port did not compile for {where}: {comp.summary}"
    else:
        summary = (f"{'PASS' if passed else 'FAIL'} on {where}: "
                   + "; ".join(f"{s['step']}: {s['summary']}" for s in steps
                               if s["step"].startswith("sycl_verify")))
    limitations = []
    if dev["device_type"] != "gpu":
        limitations.append("No GPU was used: this ran on a CPU device (allow_cpu).")
    if native_note and dev["device_type"] == "gpu":
        limitations.append(f"Original CUDA not built here: {native_note}.")
    limitations.append("Benchmarks are single-process wall times plus program-reported kernel "
                       "times; treat them as rough.")
    res = ToolResult("gpu_test", "ok" if passed else "fail", summary,
                     {"task": task["name"], "device": dev, "vendor": dev_vendor,
                      "amd_arch": amd_arch, "cuda_arch": cuda_arch, "target_flags": tf, "selector": selector,
                      "steps": steps, "results_dir": str(results)}, limitations)
    return _finish(res, out, steps)


def _finish(res: ToolResult, out: Path, steps: list[dict[str, Any]]) -> ToolResult:
    (out / "summary.json").write_text(res.to_json())
    dev = res.evidence.get("device") or {}
    md = [f"# GPU test: {res.evidence.get('task', '?')}", "",
          f"- **Result:** `{res.status}`: {res.summary}",
          f"- **Device:** {dev.get('name', 'none')} (`{dev.get('selector', '-')}`), "
          f"vendor `{res.evidence.get('vendor', '-')}`",
          f"- **SYCL target flags:** `{' '.join(res.evidence.get('target_flags') or []) or '(JIT)'}`",
          "", "| step | status | summary |", "|---|---|---|"]
    md += [f"| {s['step']} | `{s['status']}` | {str(s['summary']).replace('|', '/')} |"
           for s in steps]
    if res.limitations:
        md += ["", "Limitations:", *[f"- {x}" for x in res.limitations]]
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n")
    return res
