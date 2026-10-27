"""gpu_test device selection, bundle packing and results ingest (no GPU or toolchain needed)."""

from __future__ import annotations

import base64
import importlib
import io
import json
import tarfile
from pathlib import Path

from oneapi_agent.bundle import ingest_results, make_bundle
from oneapi_agent.toolchain import CommandResult
from oneapi_agent.tools.gpu_test import gpu_test, pick_device, target_flags

REPO = Path(__file__).resolve().parents[1]


def dev(backend, kind, index=0, name="X"):
    return {"backend": backend, "device_type": kind, "index": index, "name": name,
            "selector": f"{backend}:{index}"}


DEVICES = [dev("opencl", "cpu", 0, "Xeon"), dev("level_zero", "gpu", 0, "Max 1100"),
           dev("hip", "gpu", 0, "MI300X"), dev("cuda", "gpu", 0, "RTX")]


def test_pick_device_prefers_gpus_and_honours_vendor_and_selector():
    assert pick_device(DEVICES)["backend"] == "cuda"
    assert pick_device(DEVICES, vendor="amd")["name"] == "MI300X"
    assert pick_device(DEVICES, vendor="intel")["selector"] == "level_zero:0"
    assert pick_device(DEVICES, selector="opencl:0")["name"] == "Xeon"
    cpu_only = [DEVICES[0]]
    assert pick_device(cpu_only) is None
    assert pick_device(cpu_only, allow_cpu=True)["name"] == "Xeon"


def test_target_flags_per_vendor():
    assert target_flags(dev("cuda", "gpu")) == ["-fsycl-targets=nvptx64-nvidia-cuda"]
    # exact SM -> machine code, immune to "unsupported PTX version" on older drivers
    assert target_flags(dev("cuda", "gpu"), cuda_arch="86") == ["-fsycl-targets=nvidia_gpu_sm_86"]
    # generic triple + --offload-arch: intel/llvm has no amd_gpu_gfx950 shorthand
    for arch in ("gfx942", "gfx950"):
        assert target_flags(dev("hip", "gpu"), arch) == [
            "-fsycl-targets=amdgcn-amd-amdhsa", "-Xsycl-target-backend=amdgcn-amd-amdhsa",
            f"--offload-arch={arch}"]
    assert target_flags(dev("level_zero", "gpu")) == ["-fsycl-targets=spir64"]
    assert target_flags(dev("opencl", "cpu")) == []


def test_no_gpu_is_reported_not_faked(tmp_path, monkeypatch):
    bm = importlib.import_module("oneapi_agent.tools.benchmark")
    monkeypatch.setattr(bm, "find_tool", lambda n: "/x/sycl-ls")

    def runner(argv, **kw):
        out = "[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) OpenCL 3.0 [x]\n"
        return CommandResult(list(argv), 0, out, "", 0.0)

    r = gpu_test(str(REPO / "examples/saxpy/task.json"), str(tmp_path / "o"), runner=runner)
    assert r.status == "toolchain_missing" and "no matching SYCL GPU" in r.summary
    assert json.loads((tmp_path / "o" / "summary.json").read_text())["status"] == "toolchain_missing"


def _payload(bundle: Path) -> tarfile.TarFile:
    text = bundle.read_text()
    b64 = text.split("__PAYLOAD_BELOW__\n", 2)[-1]
    return tarfile.open(fileobj=io.BytesIO(base64.decodebytes(b64.encode())), mode="r:gz")


def test_bundle_keeps_relative_layout_and_ships_the_package(tmp_path):
    r = make_bundle(str(REPO / "examples/saxpy/task.json"), str(tmp_path / "b.sh"))
    assert r.status == "ok", r.summary
    script = (tmp_path / "b.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash") and "gpu-test" in script
    names = _payload(tmp_path / "b.sh").getnames()
    # saxpy has no shared include dirs, so the task dir itself is the bundle's task/ root
    assert "task/task.json" in names and "task/sycl/saxpy.cpp" in names
    assert '"task/task.json"' in script


def test_bundle_preserves_sibling_dirs(tmp_path):
    # A task that uses ../common and ../data (like examples/fir) keeps that layout.
    root = tmp_path / "proj"
    for d in ("app/sycl", "common", "data"):
        (root / d).mkdir(parents=True)
    (root / "app/k.cu").write_text("// cuda")
    (root / "app/sycl/k.cpp").write_text("// sycl")
    (root / "common/h.hpp").write_text("// header")
    (root / "data/in.bin").write_bytes(b"\0" * 8)
    (root / "app/task.json").write_text(json.dumps({
        "cuda_path": "k.cu", "include_dirs": ["../common"], "inputs": ["../data/in.bin"],
        "reference_build": ["true"], "reference_cmd": "r", "candidate_cmd": "c"}))
    r = make_bundle(str(root / "app/task.json"), str(tmp_path / "f.sh"))
    assert r.status == "ok", r.summary
    names = _payload(tmp_path / "f.sh").getnames()
    for n in ("task/app/task.json", "task/app/sycl/k.cpp", "task/common/h.hpp",
              "task/data/in.bin"):
        assert n in names, n
    assert '"task/app/task.json"' in (tmp_path / "f.sh").read_text()
    assert "pkg/oneapi_agent/tools/gpu_test.py" in names
    assert not any("__pycache__" in n for n in names)


def test_bundle_reports_missing_files(tmp_path):
    t = tmp_path / "task.json"
    t.write_text(json.dumps({"cuda_path": "nope.cu", "reference_build": ["true"],
                             "reference_cmd": "r", "candidate_cmd": "c"}))
    assert make_bundle(str(t), str(tmp_path / "b.sh")).status == "error"


def _archive(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


def test_ingest_places_results_by_vendor_and_device(tmp_path):
    summary = {"tool": "gpu_test", "status": "ok", "summary": "PASS",
               "evidence": {"task": "saxpy", "vendor": "amd", "device": {"name": "MI300X"}}}
    _archive(tmp_path / "r.tar.gz", {"summary.json": json.dumps(summary).encode(),
                                     "SUMMARY.md": b"# x", "results/a.json": b"{}"})
    r = ingest_results(str(tmp_path / "r.tar.gz"), str(tmp_path / "into"))
    assert r.status == "ok"
    assert (tmp_path / "into" / "amd-mi300x" / "saxpy" / "results" / "a.json").exists()


def test_ingest_rejects_path_traversal(tmp_path):
    _archive(tmp_path / "e.tar.gz", {"summary.json": b"{}", "../escape": b"x"})
    r = ingest_results(str(tmp_path / "e.tar.gz"), str(tmp_path / "into"))
    assert r.status == "error" and "unsafe" in r.summary
    assert not (tmp_path / "escape").exists()



def test_bundle_crash_still_yields_an_explaining_archive(tmp_path):
    import subprocess

    r = make_bundle(str(REPO / "examples/saxpy/task.json"), str(tmp_path / "b.sh"))
    assert r.status == "ok"
    # An unknown flag makes gpu-test exit before it writes summary.json.
    p = subprocess.run(["bash", str(tmp_path / "b.sh"), "--no-such-flag"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode != 0
    archives = list(tmp_path.glob("gputest-results-saxpy-*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0]) as t:
        summary = json.loads(t.extractfile("summary.json").read())
        assert "run.log" in t.getnames()
    assert summary["status"] == "error" and "crashed" in summary["summary"]
    assert "no-such-flag" in summary["evidence"]["run_log_tail"]


def test_bundle_runs_without_home(tmp_path):
    # cloud-init (Vultr, DigitalOcean) starts the agent with no HOME; the runner uses `set -u`.
    import os
    import subprocess

    make_bundle(str(REPO / "examples/saxpy/task.json"), str(tmp_path / "b.sh"))
    env = {k: v for k, v in os.environ.items() if k != "HOME"}
    p = subprocess.run(["bash", str(tmp_path / "b.sh"), "--no-such-flag"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=300)
    assert "unbound variable" not in p.stdout + p.stderr



def test_detect_cuda_arch_parses_nvidia_smi(monkeypatch):
    from oneapi_agent.tools.gpu_test import detect_cuda_arch

    monkeypatch.delenv("CUDA_ARCH", raising=False)

    def runner(argv, **kw):
        return CommandResult(list(argv), 0, "8.6\n9.0\n", "", 0.0)

    assert detect_cuda_arch(runner, 0) == "86" and detect_cuda_arch(runner, 1) == "90"
    assert detect_cuda_arch(runner, 2) is None
    monkeypatch.setenv("CUDA_ARCH", "sm_89")
    assert detect_cuda_arch(runner) == "89"


def _hip_tree(root: Path, headers: bool) -> Path:
    (root / "bin").mkdir(parents=True)
    hipcc = root / "bin" / "hipcc"
    hipcc.write_text("#!/bin/sh\n")
    if headers:
        (root / "include" / "hip").mkdir(parents=True)
        (root / "include" / "hip" / "hip_runtime.h").write_text("")
    return hipcc


def test_find_hip_include_prefers_hipccs_root_then_other_installs(tmp_path, monkeypatch):
    from oneapi_agent.tools.gpu_test import find_hip_include

    monkeypatch.delenv("HIP_PATH", raising=False)
    monkeypatch.delenv("ROCM_PATH", raising=False)
    rocm = tmp_path / "rocm"
    hipcc = _hip_tree(rocm / "core-7.14", headers=False)  # DigitalOcean gpu-amd-base layout
    assert find_hip_include(str(hipcc), roots=(str(rocm),)) is None
    _hip_tree(rocm / "core-7.13", headers=True)
    assert find_hip_include(str(hipcc), roots=(str(rocm),)) == rocm / "core-7.13" / "include"
    monkeypatch.setenv("HIP_PATH", str(tmp_path / "hip"))
    _hip_tree(tmp_path / "hip", headers=True)
    assert find_hip_include(str(hipcc), roots=(str(rocm),)) == tmp_path / "hip" / "include"
    (rocm / "core-7.14" / "include" / "hip").mkdir(parents=True)
    (rocm / "core-7.14" / "include" / "hip" / "hip_runtime.h").write_text("")
    assert find_hip_include(str(hipcc), roots=(str(rocm),)) == rocm / "core-7.14" / "include"


def test_native_hip_build_without_headers_is_a_stated_limitation(tmp_path, monkeypatch):
    gt = importlib.import_module("oneapi_agent.tools.gpu_test")

    hipcc = _hip_tree(tmp_path / "core-7.14", headers=False)
    monkeypatch.setenv("HIPCC", str(hipcc))
    monkeypatch.setenv("HIPIFY", "/bin/hipify-perl")
    calls = []

    def runner(argv, **kw):
        calls.append(list(argv))
        return CommandResult(list(argv), 0, "#include \"hip/hip_runtime.h\"\n", "", 0.0)

    task = {"cuda_path_abs": str(tmp_path / "k.cu")}
    monkeypatch.setattr(gt, "find_hip_include", lambda hipcc: None)
    res, note = gt._native_build(task, "amd", tmp_path, ["-Ia"], "gfx950", runner)
    assert res is None and calls == [] and note.startswith("no HIP headers on this host")

    other = tmp_path / "elsewhere" / "include"
    monkeypatch.setattr(gt, "find_hip_include", lambda hipcc: other)
    res, note = gt._native_build(task, "amd", tmp_path, ["-Ia"], "gfx950", runner)
    assert note is None and res.returncode == 0
    assert calls[-1][0] == str(hipcc) and f"-I{other}" in calls[-1]

    monkeypatch.setattr(gt, "find_hip_include", lambda hipcc: tmp_path / "core-7.14" / "include")
    gt._native_build(task, "amd", tmp_path, ["-Ia"], "gfx950", runner)
    assert not any(a.startswith(f"-I{tmp_path / 'core-7.14'}") for a in calls[-1])


def test_native_hip_build_hipifies_local_headers(tmp_path, monkeypatch):
    # llm.c: the harness includes "common.h", which includes <cuda_runtime.h>; converting only
    # the .cu left common.h as CUDA and hipcc failed with "cuda_runtime.h file not found".
    gt = importlib.import_module("oneapi_agent.tools.gpu_test")
    hipcc = _hip_tree(tmp_path / "rocm", headers=True)
    monkeypatch.setenv("HIPCC", str(hipcc))
    monkeypatch.setenv("HIPIFY", "/bin/hipify-perl")
    monkeypatch.setattr(gt, "find_hip_include", lambda h: tmp_path / "rocm" / "include")
    task_dir = tmp_path / "task"
    (task_dir / "sub").mkdir(parents=True)
    (task_dir / "k.cu").write_text('#include "common.h"\n#include <stdio.h>\nint main(){}\n')
    (task_dir / "common.h").write_text('#include <cuda_runtime.h>\n#include "sub/util.h"\n')
    (task_dir / "sub" / "util.h").write_text('#include "../common.h"\nint util;\n')
    work = tmp_path / "work"
    work.mkdir()
    calls = []

    def runner(argv, **kw):
        calls.append(list(argv))
        if argv[0] == "/bin/hipify-perl":
            text = Path(argv[1]).read_text().replace("cuda_runtime.h", "hip/hip_runtime.h")
            return CommandResult(list(argv), 0, text, "", 0.0)
        return CommandResult(list(argv), 0, "", "", 0.0)

    res, note = gt._native_build({"cuda_path_abs": str(task_dir / "k.cu")}, "amd", work,
                                 ["-Ia"], "gfx950", runner)
    assert note is None and res.returncode == 0
    hipified = [Path(c[1]).name for c in calls if c[0] == "/bin/hipify-perl"]
    assert sorted(hipified) == ["common.h", "k.cu", "util.h"]  # each header once
    assert (work / "common.h").read_text().startswith("#include <hip/hip_runtime.h>")
    assert (work / "sub" / "util.h").exists() and (work / "k.hip.cpp").exists()
    assert calls[-1][0] == str(hipcc)


def test_native_nvidia_build_appends_native_flags(tmp_path, monkeypatch):
    gt = importlib.import_module("oneapi_agent.tools.gpu_test")
    monkeypatch.setenv("NVCC", "/usr/local/cuda/bin/nvcc")
    calls = []

    def runner(argv, **kw):
        calls.append(list(argv))
        return CommandResult(list(argv), 0, "", "", 0.0)

    src = str(tmp_path / "k.cu")
    gt._native_build({"cuda_path_abs": src}, "nvidia", tmp_path, ["-Ia"], None, runner)
    assert calls[-1][calls[-1].index(src) + 1] == "-o"  # unchanged without native_flags
    task = {"cuda_path_abs": src, "native_flags": ["-lcublas", "-lcublasLt"]}
    gt._native_build(task, "nvidia", tmp_path, ["-Ia"], None, runner)
    argv = calls[-1]
    assert argv[argv.index(src) + 1:argv.index("-o")] == ["-lcublas", "-lcublasLt"]


def test_cg_reduce_header_is_mapped_to_hip_or_reported(tmp_path):
    from oneapi_agent.tools.gpu_test import map_cg_headers

    src = ('#include "hip/hip_runtime.h"\n#include <hip/hip_cooperative_groups.h>\n'
           "#include <cooperative_groups/reduce.h>\nint x;\n")
    inc = tmp_path / "include"
    # No equivalent in this ROCm: a stated reason, not a compile error.
    _, why = map_cg_headers(src, inc)
    assert why and "cooperative_groups/reduce.h" in why
    # The internal ROCm 7 header is used when the public one is absent.
    (inc / "hip" / "amd_detail").mkdir(parents=True)
    (inc / "hip/amd_detail/amd_hip_cooperative_groups_reduce.h").write_text("")
    out, why = map_cg_headers(src, inc)
    assert why is None and "#include <hip/amd_detail/amd_hip_cooperative_groups_reduce.h>" in out
    # The public path wins when both exist; other lines are untouched.
    (inc / "hip" / "cooperative_groups").mkdir()
    (inc / "hip/cooperative_groups/hip_reduce.h").write_text("")
    out, why = map_cg_headers(src, inc)
    assert why is None and "<hip/cooperative_groups/hip_reduce.h>" in out
    assert "cooperative_groups/reduce.h" not in out and out.endswith("int x;\n")
    # Sources without the header pass through unchanged.
    assert map_cg_headers("int y;\n", inc) == ("int y;\n", None)


def test_task_output_format_reaches_the_verifier(tmp_path, monkeypatch):
    gt = importlib.import_module("oneapi_agent.tools.gpu_test")
    bm = importlib.import_module("oneapi_agent.tools.benchmark")
    from oneapi_agent.result import ToolResult

    monkeypatch.setattr(bm, "find_tool", lambda n: "/x/sycl-ls")
    monkeypatch.setattr(gt, "detect_amd_arch", lambda runner=None: "gfx950")
    monkeypatch.setattr(gt, "compile_sycl", lambda *a, **k: ToolResult("compile", "ok", "ok"))
    monkeypatch.setattr(gt, "benchmark", lambda *a, **k: ToolResult("benchmark", "ok", "ok"))
    seen = []

    def fake_verify(*a, **k):
        seen.append(k)
        return ToolResult("verify", "ok", "PASS")

    monkeypatch.setattr(gt, "verify", fake_verify)

    def runner(argv, **kw):
        out = "[hip:gpu][hip:0] AMD HIP BACKEND, AMD Instinct MI350X gfx950 [HIP 7.14]\n"
        return CommandResult(list(argv), 0, out, "", 0.0)

    task = json.loads((REPO / "examples/saxpy/task.json").read_text())
    tdir = tmp_path / "t"
    tdir.mkdir()
    for f in ("saxpy.cu", "saxpy_ref.cpp"):
        (tdir / f).write_text((REPO / "examples/saxpy" / f).read_text())
    (tdir / "task.json").write_text(json.dumps({**task, "output_format": "f64"}))
    r = gpu_test(str(tdir / "task.json"), str(tmp_path / "o"), native=False, runner=runner)
    assert r.status == "ok", r.summary
    assert seen and all(k["output_format"] == "f64" for k in seen)


def test_benchmark_stats_work_on_old_numpy(monkeypatch):
    # Ubuntu 22.04's python3-numpy (1.21, the Runpod ROCm image) has no `method=` in percentile.
    import numpy as np

    bm = importlib.import_module("oneapi_agent.tools.benchmark")

    real = np.percentile

    def old_percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation="linear",
                       keepdims=False):
        return real(a, q)

    monkeypatch.setattr(bm.np, "percentile", old_percentile)
    s = bm._stats([1.0, 2.0, 3.0, 4.0])
    assert s["p95"] == real([1.0, 2.0, 3.0, 4.0], 95) and s["n"] == 4


def _fetch_task(tmp_path: Path, **extra) -> Path:
    t = tmp_path / "t"
    (t / "sycl").mkdir(parents=True, exist_ok=True)
    (t / "k.cu").write_text("// cuda")
    (t / "sycl/k.cpp").write_text("// sycl")
    (t / "big.bin").write_bytes(b"\1" * 64)  # stands for a large input fetched on the machine
    (t / "task.json").write_text(json.dumps({
        "cuda_path": "k.cu", "inputs": ["big.bin"], "reference_build": ["true"],
        "reference_cmd": "r", "candidate_cmd": "c", **extra}))
    return t / "task.json"


def test_bundle_fetches_large_files_instead_of_packing_them(tmp_path):
    sha = "a" * 64
    task = _fetch_task(tmp_path, fetch=[{"path": "big.bin", "sha256": sha,
                                         "url": "https://example.org/big.bin?download=true"}])
    (tmp_path / "t/big.bin").unlink()  # need not exist locally
    r = make_bundle(str(task), str(tmp_path / "b.sh"))
    assert r.status == "ok", r.summary
    assert "task/big.bin" not in _payload(tmp_path / "b.sh").getnames()
    script = (tmp_path / "b.sh").read_text()
    assert ("fetch_file \"$D/task\"/'big.bin' 'https://example.org/big.bin?download=true' "
            + sha) in script


def test_bundle_rejects_bad_fetch_entries(tmp_path):
    for bad in ({"path": "../x", "url": "https://e/x", "sha256": "a" * 64},
                {"path": "x", "url": "http://e/x", "sha256": "a" * 64},
                {"path": "x", "url": "https://e/x", "sha256": "nothex"},
                {"path": "x", "url": "https://e/x"}):
        task = _fetch_task(tmp_path, fetch=[bad])
        assert make_bundle(str(task), str(tmp_path / "b.sh")).status == "error", bad


def test_bundle_sources_gpu_setup_and_archives_its_log(tmp_path):
    import subprocess

    task = _fetch_task(tmp_path, gpu_setup="setup.sh")
    (tmp_path / "t/setup.sh").write_text('echo "setup ran"; export GT_SETUP_MARK=yes\n')
    r = make_bundle(str(task), str(tmp_path / "b.sh"))
    assert r.status == "ok", r.summary
    # an unknown flag ends gpu-test early; setup has run and its log is archived
    subprocess.run(["bash", str(tmp_path / "b.sh"), "--no-such-flag"], cwd=tmp_path,
                   capture_output=True, text=True, timeout=300)
    [archive] = tmp_path.glob("gputest-results-*.tar.gz")
    with tarfile.open(archive) as t:
        assert "setup ran" in t.extractfile("setup.log").read().decode()
    task2 = _fetch_task(tmp_path, gpu_setup="../escape.sh")
    assert make_bundle(str(task2), str(tmp_path / "b2.sh")).status == "error"
