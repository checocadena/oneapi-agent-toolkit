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
    assert target_flags(dev("hip", "gpu"), "gfx942") == ["-fsycl-targets=amd_gpu_gfx942"]
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
