"""End-to-end checks against the real oneAPI toolchain. Skipped when it is not installed.

Run with the toolchain on PATH, e.g.:  source scripts/oneapi-env.sh && pytest -m toolchain
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oneapi_agent.toolchain import cuda_include_dir, find_tool
from oneapi_agent.tools import compile_sycl, migrate, scan_helpers, verify

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.toolchain

needs_icpx = pytest.mark.skipif(find_tool("icpx") is None, reason="icpx not installed")
needs_c2s = pytest.mark.skipif(find_tool("c2s") is None or cuda_include_dir() is None,
                               reason="c2s or CUDA headers not installed")


@needs_c2s
def test_real_migrate_fixture(tmp_path):
    r = migrate(str(REPO / "tests" / "fixtures" / "reduce.cu"), str(tmp_path / "out"))
    assert r.status == "ok", r.summary
    assert {"DPCT1009", "DPCT1010"} <= set(r.evidence["warning_counts"])


@needs_icpx
def test_real_compile_error_and_success(tmp_path):
    bad = tmp_path / "bad.cpp"
    bad.write_text("#include <sycl/sycl.hpp>\nint main(){ return nope; }\n")
    r = compile_sycl([str(bad)], [], output=str(tmp_path / "bad"))
    assert r.status == "fail"
    assert r.evidence["diagnostics"][0]["line"] == 2

    sax = REPO / "examples" / "saxpy"
    ok = compile_sycl([str(sax / "sycl" / "saxpy.cpp")], ["-O2"], output=str(tmp_path / "saxpy"),
                      add_dpct_include=False)
    assert ok.status == "ok", ok.summary
    assert scan_helpers(str(sax / "sycl" / "saxpy.cpp")).status == "ok"


@needs_icpx
def test_real_verify_saxpy(tmp_path):
    import subprocess

    sax = REPO / "examples" / "saxpy"
    subprocess.run(["g++", "-O2", str(sax / "saxpy_ref.cpp"), "-o", str(tmp_path / "ref")],
                   check=True)
    c = compile_sycl([str(sax / "sycl" / "saxpy.cpp")], ["-O2"], output=str(tmp_path / "cand"),
                     add_dpct_include=False)
    assert c.ok
    v = verify(f"{tmp_path}/ref {{output}} 4096", f"{tmp_path}/cand {{output}} 4096",
               tolerance=1e-6)
    assert v.status == "ok", v.summary
