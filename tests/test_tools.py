"""The five tools with fake runners (no oneAPI toolchain needed).

verify is exercised with real subprocesses (the Python interpreter) because it only
needs *some* reference and candidate programs.
"""

from __future__ import annotations

import importlib
import json
import shutil
import textwrap

import numpy as np
import pytest
from conftest import PY, FakeRunner, result

from oneapi_agent.audio import make_test_signal, write_wav
from oneapi_agent.result import ToolResult
from oneapi_agent.tools import benchmark, compile_sycl, migrate, scan_helpers, verify

# ------------------------------------------------------------------ result type


def test_toolresult_validates_and_serialises():
    r = ToolResult("x", "ok", "fine", {"n": 1}, ["lim"])
    assert ToolResult.from_dict(json.loads(r.to_json())) == r
    with pytest.raises(ValueError):
        ToolResult("x", "great", "nope")  # type: ignore[arg-type]


# ------------------------------------------------------------------ toolchain missing


def test_missing_toolchain_is_reported_not_faked(no_toolchain, tmp_path, fixtures):
    src = tmp_path / "a.cpp"
    src.write_text("int main(){}")
    c = compile_sycl([str(src)], ["-O2"])
    assert c.status == "toolchain_missing"
    assert c.evidence["failed_command"].startswith("icpx -fsycl -O2")
    m = migrate(str(fixtures / "reduce.cu"), str(tmp_path / "out"))
    assert m.status == "toolchain_missing"
    assert "c2s" in m.evidence["failed_command"]
    assert "c2s" in m.evidence["reason"]


def test_runner_not_found_maps_to_toolchain_missing(tmp_path):
    src = tmp_path / "a.cpp"
    src.write_text("int main(){}")
    c = compile_sycl([str(src)], icpx_path="/definitely/not/here/icpx")
    assert c.status == "toolchain_missing"
    assert c.evidence["failed_command"].startswith("/definitely/not/here/icpx")


# ------------------------------------------------------------------ migrate


def test_migrate_parses_warnings_and_outputs(tmp_path, fixtures):
    out = tmp_path / "out"

    def respond(argv, kw):
        out.mkdir(exist_ok=True)
        shutil.copy(fixtures / "reduce.dp.cpp", out / "reduce.dp.cpp")
        return result(argv, 0, (fixtures / "c2s_stdout.txt").read_text(),
                      (fixtures / "c2s_stderr.txt").read_text())

    runner = FakeRunner(respond)
    r = migrate(str(fixtures / "reduce.cu"), str(out), extra_args=["-I/x"], runner=runner,
                c2s_path="c2s")
    argv = runner.calls[0][0]
    assert argv[0] == "c2s" and "--extra-arg=-I/x" in argv
    assert any(a.startswith("--out-root=") for a in argv)
    assert r.status == "ok"
    assert r.evidence["migrated_files"] == ["reduce.dp.cpp"]
    assert r.evidence["warning_counts"] == {"DPCT1010": 1, "DPCT1009": 1}
    w = r.evidence["warnings"][0]
    assert (w["code"], w["line"], w["migrated_file"]) == ("DPCT1010", 30, "reduce.dp.cpp")
    assert "DPCT1009x1" in r.summary


def test_migrate_exit0_with_parse_error_is_fail(tmp_path, fixtures):
    out = tmp_path / "out"

    def respond(argv, kw):
        out.mkdir(exist_ok=True)
        (out / "reduce.dp.cpp").write_text("// partial\n")
        return result(argv, 0, "", (fixtures / "c2s_stderr_fatal.txt").read_text())

    r = migrate(str(fixtures / "reduce.cu"), str(out), runner=FakeRunner(respond), c2s_path="c2s")
    assert r.status == "fail"
    assert "curand_mtgp32_kernel.h" in r.summary


def test_migrate_bad_input(tmp_path):
    assert migrate(str(tmp_path / "nope.cu"), str(tmp_path)).status == "error"
    (tmp_path / "empty").mkdir()
    assert migrate(str(tmp_path / "empty"), str(tmp_path / "o")).status == "error"


# ------------------------------------------------------------------ compile


def test_compile_fail_parses_diagnostics(tmp_path, fixtures):
    src = tmp_path / "broken.cpp"
    src.write_text("x")
    runner = FakeRunner(lambda a, kw: result(a, 1, "", (fixtures / "icpx_errors.txt").read_text()))
    r = compile_sycl([str(src)], "-O2 -DX=1", output=str(tmp_path / "b"), runner=runner,
                     icpx_path="icpx", add_dpct_include=False)
    argv = runner.calls[0][0]
    assert argv[:4] == ["icpx", "-fsycl", "-O2", "-DX=1"]
    assert argv[-2:] == ["-o", str(tmp_path / "b")]
    assert r.status == "fail"
    assert r.evidence["counts"] == {"error": 2, "warning": 1, "note": 0}
    assert "broken.cpp:6" in r.summary and "undeclared_var" in r.summary


def test_compile_ok(tmp_path):
    src = tmp_path / "ok.cpp"
    src.write_text("x")
    r = compile_sycl([str(src)], [], runner=FakeRunner(lambda a, kw: result(a, 0)),
                     icpx_path="icpx", add_dpct_include=False)
    assert r.status == "ok" and r.evidence["counts"]["error"] == 0


def test_compile_missing_source(tmp_path):
    assert compile_sycl([str(tmp_path / "nope.cpp")]).status == "error"


# ------------------------------------------------------------------ scan_helpers


def test_scan_helpers_tool(tmp_path, fixtures):
    r = scan_helpers(str(fixtures / "reduce.dp.cpp"))
    assert r.status == "fail" and r.evidence["total"] == 16
    assert set(r.evidence["suggestions"]) == set(r.evidence["by_category"])
    kinds = {a["kind"] for a in r.evidence["advisories"]}
    assert {"hardcoded_subgroup_32", "nd_item3_for_1d"} <= kinds
    assert len(r.evidence["unresolved_dpct_comments"]) == 2

    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "k.cpp").write_text(
        "#include <sycl/sycl.hpp>\n// was dpct::get_current_device()\n"
        "float f(float x){ return sycl::native::exp(x); }\n")
    r2 = scan_helpers(str(clean))
    assert r2.status == "ok" and r2.evidence["total"] == 0
    assert [a["kind"] for a in r2.evidence["advisories"]] == ["native_math"]


def test_scan_helpers_flags_onemkl_product_api(tmp_path):
    # Shape of the c2s output for cublasSgemm (examples/gemm_cublas): a dpct::blas handle
    # plus the oneMKL product API.
    mig = tmp_path / "gemm.dp.cpp"
    mig.write_text(
        "#include <sycl/sycl.hpp>\n#include <dpct/dpct.hpp>\n#include <dpct/blas_utils.hpp>\n"
        "void f(float* a, float* b, float* c) {\n"
        "  dpct::blas::descriptor_ptr handle = new dpct::blas::descriptor();\n"
        "  oneapi::mkl::blas::column_major::gemm(handle->get_queue(),\n"
        "      oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,\n"
        "      4, 4, 4, 1.f, a, 4, b, 4, 0.f, c, 4);\n}\n")
    r = scan_helpers(str(mig))
    assert r.status == "fail"
    assert r.evidence["by_category"]["library_wrappers"] == 2
    assert "oneapi::math" in r.evidence["suggestions"]["library_wrappers"]
    lines = [a["line"] for a in r.evidence["advisories"] if a["kind"] == "onemkl_product_api"]
    assert lines == [6, 7]

    port = tmp_path / "gemm.cpp"  # the oneMath port: no helpers and no advisory
    port.write_text(
        "#include <oneapi/math.hpp>\n// was oneapi::mkl::blas::column_major::gemm\n"
        "void f(sycl::queue& q, float* a, float* b, float* c) {\n"
        "  oneapi::math::blas::column_major::gemm(q, oneapi::math::transpose::nontrans,\n"
        "      oneapi::math::transpose::nontrans, 4, 4, 4, 1.f, a, 4, b, 4, 0.f, c, 4);\n}\n")
    r2 = scan_helpers(str(port))
    assert r2.status == "ok" and r2.evidence["advisories"] == []


# ------------------------------------------------------------------ verify (real subprocesses)

WRITE_ARRAY = textwrap.dedent("""
    import sys, numpy as np
    out, delta = sys.argv[1], float(sys.argv[2])
    x = np.linspace(0, 1, 1000, dtype=np.float32)
    x[500] += np.float32(delta)
    x.tofile(out)
""")

AUDIO_FX = textwrap.dedent("""
    import sys, numpy as np
    sys.path.insert(0, sys.argv[4])
    from oneapi_agent.audio import read_wav, write_wav
    w = read_wav(sys.argv[1])
    y = np.tanh(4.0 * w.samples) * 0.5
    y[100, :] += float(sys.argv[3])
    write_wav(sys.argv[2], y, w.sample_rate)
""")


@pytest.fixture
def scripts(tmp_path):
    (tmp_path / "arr.py").write_text(WRITE_ARRAY)
    (tmp_path / "fx.py").write_text(AUDIO_FX)
    return tmp_path


def test_verify_array_pass_and_fail(scripts):
    ref = f"{PY} arr.py {{output}} 0"
    ok = verify(ref, f"{PY} arr.py {{output}} 1e-7", tolerance=1e-6, cwd=str(scripts))
    assert ok.status == "ok", ok.summary
    st = ok.evidence["cases"][0]["stats"]
    assert st["n_reference"] == 1000 and st["max_abs_err"] > 0
    bad = verify(ref, f"{PY} arr.py {{output}} 1e-3", tolerance=1e-6, cwd=str(scripts))
    assert bad.status == "fail"
    assert bad.evidence["cases"][0]["stats"]["first_mismatches"][0]["index"] == 500


def test_verify_stdout_mode():
    r = verify([PY, "-c", "print('result:', 1.0, 2.0)"], [PY, "-c", "print(1.0, 2.0000001)"],
               tolerance=1e-6)
    assert r.status == "ok"
    assert r.evidence["cases"][0]["reference_format"] == "stdout-text"
    assert any("Text outputs" in lim for lim in r.limitations)


def test_verify_candidate_crash_is_fail_reference_crash_is_error():
    good = [PY, "-c", "print(1)"]
    crash = [PY, "-c", "import sys; sys.exit(3)"]
    c = verify(good, crash)
    assert c.status == "fail" and "exited with code 3" in c.summary
    r = verify(crash, good)
    assert r.status == "error"
    nf = verify(good, ["/no/such/candidate"])
    assert nf.status == "error" and "not found" in nf.summary


def test_verify_requires_inputs_for_input_placeholder():
    assert verify("prog {input}", "prog {input}").status == "error"


def test_verify_audio_null_test(scripts, tmp_path):
    from conftest import REPO

    src = str(REPO / "src")
    wav = tmp_path / "in.wav"
    write_wav(wav, np.stack([make_test_signal(8000, 0.2, seed=3)] * 2, 1), 8000)
    ref = f"{PY} fx.py {{input}} {{output}} 0 {src}"
    passing = verify(ref, f"{PY} fx.py {{input}} {{output}} 1e-6 {src}", inputs=[str(wav)],
                     cwd=str(scripts))
    assert passing.status == "ok", passing.summary
    assert passing.evidence["mode"] == "audio"
    stats = passing.evidence["cases"][0]["stats"]
    assert stats["residual_peak_dbfs"] == pytest.approx(-120, abs=0.5)  # 1e-6 step, float32 WAV rounding
    assert stats["per_channel"][0]["peak_frame"] == 100

    failing = verify(ref, f"{PY} fx.py {{input}} {{output}} 1e-4 {src}", inputs=[str(wav)],
                     cwd=str(scripts))
    assert failing.status == "fail"
    assert "-80" in failing.summary or "exceeds" in failing.summary


def test_verify_audio_bit_exact(scripts, tmp_path):
    from conftest import REPO

    wav = tmp_path / "in.wav"
    write_wav(wav, make_test_signal(8000, 0.1), 8000)
    cmd = f"{PY} fx.py {{input}} {{output}} 0 {REPO / 'src'}"
    r = verify(cmd, cmd, inputs=[str(wav)], cwd=str(scripts))
    assert r.status == "ok" and "bit-exact" in r.summary


# ------------------------------------------------------------------ benchmark

SYCL_LS = ("[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) CPU OpenCL 3.0 (Build 0) [x]\n"
           "[level_zero:gpu][level_zero:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics "
           "1.3 [1.3.1]\n")


def test_benchmark_per_device_stats(monkeypatch):
    tb = importlib.import_module("oneapi_agent.tools.benchmark")
    monkeypatch.setattr(tb, "find_tool", lambda name: "/fake/sycl-ls")
    durations = iter([0.5, 0.1, 0.2, 0.3, 9.9, 0.04, 0.01, 0.02, 0.03])

    def respond(argv, kw):
        if argv[0].endswith("sycl-ls"):
            return result(argv, 0, SYCL_LS)
        return result(argv, 0, "kernel_ms=1.5\n", duration=next(durations))

    runner = FakeRunner(respond)
    r = benchmark("./app x", runs=3, warmup=1, runner=runner, report_regex=r"kernel_ms=([\d.]+)")
    assert r.status == "ok"
    res = r.evidence["results"]
    assert [x["selector"] for x in res] == ["opencl:0", "level_zero:0"]
    assert res[0]["samples_s"] == [0.1, 0.2, 0.3]  # warm-up (0.5) excluded
    assert res[0]["wall_time_s"]["median"] == pytest.approx(0.2)
    assert res[1]["wall_time_s"]["median"] == pytest.approx(0.02)
    assert res[0]["reported"]["median"] == 1.5
    envs = [kw.get("env") for argv, kw in runner.calls if not argv[0].endswith("sycl-ls")]
    assert envs[0] == {"ONEAPI_DEVICE_SELECTOR": "opencl:0"}
    assert "Arc(TM) A770" in r.summary


def test_benchmark_without_sycl_ls_runs_unselected(no_toolchain):
    runner = FakeRunner(lambda a, kw: result(a, 0, duration=0.01))
    r = benchmark("./app", runs=2, runner=runner)
    # sycl-ls lists nothing -> one unselected run set, flagged in limitations
    assert r.status == "ok"
    assert [x["selector"] for x in r.evidence["results"]] == ["none"]
    assert any("No SYCL devices" in lim for lim in r.limitations)


def test_benchmark_real_process_no_devices(no_toolchain):
    r = benchmark([PY, "-c", "pass"], runs=2, warmup=0)
    assert r.status == "ok"
    assert r.evidence["results"][0]["selector"] == "none"
    assert any("No SYCL devices" in lim for lim in r.limitations)
    assert len(r.evidence["results"][0]["samples_s"]) == 2


def test_benchmark_failure_reported():
    r = benchmark([PY, "-c", "import sys; sys.exit(2)"], runs=2, devices=["none"])
    assert r.status == "fail"
    assert r.evidence["results"][0]["failure"]["returncode"] == 2
