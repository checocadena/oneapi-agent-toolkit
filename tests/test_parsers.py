"""Parsers against real captured c2s / icpx / sycl-ls output (tests/fixtures)."""

from __future__ import annotations

from oneapi_agent.parsers.c2s import (
    merge_warnings,
    parse_c2s_errors,
    parse_c2s_stderr,
    parse_dpct_comments,
)
from oneapi_agent.parsers.diagnostics import parse_clang_diagnostics, summarize
from oneapi_agent.parsers.dpct_scan import categorize, scan_source, strip_comments_and_strings
from oneapi_agent.parsers.sycl_ls import parse_sycl_ls

# ------------------------------------------------------------------ c2s


def test_c2s_stderr_warnings(fixtures):
    ws = parse_c2s_stderr((fixtures / "c2s_stderr.txt").read_text())
    assert [(w.code, w.id, w.line, w.column) for w in ws] == [
        ("DPCT1010", 0, 30, 21),
        ("DPCT1009", 1, 31, 56),
    ]
    assert ws[0].file == "/work/reduce/reduce.cu"
    assert ws[0].message.startswith("SYCL uses exceptions to report errors")
    assert "get_error_string_dummy" in ws[1].message


def test_c2s_comment_blocks_multiline(fixtures):
    ws = parse_dpct_comments((fixtures / "reduce.dp.cpp").read_text(), "reduce.dp.cpp")
    assert [(w.code, w.id) for w in ws] == [("DPCT1010", 0), ("DPCT1009", 1)]
    # multi-line comment text is joined, and the location is in the migrated file
    assert ws[0].message.endswith("You need to rewrite this code.")
    assert ws[0].migrated_file == "reduce.dp.cpp"
    assert ws[0].migrated_line is not None and ws[0].migrated_line > 60


def test_c2s_merge_joins_both_views(fixtures):
    merged = merge_warnings(
        parse_c2s_stderr((fixtures / "c2s_stderr.txt").read_text()),
        parse_dpct_comments((fixtures / "reduce.dp.cpp").read_text(), "reduce.dp.cpp"),
    )
    assert len(merged) == 2
    assert all(w.line is not None and w.migrated_line is not None for w in merged)


def test_c2s_comment_only_warning_kept():
    src = "int x;\n// DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the limit.\n"
    merged = merge_warnings([], parse_dpct_comments(src, "a.cpp"))
    assert [(w.code, w.id, w.migrated_line) for w in merged] == [("DPCT1049", 7, 2)]


def test_c2s_fatal_errors(fixtures):
    errs = parse_c2s_errors((fixtures / "c2s_stderr_fatal.txt").read_text())
    assert len(errs) == 1
    assert errs[0]["severity"] == "fatal error"
    assert "curand_mtgp32_kernel.h" in errs[0]["message"]
    assert errs[0]["line"] == 534


def test_c2s_stdout_has_no_warnings(fixtures):
    assert parse_c2s_stderr((fixtures / "c2s_stdout.txt").read_text()) == []


# ------------------------------------------------------------------ icpx


def test_icpx_errors_and_warning(fixtures):
    diags = parse_clang_diagnostics((fixtures / "icpx_errors.txt").read_text())
    assert [(d.severity, d.line, d.column) for d in diags] == [
        ("warning", 2, 28),
        ("error", 6, 66),
        ("error", 7, 3),
    ]
    assert diags[0].flag == "-Wunused-variable"
    assert diags[0].message == "unused variable 'y'"
    assert diags[1].message == "use of undeclared identifier 'undeclared_var'"
    assert "undeclared_var" in (diags[1].snippet or "")
    assert summarize(diags) == {"error": 2, "warning": 1, "note": 0}


def test_icpx_link_error(fixtures):
    diags = parse_clang_diagnostics((fixtures / "icpx_link_error.txt").read_text())
    msgs = [d.message for d in diags]
    assert any("undefined reference to `helper(float*)'" in m for m in msgs)
    assert any(m.startswith("icpx: linker command failed") for m in msgs)
    assert all(d.is_error for d in diags)


def test_icpx_note_and_fatal():
    text = (
        "a.cpp:3:10: fatal error: 'missing.h' file not found\n"
        "b.cpp:9:5: note: candidate function not viable\n"
        "c.cpp:1:1: remark: loop vectorized [-Rpass=loop-vectorize]\n"
    )
    d = parse_clang_diagnostics(text)
    assert [x.severity for x in d] == ["fatal error", "note", "remark"]
    assert d[0].is_error and not d[1].is_error


# ------------------------------------------------------------------ dpct scan


def test_scan_real_migrated_file(fixtures):
    uses = scan_source((fixtures / "reduce.dp.cpp").read_text(), "reduce.dp.cpp")
    symbols = [u.symbol for u in uses]
    assert len(uses) == 16
    assert "#include <dpct/dpct.hpp>" in symbols
    assert "DPCT_PROFILING_ENABLED" in symbols
    assert symbols.count("dpct::dpct_free") == 2
    cats = {u.symbol: u.category for u in uses}
    assert cats["dpct::get_current_device"] == "device_queue_management"
    assert cats["dpct::constant_memory"] == "memory_management"
    assert cats["dpct::shift_sub_group_left"] == "subgroup_warp"
    assert cats["dpct::atomic_fetch_add"] == "atomics"
    assert cats["dpct::event_ptr"] == "events_timing"
    assert cats["dpct::err0"] == "error_handling"


def test_scan_ignores_comments_and_strings_keeps_lines():
    src = (
        "/* dpct::get_current_device() in a block comment\n"
        "   spanning lines */\n"
        "const char* s = \"dpct::dpct_malloc\"; // dpct::dpct_free\n"
        "auto q = dpct::get_default_queue();\n"
    )
    uses = scan_source(src, "x.cpp")
    assert [(u.line, u.symbol) for u in uses] == [(4, "dpct::get_default_queue")]
    assert strip_comments_and_strings(src).count("\n") == src.count("\n")


def test_categorize_examples():
    assert categorize("blas::gemm") == "library_wrappers"
    assert categorize("blas::descriptor_ptr") == "library_wrappers"  # c2s's cublasHandle_t
    assert categorize("dim3") == "types_launch"
    assert categorize("select_from_sub_group") == "subgroup_warp"
    assert categorize("something_new") == "other"


def test_scan_clean_sycl_has_no_uses():
    src = "#include <sycl/sycl.hpp>\nint main(){ sycl::queue q; q.wait(); }\n"
    assert scan_source(src, "ok.cpp") == []


# ------------------------------------------------------------------ sycl-ls


def test_sycl_ls_new_and_old_formats():
    text = (
        "[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Xeon(R) Processor @ 2.80GHz "
        "OpenCL 3.0 (Build 0) [2026.21.7.0.24_160000]\n"
        "[level_zero:gpu][level_zero:0] Intel(R) Level-Zero, Intel(R) Arc(TM) A770 Graphics "
        "1.3 [1.3.26516]\n"
        "[cuda:gpu:0] NVIDIA CUDA BACKEND, NVIDIA A100 8.0 [CUDA 12.2]\n"
        "Warning: something unrelated\n"
    )
    devs = parse_sycl_ls(text)
    assert [(d.backend, d.device_type, d.index, d.selector) for d in devs] == [
        ("opencl", "cpu", 0, "opencl:0"),
        ("level_zero", "gpu", 0, "level_zero:0"),
        ("cuda", "gpu", 0, "cuda:0"),
    ]
    assert devs[0].name == "Intel(R) Xeon(R) Processor @ 2.80GHz"
    assert devs[1].name == "Intel(R) Arc(TM) A770 Graphics"
    assert devs[0].platform == "Intel(R) OpenCL"
