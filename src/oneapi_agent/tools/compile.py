"""compile: build SYCL sources with ``icpx -fsycl`` and return structured diagnostics."""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path

from ..parsers.diagnostics import parse_clang_diagnostics, summarize
from ..result import ToolResult
from ..toolchain import Runner, dpct_include_dir, find_tool, missing_tool_message, run_command


def compile_sycl(
    sources: Sequence[str] | str,
    flags: Sequence[str] | str = (),
    *,
    output: str | None = None,
    cwd: str | None = None,
    add_dpct_include: bool = True,
    timeout: float = 900.0,
    runner: Runner | None = None,
    icpx_path: str | None = None,
) -> ToolResult:
    """Compile ``sources`` with ``icpx -fsycl <flags>``.

    ``flags`` may be a list or a shell-like string. When ``output`` is omitted the
    binary goes next to the first source as ``<stem>.sycl.out``. If the SYCLomatic
    helper headers are discoverable they are added with ``-I`` (needed for
    unrefactored dpct output); the evidence records whether that happened.
    """
    srcs = [sources] if isinstance(sources, str) else list(sources)
    if not srcs:
        return ToolResult("compile", "error", "no sources given")
    base = Path(cwd) if cwd else Path.cwd()
    missing = [s for s in srcs if not (base / s).exists()]
    if missing:
        return ToolResult("compile", "error", f"source(s) not found: {missing}",
                          {"missing": missing, "cwd": str(base)})

    flag_list = shlex.split(flags) if isinstance(flags, str) else list(flags)
    out = output or str(Path(srcs[0]).with_suffix(".sycl.out"))

    icpx = icpx_path or find_tool("icpx")
    if runner is None and icpx is None:
        return ToolResult(
            "compile",
            "toolchain_missing",
            "Intel oneAPI DPC++ compiler (icpx) is not installed; nothing was compiled",
            {"failed_command": shlex.join(["icpx", "-fsycl", *flag_list, *srcs, "-o", out]),
             "reason": missing_tool_message("icpx")},
            ["No compilation was attempted."],
        )
    icpx = icpx or "icpx"

    argv = [icpx, "-fsycl", *flag_list]
    dpct_inc = dpct_include_dir() if add_dpct_include else None
    if dpct_inc and not any(f.startswith("-I") and dpct_inc in f for f in flag_list):
        argv.append(f"-I{dpct_inc}")
    argv += [*srcs, "-o", out]

    res = (runner or run_command)(argv, cwd=cwd, timeout=timeout)
    evidence: dict = {"run": res.evidence(), "output": out, "dpct_include_added": dpct_inc}
    limitations = ["A successful compile says nothing about numerical correctness; run verify."]

    if res.not_found:
        return ToolResult("compile", "toolchain_missing", f"could not execute {icpx}",
                          {**evidence, "failed_command": res.command}, limitations)
    if res.timed_out:
        return ToolResult("compile", "timeout", f"icpx exceeded {timeout}s", evidence, limitations)

    diags = parse_clang_diagnostics(res.stderr + "\n" + res.stdout)
    counts = summarize(diags)
    evidence["diagnostics"] = [d.to_dict() for d in diags]
    evidence["counts"] = counts

    if res.returncode != 0:
        first = next((d for d in diags if d.is_error), None)
        where = f" at {first.file}:{first.line}" if first and first.file else ""
        msg = f": {first.message}" if first else ""
        if not diags:
            limitations.append("icpx failed but printed no parseable diagnostics; see raw stderr.")
        return ToolResult(
            "compile",
            "fail",
            f"compile failed with {counts['error']} error(s){where}{msg}",
            evidence,
            limitations,
        )
    return ToolResult(
        "compile",
        "ok",
        f"compiled {len(srcs)} source(s) -> {out} ({counts['warning']} warning(s))",
        evidence,
        limitations,
    )
