"""MCP server exposing the evidence tools over stdio.

Run with ``oneapi-agent serve`` (or ``python -m oneapi_agent.server``).
Every tool returns a JSON object: {tool, status, summary, evidence, limitations}.
"""

from __future__ import annotations

from typing import Any

from .toolchain import toolchain_report
from .tools import benchmark as _benchmark
from .tools import compile_sycl as _compile
from .tools import gpu_test as _gpu_test
from .tools import migrate as _migrate
from .tools import scan_helpers as _scan_helpers
from .tools import verify as _verify

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[attr-defined]
except ImportError:  # mcp 2.x renamed FastMCP -> MCPServer (same decorator API)
    from mcp.server.mcpserver import MCPServer as _Server  # type: ignore[no-redef]

INSTRUCTIONS = """\
oneAPI/SYCL last-mile migration tools. Every result has a `status`
(ok | fail | error | toolchain_missing | timeout), a `summary`, structured `evidence`
and `limitations`. Base claims on evidence only. Never claim a port works unless
`verify` returned status "ok" on the current code. Typical order:
migrate -> compile -> (fix) -> scan_helpers -> refactor to idiomatic SYCL -> compile
-> verify -> benchmark.
GPU rental keys (runpod, vast, vultr, digitalocean) are the user's: `providers_status` says
which are set up. Never ask for, store or pass a key; when one is missing, give the user the
console_url and ask them to run the login_command in their own terminal.
"""

STATUS_NOTE = (
    " Returns {tool, status, summary, evidence, limitations}; status is one of "
    "ok|fail|error|toolchain_missing|timeout."
)

mcp = _Server("oneapi-agent", instructions=INSTRUCTIONS)


@mcp.tool(
    name="migrate",
    description=(
        "Migrate CUDA source (a .cu file or a directory of them) to SYCL with SYCLomatic (c2s). "
        "Evidence: migrated_files, warnings[] (code like DPCT1010, id, message, original "
        "file/line, migrated_file/line), warning_counts, parse errors, and the raw c2s log. "
        "extra_args are passed as --extra-arg (e.g. '-I/path/include')." + STATUS_NOTE
    ),
)
def migrate(cuda_path: str, out_dir: str, in_root: str | None = None,
            extra_args: list[str] | None = None) -> dict[str, Any]:
    return _migrate(cuda_path, out_dir, in_root=in_root, extra_args=extra_args or []).to_dict()


@mcp.tool(
    name="compile",
    description=(
        "Compile SYCL sources with `icpx -fsycl <flags>`. Evidence: diagnostics[] "
        "(file, line, column, severity, message, flag, snippet), counts, output path, raw log. "
        "SYCLomatic helper headers are added to the include path automatically "
        "(set add_dpct_include=false to prove the code no longer needs them)." + STATUS_NOTE
    ),
)
def compile(sources: list[str], flags: list[str] | None = None, output: str | None = None,
            cwd: str | None = None, add_dpct_include: bool = True) -> dict[str, Any]:
    return _compile(sources, flags or [], output=output, cwd=cwd,
                    add_dpct_include=add_dpct_include).to_dict()


@mcp.tool(
    name="scan_helpers",
    description=(
        "Inventory remaining SYCLomatic helper usage (dpct:: symbols, DPCT_* macros, "
        "#include <dpct/...>) in a file or directory, ignoring comments. Evidence: total, "
        "by_category, by_symbol, by_file, occurrences (file/line/text), per-category "
        "suggestions for idiomatic SYCL 2020 replacements, unresolved DPCT comments and "
        "migration-artifact advisories. status ok means zero helpers remain." + STATUS_NOTE
    ),
)
def scan_helpers(path: str) -> dict[str, Any]:
    return _scan_helpers(path).to_dict()


@mcp.tool(
    name="verify",
    description=(
        "Prove behavioural equivalence: run reference_cmd and candidate_cmd on the same inputs "
        "and compare outputs. Commands may use {input}, {input0}.. and {output} placeholders; "
        "without {output}, stdout numbers are compared. mode=array: max abs/RMS/rel error and "
        "float32 ULP stats with tolerance (atol) and rtol. mode=audio: WAV null test, residual "
        "peak/RMS in dBFS, pass if peak <= threshold_dbfs (default -96). mode=auto picks audio "
        "for WAV." + STATUS_NOTE
    ),
)
def verify(reference_cmd: str, candidate_cmd: str, inputs: list[str] | None = None,
           tolerance: float = 1e-5, rtol: float = 0.0, mode: str = "auto",
           threshold_dbfs: float = -96.0, max_ulp: int | None = None,
           cwd: str | None = None) -> dict[str, Any]:
    return _verify(reference_cmd, candidate_cmd, inputs or [], tolerance, rtol=rtol, mode=mode,
                   threshold_dbfs=threshold_dbfs, max_ulp=max_ulp, cwd=cwd).to_dict()


@mcp.tool(
    name="benchmark",
    description=(
        "Time a command `runs` times on each SYCL device (from sycl-ls, selected through "
        "ONEAPI_DEVICE_SELECTOR) or on the given selectors (e.g. ['opencl:cpu'], ['none'] for "
        "no selector). Evidence: per-device wall time min/median/p95/max/stdev, raw samples, "
        "device names, optional program-reported times via report_regex. Only measured "
        "numbers are reported." + STATUS_NOTE
    ),
)
def benchmark(cmd: str, runs: int = 10, devices: list[str] | None = None, warmup: int = 1,
              cwd: str | None = None, report_regex: str | None = None) -> dict[str, Any]:
    return _benchmark(cmd, runs, devices, warmup=warmup, cwd=cwd,
                      report_regex=report_regex).to_dict()


@mcp.tool(
    name="gpu_test",
    description=(
        "Verify a task's SYCL port on this machine's GPU (NVIDIA via CUDA, AMD via HIP, or Intel "
        "via Level Zero), found with sycl-ls. Builds the task's C++ reference and, where "
        "possible, the ORIGINAL CUDA (nvcc, or hipify+hipcc on AMD); compiles the port ahead of "
        "time for the GPU; verifies it against both; benchmarks it. task_file uses the "
        "examples/*/task.json format. Writes results/, SUMMARY.md and summary.json under "
        "out_dir. For a GPU this machine does not have, use the CLI: `oneapi-agent bundle "
        "task.json` makes a one-file test to run there, `oneapi-agent ingest` imports the "
        "results." + STATUS_NOTE
    ),
)
def gpu_test(task_file: str, out_dir: str, vendor: str | None = None,
             device: str | None = None, allow_cpu: bool = False) -> dict[str, Any]:
    return _gpu_test(task_file, out_dir, vendor=vendor, device=device,
                     allow_cpu=allow_cpu).to_dict()


@mcp.tool(
    name="doctor",
    description="Report which oneAPI binaries (icpx, c2s, sycl-ls) and include dirs were found.",
)
def doctor() -> dict[str, Any]:
    report = toolchain_report()
    missing = [k for k in ("icpx", "c2s", "sycl-ls") if not report[k]]
    return {
        "tool": "doctor",
        "status": "ok" if not missing else "toolchain_missing",
        "summary": "all tools found" if not missing else f"missing: {', '.join(missing)}",
        "evidence": report,
        "limitations": [],
    }


@mcp.tool(
    name="providers_status",
    description=(
        "Read-only: for each GPU rental provider (runpod, vast, vultr, digitalocean), whether an "
        "API key is configured on this machine and where it comes from (environment variable or "
        "the OS password store; never the key itself), checked with one read-only API call, "
        "plus account facts such as balance. A provider without a key lists console_url (where "
        "the user creates one) and login_command (`oneapi-agent login <provider>`, which the "
        "USER runs in their own terminal). No tool takes a key: never ask for one, store it or "
        "pass it on. Nothing is created or deleted." + STATUS_NOTE
    ),
)
def providers_status() -> dict[str, Any]:
    from .keys import providers

    return providers().to_dict()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
