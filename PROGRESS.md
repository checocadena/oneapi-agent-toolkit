# Progress

Status legend: ✅ done · 🟡 in progress · ⛔ blocked · ⬜ not started

| # | Milestone | Status | Notes |
|---|-----------|--------|-------|
| 0 | Environment probe | ✅ | Real toolchain installed. Details below. |
| 1 | Core tool library | ✅ | 5 tools + parsers; smoke-tested on the real toolchain |
| 2 | MCP server + CLI | ✅ | 5 tools + `doctor` over stdio; `oneapi-agent` CLI; registration docs for Claude Code, Codex, Cursor |
| 3 | Agent skill | ⬜ | |
| 4 | Standalone agent loop | ⬜ | |
| 5 | Examples | ⬜ | |
| 6 | Tests | ⬜ | |
| 7 | Docs | ⬜ | |

## Milestone 0: environment probe (2026-10-15, about 8 minutes)

Machine: Ubuntu 24.04.5, x86_64, 4 vCPU (Intel Xeon @ 2.80GHz), no GPU, Python 3.13, outbound traffic through an allow-list proxy.

| Attempt | Result |
|---------|--------|
| `apt` Intel oneAPI repo (`apt.repos.intel.com`) | ❌ proxy `403 Forbidden` on CONNECT; GPG key download also 403 |
| `registrationcenter-download.intel.com` offline installer | ❌ proxy 403 |
| `pip` | ⚠️ only runtimes exist (`dpcpp-cpp-rt`, `intel-sycl-rt`, `intel-cmplr-lib-rt` 2026.1.2); no compiler or c2s on PyPI |
| GitHub API (`api.github.com/repos/...`) | ❌ not enabled for this session |
| `api.anaconda.org` | ❌ proxy 403 |
| SYCLomatic `20240403/linux_release.tgz` (GitHub release download) | ✅ downloaded, `c2s --version` → dpct 19.0.0. Its helper headers fail to compile with icpx 2026 (see DECISIONS D2) |
| conda-forge `dpcpp_linux-64` + `intel-opencl-rt` via micromamba 2.9.0 | ✅ **icpx 2026.1.1**; `sycl-ls` lists `[opencl:cpu][opencl:0] Intel(R) Xeon(R) Processor @ 2.80GHz OpenCL 3.0`; a test kernel compiled and ran |
| conda-forge CUDA 12.4 headers | ⚠️ c2s: "CUDA version is newer than the latest supported version 12.3", then fatal `curand_mtgp32_kernel.h` not found |
| conda-forge CUDA 12.3 headers + libcurand-dev | ✅ c2s parses `.cu` files |
| SYCLomatic `20260701` release | ✅ **c2s = dpct 21.0.0**; migrated output compiles with icpx 2026.1.1 and runs |

To reproduce: `scripts/install-toolchain.sh`, then `source scripts/oneapi-env.sh`.
The runtime needs `LD_LIBRARY_PATH=$DPCPP_PREFIX/lib`; `oneapi-env.sh` sets it.

Not available: an Intel or NVIDIA GPU, `nvcc`, and the CUDA runtime.
CUDA versions of the examples are therefore **never executed**. The plain C++ reference is the ground truth for verify.
Only the OpenCL CPU SYCL device exists, so every benchmark number in this repo comes from that CPU device.

## Milestone 1: core tool library

All five tools are implemented in `src/oneapi_agent/tools/` and return `ToolResult`. A smoke test against the real toolchain (c2s fixture `tests/fixtures/reduce.cu`) gave:
- migrate → `ok`, 2 DPCT warnings (DPCT1009, DPCT1010), parsed from both stderr and the inline comments
- compile → `ok` (icpx 2026.1.1)
- scan_helpers → `fail`, 16 helper uses across 5 categories, plus 5 advisories
- benchmark → `ok`, real wall-clock on the OpenCL CPU device

## Milestone 2: MCP server + CLI

- `src/oneapi_agent/server.py` registers `migrate`, `compile`, `scan_helpers`, `verify`, `benchmark` and `doctor`. Each tool's description spells out its evidence fields.
- Works with mcp 1.x (`FastMCP`) and 2.x (`MCPServer`); see DECISIONS D9. Verified in-process with `list_tools` and `call_tool` on mcp 2.3.0.
- `oneapi-agent <tool>` CLI (`src/oneapi_agent/cli.py`) prints the same JSON. Exit codes: 0 = ok, 1 = fail, 2 = anything else.
- `docs/install.md` covers registration for Claude Code (`claude mcp add` / `.mcp.json`), Codex (`config.toml`) and Cursor (`.cursor/mcp.json`).
  Those configs follow each tool's documented format but were **not exercised inside those clients** here, since none is available in this sandbox.
