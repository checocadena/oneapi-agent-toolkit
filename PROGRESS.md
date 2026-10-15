# Progress

Status legend: ✅ done · 🟡 in progress · ⛔ blocked · ⬜ not started

| # | Milestone | Status | Notes |
|---|-----------|--------|-------|
| 0 | Environment probe | ✅ | Real toolchain installed. Details below. |
| 1 | Core tool library | ✅ | 5 tools + parsers; smoke-tested on the real toolchain |
| 2 | MCP server + CLI | ✅ | 5 tools + `doctor` over stdio; `oneapi-agent` CLI; registration docs for Claude Code, Codex, Cursor |
| 3 | Agent skill | ✅ | `skills/oneapi-port/SKILL.md` |
| 4 | Standalone agent loop | ✅ | `src/oneapi_agent/loop.py`; dry-run scenarios success/stuck/max_iterations verified. Real-model runs: 4/4 `success` on `claude-sonnet-5-5` (see below) |
| 5 | Examples | ✅ | fuzz, fir, saxpy, HeCBench softmax; real pipeline results in `examples/RESULTS.md` |
| 6 | Tests | ✅ | 78 tests: 78 pass with the toolchain; 75 pass + 3 skip without it; 73 pass + 5 skip in a minimal install without mcp/anthropic. ruff + mypy clean |
| 7 | Docs | ✅ | README (pitch, problem, Mermaid architecture, quickstart, real example output, status, roadmap, credits), docs/install.md, examples/README.md |

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

## Milestone 3: agent skill

`skills/oneapi-port/SKILL.md` covers:
- frontmatter for Claude Code auto-loading;
- ground rules: no CUDA idioms, evidence only, never claim success without a passing verify on the current code, report `toolchain_missing` honestly;
- the 8-step procedure, with per-category refactoring guidance;
- loop discipline: `max_iterations` (default 8), keep the best passing snapshot, the stuck rule (same error 3×), no unjustified tolerance widening;
- a fixed final evidence-report template.

The same rules are enforced in code by the standalone loop (M4).

## Milestone 4: standalone agent loop

`src/oneapi_agent/loop.py` and `oneapi-agent loop`:

- **Model:** the Anthropic SDK with a manual tool-use loop. The model comes from `ONEAPI_AGENT_MODEL` (default `claude-sonnet-5-5`) and the key from `ANTHROPIC_API_KEY`.
  - Assistant content blocks are passed back verbatim, so thinking blocks are preserved.
  - Automatic prompt caching is on.
  - Server-side refusal fallback (`fallbacks: "default"`) is on by default; disable it with `ONEAPI_AGENT_FALLBACKS=0`.
  - A `refusal` stop reason ends the run.
- **State:** goal, CUDA source, iteration count, a history of tool events (each tagged with a hash of the source tree), the best passing snapshot (`runs/<ts>/best/`), and token/cost usage.
- **Success is computed by the loop:** compile ok, verify ok and zero-helper scan, all on the *same current* source hash.
  - The model cannot choose verify's commands or tolerances, and cannot write outside `runs/<ts>/work/`.
  - `finish` is rejected unless the conditions hold.
- **Stop reasons:** `success`, `max_iterations`, `stuck` (same failure signature in 3 consecutive iterations), `budget` (tokens or USD, using the price table), `model_stopped`, `blocked` (toolchain missing), `error` (reference build failed).
- **Outputs:** `runs/<ts>/trace.jsonl` (every model response, tool call and tool result) and `report.md` (iteration table, failures and how each was fixed, final evidence).
- **`--dry-run`:** a scripted fake model. `--mock-tools` adds canned tool results that are clearly marked MOCK. Without `--mock-tools`, the scripted model drives the *real* tools.

Verified: the mocked scenarios end with `success` (7 iterations), `stuck` (3) and `max_iterations` (6).
Real model (later, with `ONEAPI_AGENT_API_KEY`): `claude-sonnet-5-5` reached `success` on saxpy (8 it.), fuzz (5), fir (6) and softmax (6) for about $0.30 in total. Traces in `docs/real-runs/`.

## Milestone 5: examples

Four examples, each with a C++ reference, a CUDA version, a hand-written idiomatic SYCL port and a `task.json`: `fuzz` (tanh soft clip), `fir` (64 taps), `saxpy`, and `softmax_hecbench` (BSD-3, HeCBench commit f0c817aa, license copied).
WAV inputs are generated procedurally by `examples/gen_inputs.py`.

`examples/run_pipeline.py` ran everything **for real** (icpx 2026.1.1, c2s 21.0.0, OpenCL CPU device):
- All 4 CUDA files migrate; all 4 raw migrations compile.
- Raw migrations verify for fuzz, fir and saxpy. **softmax fails verify**: c2s emitted `sycl::native::exp` with no warning. The cause was confirmed by a one-line fix check, and `scan_helpers` gained a `native_math` advisory as a result.
- All 4 hand-written ports compile **without** dpct headers, have 0 helpers, pass verify and were benchmarked.

The scripted dry run with real tools on saxpy ended in `success` in 7 iterations, including a deliberately broken edit and its fix. Its trace and report are committed in `docs/example-run/`.

Caveat: benchmarks are CPU-only and wall time includes JIT (see `examples/README.md`). The CUDA versions were later executed on an NVIDIA RTX 2000 Ada (see "NVIDIA GPU run" below).

## Milestone 6: tests

`tests/` holds 78 tests:

| File | Covers |
|------|--------|
| `test_parsers.py` | c2s/icpx parsers on **real captured output** in `tests/fixtures/`, the dpct scan, sycl-ls formats |
| `test_verify_math.py` | tolerances, NaN/inf, ULP, the audio null test at thresholds around -96 dBFS on synthetic signals, WAV PCM16/24/float/extensible, output sniffing |
| `test_tools.py` | every tool with fake runners; verify with real Python subprocesses (array, stdout and audio modes; crash semantics); `toolchain_missing` paths |
| `test_loop.py` | dry-run success, stuck and max_iterations paths; best-snapshot retention; token budget; a verify on a stale binary does not count; `finish` rejected without evidence; path confinement |
| `test_interfaces.py` | CLI and MCP server |
| `test_toolchain_real.py` | real toolchain; auto-skipped when it is absent |

One bug was found while writing the tests:
- **Symptom:** a fixture patched `find_tool` on the wrong object, so the tests were not doing what they claimed.
- **Cause:** `import oneapi_agent.tools.benchmark as tb` yields the re-exported *function*, not the submodule.
- **Fix:** the tests now use `importlib.import_module`.

CI (`.github/workflows/ci.yml`) runs ruff, mypy and pytest on Python 3.11 and 3.13 without the toolchain.

## Milestone 7: docs

- `README.md`:
  - every number in it comes from `examples/RESULTS.md` or `docs/example-run/`;
  - the example run is labelled as a *scripted* model driving real tools;
  - the Status table separates working, documented-but-not-exercised, and not-done.
- The README quickstart commands were run for real and all succeed.

## NVIDIA GPU run (Runpod)

`scripts/gpu-run.sh` on an RTX 2000 Ada pod (the first pod, an RTX 3070 community host, failed `cuInit` with error 999 and was replaced; that log is in `docs/gpu-run/attempt1-broken-host/`).
All four CUDA examples compile with nvcc and verify against the C++ references; all four SYCL ports compile with open-source DPC++ for `nvptx64-nvidia-cuda`, verify against both the C++ reference and the CUDA binary (fuzz and fir bit-exact, saxpy max err 0, softmax 7.45e-9). Results: `docs/gpu-run/rtx2000ada/`. The pod was deleted afterwards.

## Setup command

`oneapi-agent setup` registers the MCP server with Claude Code, Codex, Cursor and Gemini CLI, and installs the skill for Claude Code. Verified against the real `claude` CLI in a temporary HOME (`claude mcp list` → Connected; re-run leaves one entry). The other three are unit-tested config merges.

## Known gaps / next steps (honest list)

- One real-model run per example; no comparison yet against the same agent without the toolkit.
- No Intel GPU tested. GPU kernel timings are single small launches, not a performance claim.
- Codex, Cursor and Gemini registration is unit-tested but not exercised in those clients.
- The idiomatic SYCL ports in `examples/*/sycl/` were written by hand as target end states; the loop's own edits are recorded in `docs/real-runs/*/trace.jsonl`.
