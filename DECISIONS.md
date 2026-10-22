# Decisions

These decisions were made during the autonomous overnight build, with no human available to ask.
Each entry gives the decision, the reason, and how to undo it.

## D1. Toolchain sources: conda-forge for DPC++, GitHub release for SYCLomatic
The sandbox's network policy blocks Intel's apt repo (`apt.repos.intel.com`) and
`registrationcenter-download.intel.com` with a proxy 403. PyPI only has runtimes
(`dpcpp-cpp-rt`, `intel-sycl-rt`), not the compiler. These routes did work:
- `dpcpp_linux-64` (Intel oneAPI DPC++/C++ 2026.1.1) + `intel-opencl-rt` from conda-forge, installed with micromamba.
- SYCLomatic `linux_release.tgz` from `github.com/oneapi-src/SYCLomatic/releases`.
- CUDA 12.3 headers from conda-forge. c2s needs CUDA headers to parse `.cu`; it does not need a GPU or driver.

`scripts/install-toolchain.sh` reproduces all of this. With apt access, Intel's official packages work equally well.
The tools find binaries on `PATH` or through `ONEAPI_AGENT_*` env overrides.

## D2. SYCLomatic 20260701 rather than the first release found (20240403)
The 2024 helper headers (`dpct/*.hpp`) do not compile with icpx 2026.1. Bindless-image
API drift gives `no member named 'channel_order'`. The helper headers must match the
compiler's era. 20260701 was the newest tag reachable; later monthly tags returned 404.

## D3. CUDA 12.3 headers (not 12.4+)
With 12.4, c2s warned that the CUDA version was newer than its latest supported version (12.3) and then failed on a missing curand header.
12.3 plus `libcurand-dev` parses cleanly.

## D4. Status vocabulary: `ok | fail | error | toolchain_missing | timeout`
- `fail`: the tool ran and the property does not hold (compile errors, numeric mismatch, helpers remain).
- `error`: the tool could not do its job (bad input, missing file, the reference crashed).
- `toolchain_missing`: always includes `failed_command` and `reason`.

`scan_helpers` returns `fail` when helpers remain, so that "zero helpers" is a hard pass/fail signal the loop can use.

## D5. verify command templates
Commands are strings or argv lists with `{input}`, `{input0..N}` and `{output}` placeholders.
A command without `{output}` is compared on its stdout, parsed as numbers.
Output files are sniffed: RIFF goes to WAV, `\x93NUMPY` to npy, text to numbers, anything else to raw float32.
`mode="auto"` selects audio when inputs or outputs are WAV.

If the reference command fails, the result is `error` (nothing to compare against).
If the candidate fails, the result is `fail`.

## D6. Audio null test passes on the residual *peak* (default ≤ -96 dBFS), not RMS
Peak is the stricter of the two measures, and it is what the spec asked for. RMS and null depth are reported as well.
When the residual is exactly zero, the dBFS values are `null` and `bit_exact: true`.
Using `null` keeps the JSON valid (no `-Infinity`).

The examples write 32-bit float WAV, because 16-bit PCM quantization alone limits a null to about -90 dBFS.
verify adds a limitation note when it sees 16-bit outputs.

## D7. ULP statistics are computed after rounding to float32 ("ULP-ish")
Outputs are usually float32, and text outputs are parsed into float64.
`max_ulp` can be used as an extra gate. It is off by default.

## D8. benchmark reports wall-clock process time only, per ONEAPI_DEVICE_SELECTOR
Devices come from `sycl-ls`, and each is selected with `ONEAPI_DEVICE_SELECTOR=<backend>:<index>`.
An optional `report_regex` captures the program's own kernel timing.
The tool never estimates or extrapolates numbers. Without sycl-ls, it runs without a selector and says so in `limitations`.

## D9. MCP SDK compatibility
The installed `mcp` is 2.x, which renamed `FastMCP` to `MCPServer`. The server imports `mcp.server.fastmcp.FastMCP` (mcp 1.x) and falls back to `mcp.server.mcpserver.MCPServer` (2.x).
The decorator API is the same in both.

## D10. compile auto-adds the dpct helper include dir
Raw c2s output always includes `<dpct/dpct.hpp>`, so `compile` adds `-I<c2s>/../include` when it can find it.
This is recorded in the evidence (`dpct_include_added`). Pass `add_dpct_include=False` to prove that refactored code no longer needs it.

## D11. No silent "free" runs: an unpriced model refuses to start
The per-run USD cap (`max_cost_usd`) needs a price per token. The built-in table covers
Claude models only, and before this change any other model name made the estimate `None`,
which silently disabled the cap. Now `ONEAPI_AGENT_PRICE_IN` / `ONEAPI_AGENT_PRICE_OUT`
(USD per million tokens, optional `ONEAPI_AGENT_PRICE_CACHED_IN`) price any model; an
override wins over the table. A real-model run with neither refuses to start.
`ONEAPI_AGENT_ALLOW_UNPRICED=1` is the explicit opt-in to run with only the token cap; it
prints a warning and the trace records a `warning` event. Cached input defaults to the full
input price under an override (over-estimating is the safe direction for a cap).

## D12. Real-model preconditions are checked before any work
A missing API key used to surface as an SDK `TypeError` traceback on the first model call,
after the reference build had already run. `preflight_real_model` checks the key and the
price first and exits with one line.

## D13. Agent-neutral access is documented, not claimed
`AGENTS.md` gives any agent the porting procedure and the CLI form of each tool (the CLI
prints the same `ToolResult` JSON the MCP server returns; exit code 0 ok, 1 fail, 2 other).
`docs/tool-result-schema.json` is generated from `ToolResult` and a test keeps them equal.
What is verified is stated as such: measured ports come from the bundled loop with Claude
models; MCP registration was checked in Claude Code, Gemini CLI, Cursor and Codex; nothing
else is claimed.

## D14. AGPL-3.0 plus commercial licenses (dual licensing)
The author wants others to be able to use and build on the toolkit, without anyone turning
it into a closed product without the author's involvement. MIT allowed exactly that, so
from this change the code is under AGPL-3.0-only, and the author offers separate commercial
licenses for closed-source use. Consequences:
- Versions up to and including `b08b687` stay MIT for anyone who has
  them; that cannot be revoked.
- Dual licensing needs the author to hold the rights to all code, so outside code
  contributions wait for a contributor license agreement (`CONTRIBUTING.md`).
- Third-party example code keeps its license (BSD-3-Clause from HeCBench, MIT from AMD);
  both are compatible with the AGPL.
- AGPL is an OSI-approved open-source license, unlike "non-commercial" licenses.

## D15. No release page offers an MIT-era snapshot
GitHub offers "Source code (zip)" for the commit a release is tagged on. The AMD toolchain
release was tagged on an MIT-era commit, so its page was a one-click MIT download. The same
asset (unchanged, same sha256) is re-published under `sycl-hip-nightly-2026-10-14-rocm7.0-r3` on a post-AGPL commit and the
old tag is deleted; `remote.py` and the workflow point at the new tag. (A first attempt, `-r2`,
was created on the old commit by the release editor's default target and is deleted too.) A local `v0.2.0` tag
was never pushed to GitHub, so no MIT-labelled version is published.


## D16. One model interface, providers behind it; Meta through its OpenAI-compatible API
The loop already spoke Anthropic-shaped messages through `ModelClient.create`, so providers
plug in under that interface (`models.py`) and nothing in the loop's control flow changed.
Meta's Muse models are reached through the Meta Model API's OpenAI-compatible chat
completions endpoint, with the loop's tool calls and tool results translated both ways,
using only the standard library (no new dependency). Its transport is a plain function, so
`--dry-run --provider meta` runs the real client against a scripted server: the Meta path
is tested before API access opens. A scripted run's model is named
`scripted-dry-run (meta client, <id>)` so no report can pass for a real Muse result. Muse
gets no built-in price until Meta's pricing page is confirmed; until then the price comes
from `ONEAPI_AGENT_PRICE_IN` / `_OUT`, and D11 still refuses an unpriced run.
