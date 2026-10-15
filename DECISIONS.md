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
