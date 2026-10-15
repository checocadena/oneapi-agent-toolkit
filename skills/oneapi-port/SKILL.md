---
name: oneapi-port
description: Port CUDA code to idiomatic, verified SYCL (oneAPI/DPC++) with evidence. Use when asked to migrate, port or convert CUDA (.cu/.cuh) to SYCL/oneAPI/DPC++, to clean up SYCLomatic/dpct output (dpct:: helpers, DPCT1xxx warnings), or to prove a SYCL port matches its reference. Requires the oneapi-agent MCP server (tools migrate, compile, scan_helpers, verify, benchmark).
---

# oneapi-port: last-mile CUDA → SYCL, evidence first

You are porting CUDA to **SYCL 2020** for Intel oneAPI DPC++ (`icpx -fsycl`).
SYCLomatic does the bulk translation. Your job is the last mile:

1. Fix what c2s left broken.
2. Remove `dpct::` helpers in favour of idiomatic SYCL.
3. **Prove** the port behaves like the reference.

Every claim you make must cite a tool result.

## Ground rules

- **Do not write CUDA.** Do not keep CUDA idioms such as warp size 32, `<<<>>>` thinking, error-code plumbing or 3-D launches for 1-D work. The target is SYCL 2020: `sycl::queue`, USM, `parallel_for`, `nd_range`, group algorithms, `sycl::atomic_ref`.
- **Evidence over claims.** Quote `status`, `summary` and the key evidence numbers from the tools. Never write "should work", "is equivalent" or "is faster" without a tool result that shows it.
- **Never report success without a passing `verify`** (`status: "ok"`) on the *current* version of the code. A file edited after the last passing verify counts as unverified.
- **Never fabricate.** If a tool returns `toolchain_missing`, say so, quote `evidence.failed_command`, and stop or continue only with steps that do not need that tool. Do not invent compiler output, timings or error values.
- **Read `limitations`** on every result and carry the relevant ones into your report.

## Inputs to establish first

| Item | Example |
|------|---------|
| CUDA source (file or dir) | `examples/saxpy/saxpy.cu` |
| Reference command (ground truth) | `./saxpy_ref {output}` (a plain C++ build, or the CUDA binary on an NVIDIA box) |
| Candidate build + run | `icpx -fsycl ... -o saxpy_sycl`, then `./saxpy_sycl {output}` |
| Test inputs | `examples/data/sweep_48k.wav`, or none |
| Tolerance | `atol` for arrays; `threshold_dbfs` (default -96) for audio |
| `max_iterations` | default **8** fix cycles; the user may override |

If no reference exists, ask for one or build the plain C++/CUDA original yourself. Verification without a reference is impossible; say so.

## Procedure

```
migrate → compile → [fix errors ↺] → scan_helpers → refactor to idiomatic SYCL
        → compile → [fix ↺] → verify → [fix ↺] → scan_helpers (=0) → benchmark → report
```

1. **migrate** `(cuda_path, out_dir)`.
   - Read `evidence.warnings`. Each DPCT code marks a place c2s could not translate faithfully. Common ones:
     - DPCT1009/1010: error codes. SYCL uses exceptions.
     - DPCT1049: work-group size may exceed the device limit.
     - DPCT1065: barrier performance hint.
     - DPCT1083: kernel parameter layout.
   - `status: "fail"` with `errors` means c2s could not parse something (often missing CUDA headers or include paths). Pass `extra_args: ["-I..."]`.
2. **compile** the migrated sources.
   - On `fail`, read `evidence.diagnostics`. Fix the **first error** first; later errors are often cascades.
   - Recompile after each fix. Loop until `ok`.
3. **scan_helpers** on the migrated tree. This gives the work list: `by_category`, `occurrences` and per-category `suggestions`.
4. **Refactor to idiomatic SYCL.** Work category by category and keep each edit small:
   - `device_queue_management`: create one `sycl::queue` (in-order if the code relies on ordering) and pass it explicitly.
   - `memory_management`: use `sycl::malloc_device` / `malloc_shared`, `q.memcpy`, `sycl::free(p, q)`. Replace `constant_memory` with a kernel argument or USM.
   - `error_handling`: delete `dpct::err0` and `DPCT_CHECK_ERROR`. Use `try { } catch (const sycl::exception&)`.
   - `events_timing`: use the `sycl::event` from submit, plus `property::queue::enable_profiling`.
   - `subgroup_warp`: use `sycl::shift_group_left`, `sycl::select_from_group` and `sycl::reduce_over_group`. Do **not** hardcode a sub-group size of 32.
   - `atomics`: use `sycl::atomic_ref`.
   - `math_intrinsics`: use `sycl::` builtins.
   - Also act on `advisories`. For example, `nd_item3_for_1d` means converting to `range<1>` / `nd_range<1>`.
   - Remove `#include <dpct/...>` last.
   - Prefer removing whole categories at once over sprinkling partial changes.
5. **compile** again, and pass `add_dpct_include: false` to prove the helper headers are no longer needed. Fix until `ok`.
6. **verify** reference against candidate.
   - Array outputs: start with `tolerance` (atol) around 1e-5 for float32 and widen only with a numerical justification, such as a different reduction order. Then state that justification.
   - Audio: `mode: "audio"` with float32 WAV outputs, pass when `residual_peak_dbfs ≤ -96`.
   - On `fail`, use `first_mismatches`, `max_abs_err` and per-channel `peak_frame` to localise the bug. Typical causes: indexing or range rounding, a missing bounds check, a changed reduction order, `float` vs `double` literals, and the denormal/FTZ difference (`-fp-model=precise` might be needed).
7. **scan_helpers** must return `ok` (total = 0) on the final code.
8. **benchmark** the final candidate. If a reference binary exists, benchmark it too with `devices: ["none"]`. Report only the numbers returned.

## Loop discipline

- Count one iteration per fix attempt after a failing compile or verify. Stop at `max_iterations` and report the best version so far.
- **Keep the best passing version.** Each time verify passes, snapshot the files, for example by copying them to `best/`. "Best" means passing verify with the fewest remaining helpers. Never overwrite a passing snapshot with an unverified edit.
- **Stuck rule:** if the same error (same file, line and message, or the same failing verify reason) appears 3 times in a row, stop changing that code by trial and error. Re-read the evidence, then try a structurally different fix or report that you are stuck, quoting the error.
- Do not widen tolerances to make a failing verify pass unless you can justify the error numerically. If you widen one, report the old and new tolerance.

## Final report (always, short)

```
## Port report: <name>
Result: SUCCESS | PARTIAL (best passing version kept) | BLOCKED (<reason>)
Iterations: N (compile fixes: a, verify fixes: b)
migrate:      <status> — <k> DPCT warnings (<codes>)
compile:      <status> — <errors/warnings>, dpct include needed: yes/no
scan_helpers: <status> — <before> → <after> helpers
verify:       <status> — <cases>, max_abs_err=<x> / residual peak=<y> dBFS (threshold <t>)
benchmark:    <device>: median <m> ms, p95 <p> ms (runs=<n>)  [reference: …]
What failed and how it was fixed:
- <error> → <fix> (iteration i)
Limitations: <carried over from tool results>
```

Write SUCCESS only when verify is `ok` **and** scan_helpers is `ok` on the same final code.
