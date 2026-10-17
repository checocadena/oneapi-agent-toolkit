# AGENTS.md: porting CUDA to SYCL with oneapi-agent-toolkit

Instructions for any coding agent working in this repository or using its tools. This is a
plain-text version of the [`oneapi-port` skill](skills/oneapi-port/SKILL.md); if they ever
differ, the skill is the reference.

## What has been verified, and what has not

- **Ports measured end to end** (every result in `docs/`) were made by the bundled loop,
  `oneapi-agent loop`, driving Claude models (`claude-sonnet-5-5`, `claude-haiku-5-5`)
  through the Anthropic SDK.
- **MCP registration** with `oneapi-agent setup` was checked against the real clients:
  Claude Code, Gemini CLI (`Connected`), Cursor (`ready`, all tools listed) and Codex
  (entry parsed). See [`docs/agent-registration.md`](docs/agent-registration.md). That
  shows the server starts and its tools are listed; no port made by those clients over
  MCP has been measured.
- **Any other agent** (or the CLI route below) is untested. The tools are ordinary
  commands that print JSON, so they should work anywhere a shell does, but nothing here
  claims that.

## Two ways to call the tools

**MCP** (Claude Code, Codex, Cursor, Gemini CLI, ...): `oneapi-agent setup --agent NAME`
registers the server. Tools: `migrate`, `compile`, `scan_helpers`, `verify`, `benchmark`,
`gpu_test`, `doctor`.

**CLI** (any agent with a shell): the same functions as commands. Each prints one
**ToolResult** as JSON on stdout ([schema](docs/tool-result-schema.json)) and exits with
**0 = ok, 1 = fail, 2 = anything else** (`error`, `toolchain_missing`, `timeout`). Put
`--summary` before the command for a one-line result instead of JSON.

| MCP tool | CLI equivalent (flags as in `oneapi-agent <cmd> --help`) |
|---|---|
| `doctor` | `oneapi-agent doctor` |
| `migrate(cuda_path, out_dir)` | `oneapi-agent migrate CUDA_PATH --out OUT_DIR [--in-root DIR] [--extra-arg=ARG ...]` |
| `compile(sources, flags, output)` | `oneapi-agent compile SRC [SRC ...] [--flags "-O2 -Iinc"] [--output BIN] [--no-dpct-include]` |
| `scan_helpers(path)` | `oneapi-agent scan-helpers PATH` |
| `verify(reference_cmd, candidate_cmd, ...)` | `oneapi-agent verify --ref "REF_CMD {output}" --cand "CAND_CMD {output}" [--input FILE ...] [--atol X] [--rtol X] [--max-ulp N] [--mode auto\|array\|audio] [--threshold-dbfs DB] [--work-dir DIR]` |
| `benchmark(cmd, runs, devices)` | `oneapi-agent benchmark "CMD ARGS" [--runs N] [--warmup N] [--device SELECTOR ...] [--report-regex RE]` |
| `gpu_test(task_file, out_dir)` | `oneapi-agent gpu-test TASK_JSON [--out DIR] [--vendor nvidia\|amd\|intel] [--device SELECTOR] [--runs N] [--no-native]` |

`migrate` needs SYCLomatic (`c2s`), `compile` needs `icpx`, and `gpu-test` needs a SYCL GPU
visible to `sycl-ls`; without them the result is `toolchain_missing` and its evidence names
what is missing. `scan-helpers` is a text scan and needs nothing; `verify` and `benchmark`
run whatever commands you give them (`benchmark` uses `sycl-ls`, when present, to time each
SYCL device). See [`docs/install.md`](docs/install.md).

Example: `oneapi-agent --summary verify --ref "./ref {output}" --cand "./port {output}" --atol 1e-6`
prints e.g. `[verify] ok: PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06)`.

## Ground rules

1. Target SYCL 2020 for Intel oneAPI DPC++ (`icpx -fsycl`). Do not keep CUDA idioms: no
   hard-coded warp size 32, no error-code plumbing, no 3-D launches for 1-D work. Use
   `sycl::queue`, USM, `parallel_for`, `nd_range`, group algorithms, `sycl::atomic_ref`.
2. Every claim cites a tool result: quote `status`, `summary` and the key evidence numbers.
   Never write "should work", "is equivalent" or "is faster" without one.
3. Never report success without `verify` returning `ok` on the current code. Editing a file
   after the last passing verify makes it unverified again.
4. Never fabricate. On `toolchain_missing`, say so, quote `evidence.failed_command`, and
   continue only with steps that do not need that tool.
5. Read `limitations` on every result and carry the relevant ones into your report.
6. Do not widen tolerances to pass. If a numerical reason forces it, report old and new.

## Procedure

```
migrate -> compile -> [fix errors] -> scan_helpers -> refactor to idiomatic SYCL
        -> compile (no dpct include) -> [fix] -> verify -> [fix] -> scan_helpers (= 0)
        -> benchmark -> report
```

1. **migrate.** Read `evidence.warnings`: each DPCT code marks something c2s could not
   translate faithfully (DPCT1009/1010 error codes, DPCT1049 work-group size, DPCT1065
   barrier hint, DPCT1083 kernel parameter layout). On `fail`, c2s could not parse
   something; pass include paths as `--extra-arg=-I<dir>` (repeatable; the `=` is required
   because the value starts with `-`).
2. **compile.** Fix the first error first; later ones are often cascades. Repeat until `ok`.
3. **scan_helpers** on the migrated tree gives the work list: `by_category`,
   `occurrences`, and `suggestions` per category.
4. **Refactor**, one category at a time:
   - device/queue management: one `sycl::queue` (in-order if the code relies on ordering,
     as CUDA's default stream does), passed explicitly;
   - memory: `sycl::malloc_device` / `malloc_shared`, `q.memcpy`, `sycl::free(p, q)`;
   - error handling: remove `dpct::err0` and `DPCT_CHECK_ERROR`; use
     `try { } catch (const sycl::exception&)`;
   - timing: the `sycl::event` from submit with `property::queue::enable_profiling`;
   - sub-groups: `sycl::shift_group_left`, `select_from_group`, `reduce_over_group`; never
     assume a sub-group size of 32;
   - atomics: `sycl::atomic_ref`; math: `sycl::` builtins;
   - act on `advisories` too (e.g. `nd_item3_for_1d`: use `range<1>` / `nd_range<1>`);
   - remove `#include <dpct/...>` last.
5. **compile** again without the dpct include (`--no-dpct-include`) to prove the helpers
   are gone. Fix until `ok`.
6. **verify** against the reference. Arrays: start near `atol` 1e-5 for float32. Audio:
   `--mode audio`, pass when the residual peak is at most -96 dBFS. On `fail`, use
   `first_mismatches` and `max_abs_err` to localise the bug (indexing and range rounding,
   missing bounds checks, reduction order, float vs double literals, denormals/FTZ).
7. **scan_helpers** must be `ok` (0 helpers) on the final code.
8. **benchmark** the final candidate; report only the numbers returned.

## Loop discipline

- One iteration per fix attempt after a failing compile or verify; stop at the iteration
  limit (default 8) and report the best version so far.
- Keep the best passing version: snapshot files whenever verify passes; never overwrite a
  passing snapshot with an unverified edit.
- Stuck rule: the same error three times in a row means stop trial-and-error; re-read the
  evidence and change approach, or report that you are stuck, quoting the error.

## Final report

```
## Port report: <name>
Result: SUCCESS | PARTIAL (best passing version kept) | BLOCKED (<reason>)
Iterations: N (compile fixes: a, verify fixes: b)
migrate:      <status> - <k> DPCT warnings (<codes>)
compile:      <status> - <errors/warnings>, dpct include needed: yes/no
scan_helpers: <status> - <before> -> <after> helpers
verify:       <status> - <cases>, max_abs_err=<x> / residual peak=<y> dBFS (threshold <t>)
benchmark:    <device>: median <m> ms, p95 <p> ms (runs=<n>)
What failed and how it was fixed:
- <error> -> <fix> (iteration i)
Limitations: <carried over from tool results>
```

SUCCESS only when verify is `ok` **and** scan_helpers is `ok` on the same final code.
