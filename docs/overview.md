# oneapi-agent-toolkit in one page

## The problem

SYCLomatic (`c2s`) migrates most of a CUDA code base to SYCL. What remains is the last
mile: code it could not translate, output full of `dpct::` helpers, and no proof that the
port still behaves like the original. Coding agents can do this work, but left alone they
drift back to CUDA idioms and report success without checking. SYCLomatic itself shows the
same failure mode: on HeCBench softmax it emitted code that compiled with **zero
warnings** while `expf` had silently become the reduced-precision `sycl::native::exp`
([`examples/RESULTS.md`](../examples/RESULTS.md)).

## The approach

Five tools, each returning **evidence, never a bare claim**: a status, the exact commands
that ran, parsed diagnostics, numbers, and an explicit list of limitations
([schema](tool-result-schema.json)).

| Tool | Does |
|---|---|
| `migrate` | runs SYCLomatic and parses its warnings |
| `compile` | builds with `icpx -fsycl` and parses diagnostics |
| `scan_helpers` | inventories every `dpct::` helper left, with suggested SYCL replacements |
| `verify` | runs reference and candidate on the same inputs and compares outputs (arrays, or audio in dBFS) |
| `benchmark` | times a command per SYCL device |

A port counts as done only when `verify` passes **and** `scan_helpers` finds zero helpers,
both on the same, freshly compiled source. Agents reach the tools three ways: an MCP server
(registered with one command), a CLI that prints the same JSON ([`AGENTS.md`](../AGENTS.md)),
or a bundled agent loop (`oneapi-agent loop`) that enforces the success rule itself. The
design follows the evidence-first approach of [REA](https://github.com/morluto/rea).

A second half checks ports on real GPUs: `oneapi-agent remote-test` rents a machine (Runpod,
Vast.ai, Vultr or DigitalOcean), builds the C++ reference and the original CUDA there
(`nvcc`, or `hipify-perl` + `hipcc` on AMD), verifies the SYCL port against both, and always
deletes the machine ([`gpu-testing.md`](gpu-testing.md)).

## The evidence

Each comparison graded every final port with a hidden compile + verify + helper scan that
the agent never saw. "Correct" below means correct output **and** zero dpct helpers.

| Experiment | With the toolkit | Without its checks | Without, but with a shell | Evidence |
|---|---|---|---|---|
| 4 workloads (audio waveshaper, 64-tap FIR, SAXPY, HeCBench softmax); Claude Sonnet and Haiku; 5 runs per workload and arm | **40/40** | 23/40 | 14/40 | [`ablation/round2`](ablation/round2/README.md) |
| 5 HeCBench programs the agent had never seen; same two models; 3 runs per program and arm | **30/30** | 18/30 | 17/30 | [`agent-ports`](agent-ports/README.md) |

- **What goes wrong without the checks.** One unchecked softmax port compiled cleanly and
  got 332,679 of 524,288 outputs wrong ([replay](https://checocadena.github.io/oneapi-agent-toolkit/docs/replay/baseline-softmax-fail.html)).
  Giving the agent a shell removed wrong output in round 2 but not unfinished migration:
  most misses are dpct helpers left in while the agent reports success.
- **End to end, no human edits.** The Sonnet toolkit ports of the five unseen programs went
  unedited through `remote-test`: **10/10 PASS**, 5 on NVIDIA GPUs and 5 on an AMD Instinct
  MI350X, each against the C++ reference and the original CUDA built on the same GPU
  ([`agent-ports`](agent-ports/README.md#the-agents-ports-on-gpus)).
- **Cost.** Toolkit-arm ports cost $0.03-0.10 each with Sonnet and $0.002-0.007 with Haiku.
- **Reproducible.** Every run's trace, report and final sources are committed; CI re-checks
  the hand-written ports and the SYCLomatic behaviour with the real toolchain on every push
  ([`check_claims.py`](../scripts/check_claims.py)).

## Honest limits

- **Models.** Every measured port used Claude models (Sonnet 5.5, Haiku 5.5) through the
  bundled loop. No other vendor's model has been run.
- **Agents.** MCP registration was checked with Claude Code, Gemini CLI, Cursor and Codex
  ([`agent-registration.md`](agent-registration.md)); no port made by those clients over MCP
  has been measured.
- **Scale.** 5 or 3 runs per cell, small single-file programs, grading on a CPU device:
  indicative, not a benchmark. Harder, multi-file CUDA projects are untested.
- **Hardware.** GPU checks ran on NVIDIA and AMD only. The Intel GPU path (Level Zero,
  `spir64`) exists but has not run on Intel hardware.
- **Toolchain.** Results are tied to the versions recorded in each run (SYCLomatic 21.0.0,
  icpx 2026.1.1, and the intel/llvm builds named in the GPU results); behaviour such as the
  `expf` substitution can change with them, and its failing tolerance depends on the CPU's
  instruction set (AVX-512 fails, AVX2 passes).
- **Packaging.** Not on PyPI yet; install from a clone.

## Roadmap

See the [README roadmap](../README.md#roadmap). Planned, with no results yet: evaluating
non-Claude models with the same hidden checks, Intel GPU validation, and multi-file
projects.
