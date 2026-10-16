# CUDA in, agent's SYCL port out, verified on NVIDIA and AMD GPUs

The first end-to-end run of both halves of the toolkit, with no human edits in between:

1. **`oneapi-agent loop`** (claude-sonnet-5-5, and claude-haiku-5-5 for the comparison below)
   ported five HeCBench CUDA programs it had never seen: stencil1d, matrix_rotate,
   atomic_reduction, scan and adam.
2. **`remote-test`** took the agent's final sources unchanged, rented a GPU and checked them
   against the C++ reference and against the original CUDA built natively on the same GPU
   (`nvcc`, or `hipify-perl` + `hipcc`). Every machine was deleted and the deletion confirmed.

The hand-written ports in `examples/*/sycl/` were not used anywhere here. The agent's file tools
are confined to its own work directory, so it could not read them.

## The agent's ports on GPUs

One port per program: the toolkit arm's run 1. Max abs error, SYCL vs ...

| Program | NVIDIA GPU | ... C++ reference | ... original CUDA (nvcc) | AMD MI350X: ... C++ reference | ... original CUDA (hipified) |
|---|---|---|---|---|---|
| stencil1d | L4 | 0 | 0 | 0 | 0 |
| matrix_rotate | RTX A4500 | 0 | 0 | 0 | 0 |
| atomic_reduction | RTX 2000 Ada | 0 | 0 | 0 | 0 |
| scan | L4 | 0 | 0 | 0 | 0 |
| adam | RTX 4000 Ada | 2.38e-7 | 5.96e-8 | 2.38e-7 | 5.96e-8 |

All 10 runs PASS (the tolerances are the tasks' own: exact for the four integer programs,
atol 1e-5 + rtol 1e-4 for adam). The NVIDIA GPU differs per program because `remote-test` takes
the cheapest card in stock at that moment. All runs were on Runpod (2026-10-16): about 1-2 min
per run; GPU spend for this page about $1.50 on AMD (including the failed attempts below) and a
few cents on NVIDIA.

- Task files with the agent's sources: [`gpu-tasks/`](gpu-tasks/) (`agent_port/` is the agent's
  final work directory, copied as is; `task.json` points `sycl_sources` at the file the agent
  last compiled, and records where it came from in `port_origin`).
- Full results per run: [`gpu-run/`](gpu-run/) (`SUMMARY.md`, `summary.json`, every tool result).

## The agent runs: with and without the toolkit

Two models, the same five tasks, 3 runs per cell, every final port graded by a hidden compile +
verify + helper scan the model never sees (`scripts/ablation.py`, as in
[`docs/ablation/round2`](../ablation/round2/README.md)). Correct and 0 dpct helpers:

| Model | toolkit | baseline (no verify/scan tools) | baseline + shell |
|---|---|---|---|
| claude-sonnet-5-5 | **15/15** | 6/15 | 6/15 |
| claude-haiku-5-5 | **15/15** | 12/15 | 11/15 |
| both | **30/30** | 18/30 | 17/30 |

- **Unfinished migration** is the main miss: outside the toolkit arm, 24 of 60 runs reported
  success with dpct helpers still in the code (Sonnet 18 of 30, Haiku 6 of 30).
- **One crash**: a Haiku baseline adam port hard-codes a GPU device selector, so it aborts
  ("No device of requested type gpu") on the CPU grader. No other run produced wrong output.
- **Cost**: Sonnet toolkit ports took 4-7 iterations and $0.03-0.10 each; Haiku toolkit ports
  4-11 iterations and $0.002-0.007 each. All 90 runs: $2.89.
- The GPU step below used the Sonnet toolkit ports.

Per-program table: [`SUMMARY.md`](SUMMARY.md); traces, reports and final sources of all 90 runs:
[`runs/`](runs/).

## What went wrong on the way (nothing here was edited out of the results)

- **First agent batch discarded.** It was started without `ONEAPI_AGENT_CUDA_INCLUDE` and
  `LD_LIBRARY_PATH`, so `c2s` exited 210 on every run and the hidden grader could not start the
  binaries (exit 127). Those runs measured the environment, not the agent, and were deleted
  before re-running with the same environment as round 2.
- **The benchmark step crashed on the Runpod ROCm image** (Ubuntu 22.04, numpy 1.21: no
  `method=` in `np.percentile`). Fixed in `tools/benchmark.py`, with a test; the run was
  repeated.
- **Runpod's proxy answered the bundle upload with an empty 404** right after `/status` had
  answered 200 (three machines in a row). `remote-test` now retries the upload on proxy errors;
  a retry cannot start a second run (the agent answers 409 and `/status` shows the run). The
  retry fired once in the final AMD runs.
- **MI350X stock**: only one or two at a time, so the AMD runs went one after another.
