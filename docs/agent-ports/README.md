# CUDA in, agent's SYCL port out, verified on NVIDIA and AMD GPUs

The first end-to-end run of both halves of the toolkit, with no human edits in between:

1. **`oneapi-agent loop`** (claude-sonnet-5-5) ported five HeCBench CUDA programs it had never
   seen: stencil1d, matrix_rotate, atomic_reduction, scan and adam.
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

Same model, same five tasks, 3 runs each, every final port graded by a hidden compile + verify +
helper scan the model never sees (`scripts/ablation.py`, as in
[`docs/ablation/round2`](../ablation/round2/README.md)):

| Arm | correct + 0 dpct helpers | wrong output | helpers left, but said done | cost |
|---|---|---|---|---|
| toolkit | **15/15** | 0/15 | 0/15 | $0.86 |
| baseline (no verify/scan tools) | 6/15 | 0/15 | 9/15 | $0.86 |
| baseline + shell | 6/15 | 0/15 | 9/15 | $0.96 |

With the toolkit, each port took 4-7 iterations and $0.03-0.10. Without the checks, every arm
produced correct output on these programs; what it left behind was unfinished migration (dpct
helpers) while reporting success, as in round 2. Per-program table: [`SUMMARY.md`](SUMMARY.md);
traces, reports and final sources of all 45 runs: [`runs/`](runs/).

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
