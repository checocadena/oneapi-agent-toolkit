# Meta-powered feature: options (proposal, nothing built yet)

Constraint: real building time is about Oct 29 (afternoon) to Nov 3, plus short sessions on
Oct 26 to 28. Credit: $150 of Meta Model API usage. Costs below assume Muse Spark at the
event-page price of $1.25 / $4.25 per million input / output tokens (to confirm Oct 26) and
the token use of the existing Claude runs (about 20k to 60k tokens per small port).

## Option A: "Does Muse need the checks too?" (measured comparison)

Run Muse Spark through the same hidden-grader comparison as Claude: toolkit vs no checks vs
shell only, on the same 4 workloads and 5 HeCBench programs, same run counts.

- **Meta's role:** Muse is the agent doing every port; the result is a first public,
  reproducible measurement of Muse as a coding agent on GPU code.
- **Build:** almost none: the provider layer is done and `scripts/ablation.py` takes
  `--model`. A results page and replay pages for Muse runs.
- **Scope:** 1 to 2 days, mostly unattended runs. Fits the Oct 26 to 28 window.
- **Cost:** about 105 runs; estimate $3 to $10 of Muse credit (CPU grading; no GPUs). GPU
  re-check of the best ports: under $1.
- **Risk:** low. Whatever Muse scores is publishable; the claim is the comparison.

## Option B: Muse Port Planner (whole-repository analysis)

Muse reads a real multi-file CUDA repository and writes a porting plan the toolkit then
executes: every kernel and file, NVIDIA library calls mapped to oneAPI equivalents (cuBLAS
to oneMath BLAS, Thrust to oneDPL, cuFFT to oneMath DFT), CUDA-only assumptions flagged
(warp size 32, bf16, inline PTX), and an ordered work list with a test for each item.

- **Meta's role:** central; long-context code understanding is the whole feature.
- **Build:** a `plan` command (new: repository walker, a structured plan schema, the Muse
  call, a Markdown/JSON report), checked against `scan_helpers` and c2s warnings so the
  plan is grounded in evidence, not just model output.
- **Scope:** 2 to 3 days. **Cost:** a few dollars per repository.
- **Risk:** medium; plan quality is hard to measure. Mitigation: score the plan against
  what c2s and `scan_helpers` actually find.

## Option C: Muse Port (plan + execute on a real project)

B plus execution: the loop, driven by Muse, ports a chosen subset of a real repository
(target: llm.c's `dev/cuda` kernels, each with its own reference check; backup: thundersvm),
verified by the hidden checks and on a rented NVIDIA and AMD GPU.

- **Meta's role:** Muse plans and ports; the toolkit proves each step.
- **Build:** B, plus multi-file migrate (`c2s -p`), per-kernel tasks, and a project-level
  report. **Scope:** 4 to 5 days; the most impressive demo and the largest risk.
- **Cost:** $10 to $40 of Muse credit; a few dollars of GPU time.
- **Risk:** high on this timeline. Mitigation: fix the kernel subset on day one; anything
  not ported is reported as not ported.

## Recommendation

**A on Oct 26 to 28, then B with a small slice of C from Oct 29.** A needs almost no new
code, runs unattended while the developer is away, and produces measured Muse numbers by
Oct 29. B is the new Meta-centric feature. A few llm.c kernels ported end to end (the slice
of C) give the demo its "real repository" moment without betting the submission on a full
port. Each step is shippable on its own.

Decision needed from the author before any build: which option, the track, and the
spending cap for Muse runs.
