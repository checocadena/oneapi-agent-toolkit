# Ablation round 2: 3 arms × 2 models × 4 tasks × 5 runs (120 runs)

Same design as [round 1](../README.md), with two additions:
- **A third arm, `baseline_shell`:** the baseline plus `run`, a bash shell in the work dir. That lets the agent build and run its own tests. The compiled reference binary is in the work dir in every arm.
- **A second, smaller model.**

Every final port is graded by the same hidden check, which the agent never sees: compile the final sources, `verify` against the C++ reference, and count dpct helpers.
- Runner: `scripts/ablation.py --runs 5 --arms toolkit baseline baseline_shell --model <m>`, max 10 iterations.
- Per-run rows: [`results.json`](results.json).
- Per example: [`SUMMARY.md`](SUMMARY.md).
- Each run's trace, report and final sources: [`runs/<model>/<arm>/<example>/run<n>/`](runs/).

| Model | Arm | correct + 0 dpct helpers | wrong output | helpers left | said "done" but wasn't | model cost |
|---|---|---|---|---|---|---|
| claude-sonnet-5-5 | toolkit | **20/20** | 0 | 0 | 0 | $1.37 |
| claude-sonnet-5-5 | baseline | **10/20** | 2 | 10 | 10 | $1.23 |
| claude-sonnet-5-5 | baseline + shell | **7/20** | 0 | 13 | 12 | $1.20 |
| claude-haiku-5-5 | toolkit | **20/20** | 0 | 0 | 0 | $0.08 |
| claude-haiku-5-5 | baseline | **13/20** | 4 | 5 | 7 | $0.07 |
| claude-haiku-5-5 | baseline + shell | **7/20** | 0 | 13 | 13 | $0.10 |

Across both models: toolkit **40/40**, baseline **23/40**, baseline + shell **14/40**.

## What it shows

- **The toolkit arm was correct and fully converted in every run, on both models.**
  - In 3 Haiku toolkit runs the loop hit its 10-iteration limit before its own success check was recorded. The final sources still passed the hidden check, and they are counted that way, as in every arm.
- **A shell fixes correctness, not completeness.**
  - With a shell, no agent shipped wrong output (0/40). Without one, 6/40 did.
  - Those 6 are the softmax sub-group bug (332,679 of 524,288 wrong) twice on Sonnet, and 4 Haiku ports whose binary exited with an error.
  - But with a shell, agents left SYCLomatic's dpct helpers in 26/40 runs, even though the prompt said to remove them. They tested that the code worked and stopped there.
  - `scan_helpers` is what gets the port to idiomatic SYCL.
- **Cost:** the toolkit arm costs about the same as the baseline (Sonnet $1.37 vs $1.23 for 20 runs).

## Limits

- 5 runs per cell, two models from one vendor, and the OpenCL CPU device only.
- "Correct" means within the task's tolerance on this machine's CPU (an AVX-512 Xeon). See `examples/README.md` on hardware dependence.
- The baseline prompts state the tolerance and ask for zero dpct helpers, so the agents were told the bar. They just had no tool that checked it.
