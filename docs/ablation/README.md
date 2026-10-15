# With vs without the toolkit's checks

The same model (`claude-sonnet-5-5`) and the same four tasks, 3 runs per task per arm, at most 10 iterations each.
Runner: [`scripts/ablation.py`](../../scripts/ablation.py). Per-run rows: [`results.json`](results.json).
Every run's trace, report and final sources: [`runs/<arm>/<example>/run<n>/`](runs/). The wrong softmax port is [`runs/baseline/softmax_hecbench/run1/final/`](runs/baseline/softmax_hecbench/run1/final/).

- **toolkit:** the normal loop. The agent has migrate, compile, scan_helpers, verify, benchmark and file tools. It is done only when compile, verify and a zero-helper scan all hold on the same sources.
- **baseline:** `--baseline`. The agent has migrate, compile and the file tools, and is told the tolerance. The run ends when the agent calls `finish`.

Both arms are then graded the same way by a hidden check the agent never sees (`--evaluate`): compile the final sources, run `verify` against the C++ reference, and count dpct helpers.

## Result

| Arm | correct output + 0 dpct helpers | wrong output | dpct helpers left | said "done" but wasn't | model cost |
|---|---|---|---|---|---|
| toolkit | **12/12** | 0/12 | 0/12 | 0/12 | $0.90 |
| baseline | **7/12** | 1/12 | 4/12 | 5/12 | $0.67 |

Per example: [SUMMARY.md](SUMMARY.md).

- In 4 of 12 baseline runs the agent called `finish` with the raw migration's dpct helpers still in the code (saxpy 11, FIR 14, softmax 9).
- In 1 softmax baseline run, the agent's port compiled cleanly but **332,679 of 524,288 outputs were wrong**. It had sized the grid for the device's smallest sub-group size, while the runtime picked a larger one. Its own `finish` summary said "I did not run the binary or compare its output with the reference."
- That run did replace c2s's `sycl::native::exp` with `sycl::exp`. The model knows about that pitfall; the bug it shipped was a different one. That is the case for behavioural checks over checklists.

## Limits

- 3 runs per cell, one model, OpenCL CPU device. Treat these as indicative, not a benchmark.
- The baseline agent has no shell, so it cannot write its own test harness. An agent with a shell could; whether it would is a separate question.
- The toolkit arm costs more ($0.90 vs $0.67) because it keeps working until the checks pass.
