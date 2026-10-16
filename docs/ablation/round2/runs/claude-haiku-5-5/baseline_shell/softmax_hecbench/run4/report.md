# Port report: softmax_hecbench

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-haiku-5-5`
- **Iterations:** 10 / 10
- **Tokens:** in 40, out 12388, cache write 19907, cache read 121750; est. cost $0.0099
- **Best passing candidate:** none
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s) |
| 2 | read_file | ok | softmax.dp.cpp: 5597 chars |
| 2 | compile | ok | compiled 1 source(s) -> /home/user/oneapi-agent-toolkit/runs/ablation-20261015T141448/baseline_shell/softmax_hecbench/20261015T1426387160... |
| 3 | write_file | ok | wrote migrated/softmax.dp.cpp (4713 chars) |
| 3 | run | ok | exit 0 |
| 4 | compile | ok | compiled 1 source(s) -> /home/user/oneapi-agent-toolkit/runs/ablation-20261015T141448/baseline_shell/softmax_hecbench/20261015T1426387160... |
| 5 | run | ok | exit 0 |
| 6 | run | ok | exit 0 |
| 7 | run | ok | exit 0 |
| 8 | write_file | ok | wrote migrated/softmax.dp.cpp (4943 chars) |
| 8 | run | ok | exit 0 |
| 9 | compile | ok | compiled 1 source(s) -> /home/user/oneapi-agent-toolkit/runs/ablation-20261015T141448/baseline_shell/softmax_hecbench/20261015T1426387160... |
| 9 | run | ok | exit 0 |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e362c0142ce9afcb`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s)
- **compile** (it 9, sources `b881ad91ce25ee83`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `b881ad91ce25ee83`. Success requires compile, verify and zero-helper scan all on this hash.
