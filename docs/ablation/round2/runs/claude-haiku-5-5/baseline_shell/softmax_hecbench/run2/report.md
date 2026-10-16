# Port report: softmax_hecbench

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-haiku-5-5`
- **Iterations:** 7 / 10
- **Tokens:** in 28, out 6339, cache write 13151, cache read 65769; est. cost $0.0055
- **Best passing candidate:** none
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s) |
| 2 | read_file | ok | softmax.dp.cpp: 5597 chars |
| 2 | compile | ok | compiled 1 source(s) -> /home/user/oneapi-agent-toolkit/runs/ablation-20261015T141448/baseline_shell/softmax_hecbench/20261015T1420290107... |
| 3 | write_file | ok | wrote migrated/softmax.dp.cpp (4564 chars) |
| 3 | run | ok | exit 0 |
| 4 | compile | ok | compiled 1 source(s) -> /home/user/oneapi-agent-toolkit/runs/ablation-20261015T141448/baseline_shell/softmax_hecbench/20261015T1420290107... |
| 5 | write_file | ok | wrote check/ref.cpp (1213 chars) |
| 5 | run | ok | exit 0 |
| 6 | compile | ok | compiled 1 source(s) -> /home/user/oneapi-agent-toolkit/runs/ablation-20261015T141448/baseline_shell/softmax_hecbench/20261015T1420290107... |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e362c0142ce9afcb`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s)
- **compile** (it 6, sources `a8aba7b21c01e17d`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `a8aba7b21c01e17d`. Success requires compile, verify and zero-helper scan all on this hash.
