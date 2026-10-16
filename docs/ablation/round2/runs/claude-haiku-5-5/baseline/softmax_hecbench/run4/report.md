# Port report: softmax_hecbench

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-haiku-5-5`
- **Iterations:** 5 / 10
- **Tokens:** in 20, out 6367, cache write 11515, cache read 26588; est. cost $0.0049
- **Best passing candidate:** none
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s) |
| 2 | compile | ok | compiled 1 source(s) -> wo... |
| 3 | read_file | ok | softmax.dp.cpp: 5597 chars |
| 3 | read_file | ok | softmax_input.hpp: 431 chars |
| 4 | write_file | ok | wrote migrated/softmax.dp.cpp (5004 chars) |
| 4 | compile | ok | compiled 1 source(s) -> wo... |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e362c0142ce9afcb`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s)
- **compile** (it 4, sources `f4df979c00e2b0e3`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `f4df979c00e2b0e3`. Success requires compile, verify and zero-helper scan all on this hash.
