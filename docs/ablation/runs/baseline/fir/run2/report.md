# Port report: fir

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 4 / 10
- **Tokens:** in 16, out 2250, cache write 8166, cache read 20256; est. cost $0.0470
- **Best passing candidate:** none
- **Trace:** `runs/ablation-20261015T134507/baseline/fir/20261015T134653885970Z/trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1101x1, DPCT1113x1) |
| 2 | read_file | ok | fir.dp.cpp: 4901 chars |
| 2 | read_file | ok | fir_coeffs.hpp: 893 chars |
| 3 | write_file | ok | wrote migrated/fir.dp.cpp (2722 chars) |
| 3 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/baseline/fir/20261015T134653885970Z/work/build/cand... |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `1703b272466056fc`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1101x1, DPCT1113x1)
- **compile** (it 3, sources `100183edebb2af91`): `ok` — compiled 1 source(s) -> runs/ablation-20261015T134507/baseline/fir/20261015T134653885970Z/work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `100183edebb2af91`. Success requires compile, verify and zero-helper scan all on this hash.
