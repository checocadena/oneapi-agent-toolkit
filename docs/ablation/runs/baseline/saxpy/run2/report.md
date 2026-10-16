# Port report: saxpy

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 3 / 10
- **Tokens:** in 12, out 1196, cache write 2904, cache read 8371; est. cost $0.0209
- **Best passing candidate:** none
- **Trace:** `runs/ablation-20261015T134507/baseline/saxpy/20261015T134615323762Z/trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1) |
| 2 | write_file | ok | wrote migrated/saxpy.dp.cpp (1344 chars) |
| 2 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/baseline/saxpy/20261015T134615323762Z/work/build/ca... |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e53316efe9db7d67`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1)
- **compile** (it 2, sources `c0eb2f471d2ebae3`): `ok` — compiled 1 source(s) -> runs/ablation-20261015T134507/baseline/saxpy/20261015T134615323762Z/work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `c0eb2f471d2ebae3`. Success requires compile, verify and zero-helper scan all on this hash.
