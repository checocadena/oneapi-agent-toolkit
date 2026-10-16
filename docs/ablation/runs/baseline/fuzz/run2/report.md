# Port report: fuzz

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 4 / 10
- **Tokens:** in 16, out 1576, cache write 6603, cache read 18296; est. cost $0.0360
- **Best passing candidate:** none
- **Trace:** `runs/ablation-20261015T134507/baseline/fuzz/20261015T134636996571Z/trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2) |
| 2 | read_file | ok | fuzz.dp.cpp: 4333 chars |
| 3 | write_file | ok | wrote migrated/fuzz.dp.cpp (1829 chars) |
| 3 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/baseline/fuzz/20261015T134636996571Z/work/build/can... |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `5a25d8f34f8c89d4`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2)
- **compile** (it 3, sources `73d9ec6aba6da526`): `ok` — compiled 1 source(s) -> runs/ablation-20261015T134507/baseline/fuzz/20261015T134636996571Z/work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `73d9ec6aba6da526`. Success requires compile, verify and zero-helper scan all on this hash.
