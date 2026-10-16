# Port report: saxpy

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 6 / 10
- **Tokens:** in 24, out 1468, cache write 6048, cache read 30955; est. cost $0.0360
- **Best passing candidate:** iteration 3, 0 helper(s), `runs/ablation-20261015T134507/toolkit/saxpy/20261015T134612871245Z/best`
- **Trace:** `runs/ablation-20261015T134507/toolkit/saxpy/20261015T134612871245Z/trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1) |
| 2 | write_file | ok | wrote migrated/saxpy.dp.cpp (1406 chars) |
| 2 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/saxpy/20261015T134612871245Z/work/build/can... |
| 3 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 3 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06) |
| 4 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 241.19 ms, p95 254.79 ms |
| 6 | scan_helpers | ok | no dpct helpers in 1 file(s) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e53316efe9db7d67`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1)
- **compile** (it 2, sources `65ddef0d243de78e`): `ok` — compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/saxpy/20261015T134612871245Z/work/build/candidate (0 warning(s))
- **scan_helpers** (it 6, sources `65ddef0d243de78e`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 3, sources `65ddef0d243de78e`): `ok` — PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06)
- **benchmark** (it 4, sources `65ddef0d243de78e`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 241.19 ms, p95 254.79 ms

Current sources hash: `65ddef0d243de78e`. Success requires compile, verify and zero-helper scan all on this hash.
