# Port report: saxpy

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 5 / 10
- **Tokens:** in 20, out 1242, cache write 8289, cache read 19991; est. cost $0.0372
- **Best passing candidate:** iteration 3, 0 helper(s), `runs/ablation-20261015T134507/toolkit/saxpy/20261015T134509755349Z/best`
- **Trace:** `runs/ablation-20261015T134507/toolkit/saxpy/20261015T134509755349Z/trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1) |
| 2 | write_file | ok | wrote migrated/saxpy.dp.cpp (1481 chars) |
| 2 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/saxpy/20261015T134509755349Z/work/build/can... |
| 3 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 3 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06) |
| 4 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 253.36 ms, p95 264.68 ms |
| 5 | scan_helpers | ok | no dpct helpers in 1 file(s) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e53316efe9db7d67`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1)
- **compile** (it 2, sources `8c9781bcd6d9eabc`): `ok` — compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/saxpy/20261015T134509755349Z/work/build/candidate (0 warning(s))
- **scan_helpers** (it 5, sources `8c9781bcd6d9eabc`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 3, sources `8c9781bcd6d9eabc`): `ok` — PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06)
- **benchmark** (it 4, sources `8c9781bcd6d9eabc`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 253.36 ms, p95 264.68 ms

Current sources hash: `8c9781bcd6d9eabc`. Success requires compile, verify and zero-helper scan all on this hash.
