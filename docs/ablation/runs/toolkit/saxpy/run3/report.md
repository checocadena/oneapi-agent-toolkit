# Port report: saxpy

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 5 / 10
- **Tokens:** in 20, out 1422, cache write 5079, cache read 22759; est. cost $0.0315
- **Best passing candidate:** iteration 3, 0 helper(s), `runs/ablation-20261015T134507/toolkit/saxpy/20261015T134723266103Z/best`
- **Trace:** `runs/ablation-20261015T134507/toolkit/saxpy/20261015T134723266103Z/trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1) |
| 2 | write_file | ok | wrote migrated/saxpy.dp.cpp (1452 chars) |
| 2 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/saxpy/20261015T134723266103Z/work/build/can... |
| 3 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 3 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06) |
| 5 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 5 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 241.61 ms, p95 275.71 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e53316efe9db7d67`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1)
- **compile** (it 2, sources `13b4af056b44a3fb`): `ok` — compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/saxpy/20261015T134723266103Z/work/build/candidate (0 warning(s))
- **scan_helpers** (it 5, sources `13b4af056b44a3fb`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 3, sources `13b4af056b44a3fb`): `ok` — PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06)
- **benchmark** (it 5, sources `13b4af056b44a3fb`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 241.61 ms, p95 275.71 ms

Current sources hash: `13b4af056b44a3fb`. Success requires compile, verify and zero-helper scan all on this hash.
