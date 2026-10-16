# Port report: saxpy

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 6 / 10
- **Tokens:** in 24, out 1541, cache write 6201, cache read 31081; est. cost $0.0372
- **Best passing candidate:** iteration 3, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1) |
| 2 | write_file | ok | wrote migrated/saxpy.dp.cpp (1396 chars) |
| 2 | compile | ok | compiled 1 source(s) -> work/build/can... |
| 3 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 3 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06) |
| 4 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 285.90 ms, p95 344.97 ms |
| 6 | scan_helpers | ok | no dpct helpers in 1 file(s) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e53316efe9db7d67`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1)
- **compile** (it 2, sources `f5330fc2b492eb57`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 6, sources `f5330fc2b492eb57`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 3, sources `f5330fc2b492eb57`): `ok` — PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06)
- **benchmark** (it 4, sources `f5330fc2b492eb57`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 285.90 ms, p95 344.97 ms

Current sources hash: `f5330fc2b492eb57`. Success requires compile, verify and zero-helper scan all on this hash.
