# Port report: softmax_hecbench

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-haiku-5-5`
- **Iterations:** 4 / 10
- **Tokens:** in 16, out 4805, cache write 14302, cache read 18197; est. cost $0.0044
- **Best passing candidate:** iteration 4, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s) |
| 2 | compile | ok | compiled 1 source(s) -> wor... |
| 2 | read_file | ok | softmax.dp.cpp: 5597 chars |
| 3 | write_file | ok | wrote migrated/softmax.dp.cpp (4885 chars) |
| 3 | compile | ok | compiled 1 source(s) -> wor... |
| 4 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| 4 | scan_helpers | ok | no dpct helpers in 2 file(s) |
| 4 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 406.08 ms, p95 417.80 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e362c0142ce9afcb`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s)
- **compile** (it 3, sources `3345a7110ed90e80`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 4, sources `3345a7110ed90e80`): `ok` — no dpct helpers in 2 file(s)
- **verify** (it 4, sources `3345a7110ed90e80`): `ok` — PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06)
- **benchmark** (it 4, sources `3345a7110ed90e80`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 406.08 ms, p95 417.80 ms

Current sources hash: `3345a7110ed90e80`. Success requires compile, verify and zero-helper scan all on this hash.
