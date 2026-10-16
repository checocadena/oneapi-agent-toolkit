# Port report: softmax_hecbench

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 6 / 10
- **Tokens:** in 24, out 3064, cache write 10680, cache read 48194; est. cost $0.0670
- **Best passing candidate:** iteration 4, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s) |
| 2 | read_file | ok | softmax.dp.cpp: 5597 chars |
| 2 | read_file | ok | softmax_input.hpp: 431 chars |
| 3 | write_file | ok | wrote migrated/softmax.dp.cpp (3885 chars) |
| 3 | compile | ok | compiled 1 source(s) -> wor... |
| 4 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| 4 | scan_helpers | ok | no dpct helpers in 2 file(s) |
| 5 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 377.57 ms, p95 478.29 ms |
| 6 | scan_helpers | ok | no dpct helpers in 2 file(s) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e362c0142ce9afcb`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s)
- **compile** (it 3, sources `a43760692488ccaf`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 6, sources `a43760692488ccaf`): `ok` — no dpct helpers in 2 file(s)
- **verify** (it 4, sources `a43760692488ccaf`): `ok` — PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06)
- **benchmark** (it 5, sources `a43760692488ccaf`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 377.57 ms, p95 478.29 ms

Current sources hash: `a43760692488ccaf`. Success requires compile, verify and zero-helper scan all on this hash.
