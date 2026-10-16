# Port report: softmax_hecbench

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-haiku-5-5`
- **Iterations:** 8 / 10
- **Tokens:** in 32, out 6420, cache write 17688, cache read 92380; est. cost $0.0063
- **Best passing candidate:** iteration 8, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s) |
| 2 | read_file | ok | softmax.dp.cpp: 5597 chars |
| 2 | compile | ok | compiled 1 source(s) -> wor... |
| 3 | read_file | ok | softmax_input.hpp: 431 chars |
| 3 | scan_helpers | fail | 9 dpct helper use(s) remain in 1 file(s) (types_launch=4, device_queue_management=2, memory_management=2, include=1) |
| 4 | write_file | ok | wrote migrated/softmax.dp.cpp (5208 chars) |
| 4 | compile | ok | compiled 1 source(s) -> wor... |
| 5 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| 5 | scan_helpers | ok | no dpct helpers in 2 file(s) |
| 6 | scan_helpers | ok | no dpct helpers in 2 file(s) |
| 6 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 337.43 ms, p95 367.74 ms |
| 7 | compile | ok | compiled 1 source(s) -> wor... |
| 8 | scan_helpers | ok | no dpct helpers in 2 file(s) |
| 8 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e362c0142ce9afcb`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s)
- **compile** (it 7, sources `5dfa26299cc2eea9`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 8, sources `5dfa26299cc2eea9`): `ok` — no dpct helpers in 2 file(s)
- **verify** (it 8, sources `5dfa26299cc2eea9`): `ok` — PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06)
- **benchmark** (it 6, sources `5dfa26299cc2eea9`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 337.43 ms, p95 367.74 ms

Current sources hash: `5dfa26299cc2eea9`. Success requires compile, verify and zero-helper scan all on this hash.
