# Port report: softmax_hecbench

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 9 / 10
- **Tokens:** in 36, out 3635, cache write 13848, cache read 93525; est. cost $0.0897
- **Best passing candidate:** iteration 9, 0 helper(s), `runs/ablation-20261015T134507/toolkit/softmax_hecbench/20261015T134831889611Z/best`
- **Trace:** `runs/ablation-20261015T134507/toolkit/softmax_hecbench/20261015T134831889611Z/trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s) |
| 2 | read_file | ok | softmax.dp.cpp: 5597 chars |
| 2 | read_file | ok | softmax_input.hpp: 431 chars |
| 3 | write_file | ok | wrote softmax.cpp (3996 chars) |
| 3 | read_file | ok | softmax_input.hpp: 431 chars |
| 4 | write_file | ok | wrote softmax_input.hpp (431 chars) |
| 5 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/softmax_hecbench/20261015T134831889611Z/wor... |
| 6 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| 6 | scan_helpers | fail | 9 dpct helper use(s) remain in 1 file(s) (types_launch=4, device_queue_management=2, memory_management=2, include=1) |
| 7 | write_file | ok | wrote migrated/softmax.dp.cpp (165 chars) |
| 8 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/softmax_hecbench/20261015T134831889611Z/wor... |
| 9 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| 9 | scan_helpers | ok | no dpct helpers in 4 file(s) |
| 9 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 284.33 ms, p95 360.17 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e362c0142ce9afcb`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s)
- **compile** (it 8, sources `c68b104ac58adeb6`): `ok` — compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/softmax_hecbench/20261015T134831889611Z/work/build/candidate (0 warning(s))
- **scan_helpers** (it 9, sources `c68b104ac58adeb6`): `ok` — no dpct helpers in 4 file(s)
- **verify** (it 9, sources `c68b104ac58adeb6`): `ok` — PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06)
- **benchmark** (it 9, sources `c68b104ac58adeb6`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 284.33 ms, p95 360.17 ms

Current sources hash: `c68b104ac58adeb6`. Success requires compile, verify and zero-helper scan all on this hash.
