# Port report: softmax_hecbench

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 8 / 10
- **Tokens:** in 32, out 3663, cache write 17492, cache read 75944; est. cost $0.0956
- **Best passing candidate:** iteration 8, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s) |
| 2 | read_file | ok | softmax.dp.cpp: 5597 chars |
| 2 | read_file | ok | softmax_input.hpp: 431 chars |
| 3 | write_file | ok | wrote softmax.cpp (4033 chars) |
| 3 | write_file | ok | wrote softmax_input.hpp (431 chars) |
| 4 | compile | ok | compiled 1 source(s) -> wor... |
| 5 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| 5 | scan_helpers | fail | 9 dpct helper use(s) remain in 1 file(s) (types_launch=4, device_queue_management=2, memory_management=2, include=1) |
| 6 | write_file | ok | wrote migrated/softmax.dp.cpp (81 chars) |
| 7 | compile | ok | compiled 1 source(s) -> wor... |
| 8 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| 8 | scan_helpers | ok | no dpct helpers in 4 file(s) |
| 8 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 1002.84 ms, p95 1298.74 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e362c0142ce9afcb`): `ok` — migrated 1 CUDA file(s) -> 2 file(s); 0 DPCT warning(s)
- **compile** (it 7, sources `084f4b4d4dcbaa5a`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 8, sources `084f4b4d4dcbaa5a`): `ok` — no dpct helpers in 4 file(s)
- **verify** (it 8, sources `084f4b4d4dcbaa5a`): `ok` — PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06)
- **benchmark** (it 8, sources `084f4b4d4dcbaa5a`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 1002.84 ms, p95 1298.74 ms

Current sources hash: `084f4b4d4dcbaa5a`. Success requires compile, verify and zero-helper scan all on this hash.
