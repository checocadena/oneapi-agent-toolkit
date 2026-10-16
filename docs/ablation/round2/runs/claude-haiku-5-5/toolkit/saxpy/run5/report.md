# Port report: saxpy

- **Result:** `max_iterations` — reached 10 iterations without meeting success conditions
- **Model:** `claude-haiku-5-5`
- **Iterations:** 10 / 10
- **Tokens:** in 40, out 2882, cache write 11384, cache read 77380; est. cost $0.0036
- **Best passing candidate:** iteration 7, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1) |
| 2 | compile | ok | compiled 1 source(s) -> work/build/can... |
| 3 | scan_helpers | fail | 11 dpct helper use(s) remain in 1 file(s) (events_timing=3, macro=2, device_queue_management=2, memory_management=2) |
| 4 | read_file | ok | saxpy.dp.cpp: 2272 chars |
| 5 | write_file | ok | wrote migrated/saxpy.dp.cpp (1871 chars) |
| 6 | compile | ok | compiled 1 source(s) -> work/build/can... |
| 7 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06) |
| 7 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 8 | benchmark | ok | 10 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 258.20 ms, p95 316.86 ms |
| 9 | scan_helpers | ok | no dpct helpers in 1 file(s) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e53316efe9db7d67`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1)
- **compile** (it 6, sources `567bedc0d007d05e`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 9, sources `567bedc0d007d05e`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 7, sources `567bedc0d007d05e`): `ok` — PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06)
- **benchmark** (it 8, sources `567bedc0d007d05e`): `ok` — 10 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 258.20 ms, p95 316.86 ms

Current sources hash: `567bedc0d007d05e`. Success requires compile, verify and zero-helper scan all on this hash.
