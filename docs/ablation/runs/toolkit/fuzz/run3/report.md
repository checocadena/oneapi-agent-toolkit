# Port report: fuzz

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 8 / 10
- **Tokens:** in 32, out 2734, cache write 15649, cache read 74659; est. cost $0.0815
- **Best passing candidate:** iteration 8, 0 helper(s), `runs/ablation-20261015T134507/toolkit/fuzz/20261015T134751070117Z/best`
- **Trace:** `runs/ablation-20261015T134507/toolkit/fuzz/20261015T134751070117Z/trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2) |
| 2 | read_file | ok | fuzz.dp.cpp: 4333 chars |
| 2 | list_files | ok | 3 file(s) |
| 3 | write_file | ok | wrote fuzz.cpp (1818 chars) |
| 4 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/fuzz/20261015T134751070117Z/work/build/cand... |
| 5 | verify | ok | PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS) |
| 5 | scan_helpers | fail | 22 dpct helper use(s) remain in 1 file(s) (macro=13, events_timing=3, device_queue_management=2, memory_management=2) |
| 6 | write_file | ok | wrote migrated/fuzz.dp.cpp (1818 chars) |
| 6 | scan_helpers | ok | no dpct helpers in 2 file(s) |
| 7 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/fuzz/20261015T134751070117Z/work/build/cand... |
| 8 | verify | ok | PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS) |
| 8 | benchmark | ok | 10 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 252.12 ms, p95 304.70 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `5a25d8f34f8c89d4`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2)
- **compile** (it 7, sources `6723b28d5dd37fd2`): `ok` — compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/fuzz/20261015T134751070117Z/work/build/candidate (0 warning(s))
- **scan_helpers** (it 6, sources `6723b28d5dd37fd2`): `ok` — no dpct helpers in 2 file(s)
- **verify** (it 8, sources `6723b28d5dd37fd2`): `ok` — PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS)
- **benchmark** (it 8, sources `6723b28d5dd37fd2`): `ok` — 10 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 252.12 ms, p95 304.70 ms

Current sources hash: `6723b28d5dd37fd2`. Success requires compile, verify and zero-helper scan all on this hash.
