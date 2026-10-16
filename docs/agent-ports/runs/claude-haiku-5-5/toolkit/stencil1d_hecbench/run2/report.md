# Port report: stencil1d_hecbench

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-haiku-5-5`
- **Iterations:** 11 / 12
- **Tokens:** in 44, out 3841, cache write 17922, cache read 133099; est. cost $0.0055
- **Best passing candidate:** iteration 7, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 2 DPCT warning(s) (DPCT1101x1, DPCT1113x1) |
| 2 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |
| 3 | scan_helpers | fail | 7 dpct helper use(s) remain in 1 file(s) (device_queue_management=2, types_launch=2, memory_management=2, include=1) |
| 3 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| 4 | read_file | ok | stencil_1d.dp.cpp: 4108 chars |
| 5 | write_file | ok | wrote migrated/stencil_1d.dp.cpp (3305 chars) |
| 6 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |
| 7 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| 7 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 8 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 8 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 452.72 ms, p95 463.37 ms |
| 9 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |
| 10 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 11 | scan_helpers | ok | no dpct helpers in 1 file(s) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `9f2e59cbcdda5b45`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 2 DPCT warning(s) (DPCT1101x1, DPCT1113x1)
- **compile** (it 9, sources `502468643b7285a9`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 11, sources `502468643b7285a9`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 7, sources `502468643b7285a9`): `ok` — PASS: 1/1 case(s) match (worst max abs err 0, atol 0)
- **benchmark** (it 8, sources `502468643b7285a9`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 452.72 ms, p95 463.37 ms

Current sources hash: `502468643b7285a9`. Success requires compile, verify and zero-helper scan all on this hash.
