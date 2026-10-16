# Port report: stencil1d_hecbench

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-haiku-5-5`
- **Iterations:** 10 / 12
- **Tokens:** in 40, out 3585, cache write 19802, cache read 110168; est. cost $0.0054
- **Best passing candidate:** iteration 6, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 2 DPCT warning(s) (DPCT1101x1, DPCT1113x1) |
| 2 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |
| 3 | read_file | ok | stencil_1d.dp.cpp: 4108 chars |
| 3 | scan_helpers | fail | 7 dpct helper use(s) remain in 1 file(s) (device_queue_management=2, types_launch=2, memory_management=2, include=1) |
| 4 | write_file | ok | wrote migrated/stencil_1d.dp.cpp (3236 chars) |
| 5 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |
| 6 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| 6 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 7 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 7 | benchmark | ok | 10 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 277.57 ms, p95 475.87 ms |
| 8 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |
| 9 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 10 | scan_helpers | ok | no dpct helpers in 1 file(s) |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `9f2e59cbcdda5b45`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 2 DPCT warning(s) (DPCT1101x1, DPCT1113x1)
- **compile** (it 8, sources `cec512cb52171bdc`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 10, sources `cec512cb52171bdc`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 6, sources `cec512cb52171bdc`): `ok` — PASS: 1/1 case(s) match (worst max abs err 0, atol 0)
- **benchmark** (it 7, sources `cec512cb52171bdc`): `ok` — 10 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 277.57 ms, p95 475.87 ms

Current sources hash: `cec512cb52171bdc`. Success requires compile, verify and zero-helper scan all on this hash.
