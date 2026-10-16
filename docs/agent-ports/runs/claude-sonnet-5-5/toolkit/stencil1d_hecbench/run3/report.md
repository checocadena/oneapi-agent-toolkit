# Port report: stencil1d_hecbench

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 4 / 12
- **Tokens:** in 16, out 1790, cache write 7613, cache read 21860; est. cost $0.0413
- **Best passing candidate:** iteration 4, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 2 DPCT warning(s) (DPCT1101x1, DPCT1113x1) |
| 2 | read_file | ok | stencil_1d.dp.cpp: 4108 chars |
| 3 | write_file | ok | wrote migrated/stencil_1d.dp.cpp (2701 chars) |
| 3 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |
| 4 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 4 | verify | ok | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| 4 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 226.74 ms, p95 265.43 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `9f2e59cbcdda5b45`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 2 DPCT warning(s) (DPCT1101x1, DPCT1113x1)
- **compile** (it 3, sources `57be9b25f8cef53a`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 4, sources `57be9b25f8cef53a`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 4, sources `57be9b25f8cef53a`): `ok` — PASS: 1/1 case(s) match (worst max abs err 0, atol 0)
- **benchmark** (it 4, sources `57be9b25f8cef53a`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.10GHz: median 226.74 ms, p95 265.43 ms

Current sources hash: `57be9b25f8cef53a`. Success requires compile, verify and zero-helper scan all on this hash.
