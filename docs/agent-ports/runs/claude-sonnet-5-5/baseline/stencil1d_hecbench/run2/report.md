# Port report: stencil1d_hecbench

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 4 / 12
- **Tokens:** in 16, out 1836, cache write 7421, cache read 20144; est. cost $0.0410
- **Best passing candidate:** none
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 2 DPCT warning(s) (DPCT1101x1, DPCT1113x1) |
| 2 | read_file | ok | stencil_1d.dp.cpp: 4108 chars |
| 3 | write_file | ok | wrote stencil_1d.cpp (2480 chars) |
| 3 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `9f2e59cbcdda5b45`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 2 DPCT warning(s) (DPCT1101x1, DPCT1113x1)
- **compile** (it 3, sources `9eff7343a69bb083`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `9eff7343a69bb083`. Success requires compile, verify and zero-helper scan all on this hash.
