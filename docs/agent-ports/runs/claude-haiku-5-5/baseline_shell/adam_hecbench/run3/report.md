# Port report: adam_hecbench

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-haiku-5-5`
- **Iterations:** 9 / 12
- **Tokens:** in 36, out 5529, cache write 12794, cache read 93408; est. cost $0.0053
- **Best passing candidate:** none
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 3 file(s); 0 DPCT warning(s) |
| 2 | read_file | ok | adam.dp.cpp: 4772 chars |
| 2 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |
| 3 | run | ok | exit 0 |
| 4 | write_file | ok | wrote migrated/adam.dp.cpp (4368 chars) |
| 4 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |
| 5 | run | fail | exit 1: ref.cpp:5:10: fatal error: adam_input.hpp: No such file or directory     5 \| #include "adam_input.hpp"       \|          ^~~~~~~... |
| 6 | run | ok | exit 0 |
| 7 | run | ok | exit 0 |
| 8 | compile | ok | compiled 1 source(s) -> /tmp/claude-0/-home-user-oneapi-agent-toolkit/683d0d3b-7a3a-5352-91e1-192161487740/scratchpad/wt-amd/runs/ablatio... |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `6ef9b58bc665b5cc`): `ok` — migrated 1 CUDA file(s) -> 3 file(s); 0 DPCT warning(s)
- **compile** (it 8, sources `0fb9a4f3629142b4`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `0fb9a4f3629142b4`. Success requires compile, verify and zero-helper scan all on this hash.
