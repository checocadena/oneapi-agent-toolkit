# Port report: fuzz

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-haiku-5-5`
- **Iterations:** 10 / 10
- **Tokens:** in 40, out 4879, cache write 14648, cache read 106123; est. cost $0.0053
- **Best passing candidate:** none
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2) |
| 2 | read_file | ok | fuzz.dp.cpp: 4333 chars |
| 2 | compile | ok | compiled 1 source(s) -> work/bui... |
| 3 | write_file | ok | wrote migrated/fuzz.dp.cpp (2125 chars) |
| 3 | compile | ok | compiled 1 source(s) -> work/bui... |
| 4 | run | ok | exit 0 |
| 5 | run | ok | exit 0 |
| 6 | run | fail | exit 1: SYCL error: No device of requested type 'info::device_type::gpu' available. Please check https://software.intel.com/content/www/u... |
| 7 | run | ok | exit 0 |
| 8 | compile | ok | compiled 1 source(s) -> work/bui... |
| 8 | run | ok | exit 0 |
| 9 | run | ok | exit 0 |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `5a25d8f34f8c89d4`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2)
- **compile** (it 8, sources `9fe9ae5bc4dae320`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `9fe9ae5bc4dae320`. Success requires compile, verify and zero-helper scan all on this hash.
