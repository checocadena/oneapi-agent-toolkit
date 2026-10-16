# Port report: fuzz

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-haiku-5-5`
- **Iterations:** 8 / 10
- **Tokens:** in 32, out 3688, cache write 10730, cache read 66453; est. cost $0.0039
- **Best passing candidate:** none
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2) |
| 2 | read_file | ok | fuzz.dp.cpp: 4333 chars |
| 2 | compile | ok | compiled 1 source(s) -> work/bui... |
| 3 | write_file | ok | wrote fuzz_sycl.cpp (1875 chars) |
| 3 | compile | ok | compiled 1 source(s) -> work/bui... |
| 4 | run | ok | exit 0 |
| 5 | run | ok | exit 0 |
| 6 | run | ok | exit 0 |
| 7 | run | ok | exit 0 |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `5a25d8f34f8c89d4`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2)
- **compile** (it 3, sources `9ba642369fd44e41`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `9ba642369fd44e41`. Success requires compile, verify and zero-helper scan all on this hash.
