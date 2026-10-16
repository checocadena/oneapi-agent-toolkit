# Port report: saxpy

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-haiku-5-5`
- **Iterations:** 7 / 10
- **Tokens:** in 28, out 4101, cache write 8456, cache read 40713; est. cost $0.0035
- **Best passing candidate:** none
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1) |
| 2 | read_file | ok | saxpy.dp.cpp: 2272 chars |
| 3 | write_file | ok | wrote saxpy.sycl.cpp (2085 chars) |
| 3 | compile | ok | compiled 1 source(s) -> work/bu... |
| 4 | run | fail | exit 134: ble. Please check https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-dpcpp-system-requirements.html bas... |
| 5 | write_file | ok | wrote saxpy.sycl.cpp (1933 chars) |
| 5 | compile | ok | compiled 1 source(s) -> work/bu... |
| 5 | run | ok | exit 0 |
| 6 | run | ok | exit 0 |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `e53316efe9db7d67`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 1 DPCT warning(s) (DPCT1009x1)
- **compile** (it 5, sources `bd09230654d10619`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `bd09230654d10619`. Success requires compile, verify and zero-helper scan all on this hash.
