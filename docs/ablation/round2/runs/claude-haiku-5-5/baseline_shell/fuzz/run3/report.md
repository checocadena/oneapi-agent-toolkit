# Port report: fuzz

- **Result:** `model_finished` — model called finish (baseline: no loop-side success check)
- **Model:** `claude-haiku-5-5`
- **Iterations:** 8 / 10
- **Tokens:** in 32, out 4198, cache write 13045, cache read 73136; est. cost $0.0045
- **Best passing candidate:** none
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2) |
| 2 | read_file | ok | fuzz.dp.cpp: 4333 chars |
| 2 | compile | ok | compiled 1 source(s) -> work/bui... |
| 3 | run | ok | exit 0 |
| 4 | write_file | ok | wrote migrated/fuzz.sycl.cpp (2119 chars) |
| 4 | compile | ok | compiled 1 source(s) -> work/bui... |
| 5 | run | fail | exit 1: d(); i=b.find(b'data'); n=struct.unpack('<I',b[i+4:i+8])[0]; return struct.unpack('<%df'%(n//4),b[i+8:i+8+n])       ~~~~^^^^^^^^ ... |
| 6 | run | ok | exit 0 |
| 7 | compile | ok | compiled 1 source(s) -> work/bui... |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `5a25d8f34f8c89d4`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2)
- **compile** (it 7, sources `3a4416d8e6f05eac`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers:** not run
- **verify:** not run
- **benchmark:** not run

Current sources hash: `3a4416d8e6f05eac`. Success requires compile, verify and zero-helper scan all on this hash.
