# Port report: fuzz

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 7 / 10
- **Tokens:** in 28, out 2592, cache write 14705, cache read 63348; est. cost $0.0754
- **Best passing candidate:** iteration 7, 0 helper(s), `runs/ablation-20261015T134507/toolkit/fuzz/20261015T134630871341Z/best`
- **Trace:** `runs/ablation-20261015T134507/toolkit/fuzz/20261015T134630871341Z/trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2) |
| 1 | list_files | ok | 3 file(s) |
| 2 | read_file | ok | fuzz.dp.cpp: 4333 chars |
| 3 | write_file | ok | wrote migrated/fuzz.dp.cpp (1778 chars) |
| 4 | compile | fail | compile failed with 1 error(s) at runs/ablation-20261015T134507/toolkit/fuzz/20261015T134630871341Z/work/... |
| 5 | write_file | ok | wrote migrated/fuzz.dp.cpp (1819 chars) |
| 6 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/fuzz/20261015T134630871341Z/work/build/cand... |
| 7 | verify | ok | PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS) |
| 7 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 7 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 262.64 ms, p95 277.52 ms |

## What failed and what fixed it

- it 4: `compile:fuzz.dp.cpp:27:no matching constructor for initialization of 'sycl::queue'` → fixed by iteration 6 after editing fuzz.dp.cpp

## Final evidence

- **migrate** (it 1, sources `5a25d8f34f8c89d4`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2)
- **compile** (it 6, sources `d86f4f2befdf6c61`): `ok` — compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/fuzz/20261015T134630871341Z/work/build/candidate (0 warning(s))
- **scan_helpers** (it 7, sources `d86f4f2befdf6c61`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 7, sources `d86f4f2befdf6c61`): `ok` — PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS)
- **benchmark** (it 7, sources `d86f4f2befdf6c61`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 262.64 ms, p95 277.52 ms

Current sources hash: `d86f4f2befdf6c61`. Success requires compile, verify and zero-helper scan all on this hash.
