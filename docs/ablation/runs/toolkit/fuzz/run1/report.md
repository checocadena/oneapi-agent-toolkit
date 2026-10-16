# Port report: fuzz

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-sonnet-5-5`
- **Iterations:** 10 / 10
- **Tokens:** in 40, out 4004, cache write 28340, cache read 144084; est. cost $0.1398
- **Best passing candidate:** iteration 10, 0 helper(s), `runs/ablation-20261015T134507/toolkit/fuzz/20261015T134509816592Z/best`
- **Trace:** `runs/ablation-20261015T134507/toolkit/fuzz/20261015T134509816592Z/trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2) |
| 1 | list_files | ok | 3 file(s) |
| 2 | read_file | ok | fuzz.dp.cpp: 4333 chars |
| 3 | write_file | ok | wrote fuzz.cpp (1758 chars) |
| 3 | compile | fail | compile failed with 1 error(s) at runs/ablation-20261015T134507/toolkit/fuzz/20261015T134509816592Z/work/... |
| 4 | read_file | ok | fuzz.cpp: 1758 chars |
| 5 | write_file | ok | wrote fuzz.cpp (1815 chars) |
| 5 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/fuzz/20261015T134509816592Z/work/build/cand... |
| 6 | verify | ok | PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS) |
| 6 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 7 | scan_helpers | fail | 22 dpct helper use(s) remain in 1 file(s) (macro=13, events_timing=3, device_queue_management=2, memory_management=2) |
| 8 | write_file | ok | wrote migrated/fuzz.dp.cpp (1859 chars) |
| 9 | compile | ok | compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/fuzz/20261015T134509816592Z/work/build/cand... |
| 9 | scan_helpers | ok | no dpct helpers in 2 file(s) |
| 10 | verify | ok | PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS) |
| 10 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 241.41 ms, p95 266.19 ms |

## What failed and what fixed it

- it 3: `compile:fuzz.cpp:27:no matching constructor for initialization of 'sycl::queue'` → fixed by iteration 5 after editing fuzz.cpp

## Final evidence

- **migrate** (it 1, sources `5a25d8f34f8c89d4`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2)
- **compile** (it 9, sources `756dc9a077f7b02e`): `ok` — compiled 1 source(s) -> runs/ablation-20261015T134507/toolkit/fuzz/20261015T134509816592Z/work/build/candidate (0 warning(s))
- **scan_helpers** (it 9, sources `756dc9a077f7b02e`): `ok` — no dpct helpers in 2 file(s)
- **verify** (it 10, sources `756dc9a077f7b02e`): `ok` — PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS)
- **benchmark** (it 10, sources `756dc9a077f7b02e`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 241.41 ms, p95 266.19 ms

Current sources hash: `756dc9a077f7b02e`. Success requires compile, verify and zero-helper scan all on this hash.
