# Port report: fuzz

- **Result:** `success` — verify passed and zero dpct helpers on the current compiled sources
- **Model:** `claude-haiku-5-5`
- **Iterations:** 4 / 10
- **Tokens:** in 16, out 1736, cache write 10856, cache read 17382; est. cost $0.0024
- **Best passing candidate:** iteration 4, 0 helper(s), `best`
- **Trace:** `trace.jsonl`

## Iterations

| it | tool | status | summary |
|----|------|--------|---------|
| 1 | migrate | ok | migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2) |
| 2 | read_file | ok | fuzz.dp.cpp: 4333 chars |
| 2 | compile | ok | compiled 1 source(s) -> work/build/cand... |
| 3 | write_file | ok | wrote migrated/fuzz.dp.cpp (2084 chars) |
| 3 | compile | ok | compiled 1 source(s) -> work/build/cand... |
| 4 | scan_helpers | ok | no dpct helpers in 1 file(s) |
| 4 | verify | ok | PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS) |
| 4 | benchmark | ok | 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 224.56 ms, p95 238.98 ms |

## What failed and what fixed it

- (no compile/verify failures)

## Final evidence

- **migrate** (it 1, sources `5a25d8f34f8c89d4`): `ok` — migrated 1 CUDA file(s) -> 1 file(s); 4 DPCT warning(s) (DPCT1009x1, DPCT1010x1, DPCT1024x2)
- **compile** (it 3, sources `4a9cb8561d29ab18`): `ok` — compiled 1 source(s) -> work/build/candidate (0 warning(s))
- **scan_helpers** (it 4, sources `4a9cb8561d29ab18`): `ok` — no dpct helpers in 1 file(s)
- **verify** (it 4, sources `4a9cb8561d29ab18`): `ok` — PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS)
- **benchmark** (it 4, sources `4a9cb8561d29ab18`): `ok` — 5 run(s) each; Intel(R) Xeon(R) Processor @ 2.80GHz: median 224.56 ms, p95 238.98 ms

Current sources hash: `4a9cb8561d29ab18`. Success requires compile, verify and zero-helper scan all on this hash.
