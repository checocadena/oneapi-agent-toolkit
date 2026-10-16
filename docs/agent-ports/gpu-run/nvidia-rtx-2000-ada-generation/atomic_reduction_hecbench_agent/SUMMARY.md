# GPU test: atomic_reduction_hecbench_agent

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0, atol 0)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 -I/tmp/tmp.p94u02Cq3l/task /tmp/tmp.p94u02Cq3l/task/reduction_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.p94u02Cq3l/task /tmp/tmp.p94u02Cq3l/task/reduction.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.p94u02Cq3l/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 432.01 ms, p95 452.11 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 401.11 ms, p95 446.12 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
