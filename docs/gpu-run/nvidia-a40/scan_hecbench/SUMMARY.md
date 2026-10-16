# GPU test: scan_hecbench

- **Result:** `ok`: PASS on NVIDIA A40 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0, atol 0)
- **Device:** NVIDIA A40 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 -I/tmp/tmp.wzYqV44eL3/task /tmp/tmp.wzYqV44eL3/task/scan_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.wzYqV44eL3/task /tmp/tmp.wzYqV44eL3/task/scan.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.wzYqV44eL3/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 374.61 ms, p95 475.38 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 381.82 ms, p95 447.87 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
