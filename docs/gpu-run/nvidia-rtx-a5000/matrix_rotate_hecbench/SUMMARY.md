# GPU test: matrix_rotate_hecbench

- **Result:** `ok`: PASS on NVIDIA RTX A5000 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0, atol 0)
- **Device:** NVIDIA RTX A5000 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_86`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 /tmp/tmp.KGKnuT86bF/task/matrix_rotate_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native /tmp/tmp.KGKnuT86bF/task/matrix_rotate.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.KGKnuT86bF/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 356.49 ms, p95 436.12 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 396.19 ms, p95 444.62 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
