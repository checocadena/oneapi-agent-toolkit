# GPU test: softmax_hecbench

- **Result:** `ok`: PASS on NVIDIA GeForce RTX 4090 (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 7.45e-09, atol 1e-06)
- **Device:** NVIDIA GeForce RTX 4090 (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 -I/tmp/tmp.s2UTLv4Bj4/task /tmp/tmp.s2UTLv4Bj4/task/softmax_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.s2UTLv4Bj4/task /tmp/tmp.s2UTLv4Bj4/task/softmax.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.s2UTLv4Bj4/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 7.45e-09, atol 1e-06) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 300.07 ms, p95 331.45 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 334.13 ms, p95 361.46 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
