# GPU test: adam_hecbench_agent

- **Result:** `ok`: PASS on NVIDIA RTX 4000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 1e-05)
- **Device:** NVIDIA RTX 4000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvidia_gpu_sm_89`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 -I/tmp/tmp.Be0qV9dWk2/task /tmp/tmp.Be0qV9dWk2/task/adam_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.Be0qV9dWk2/task /tmp/tmp.Be0qV9dWk2/task/adam.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.Be0qV9dWk2/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 1e-05) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 357.25 ms, p95 377.02 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 341.71 ms, p95 389.84 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
