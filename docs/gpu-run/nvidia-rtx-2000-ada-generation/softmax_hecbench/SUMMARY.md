# GPU test: softmax_hecbench

- **Result:** `ok`: PASS on NVIDIA RTX 2000 Ada Generation (cuda:0, nvidia): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 7.45e-09, atol 1e-06)
- **Device:** NVIDIA RTX 2000 Ada Generation (`cuda:0`), vendor `nvidia`
- **SYCL target flags:** `-fsycl-targets=nvptx64-nvidia-cuda`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 -I/tmp/tmp.avma5wL9W1/task /tmp/tmp.avma5wL9W1/task/softmax_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/tmp/tmp.avma5wL9W1/task /tmp/tmp.avma5wL9W1/task/softmax.cu -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.avma5wL9W1/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 7.45e-09, atol 1e-06) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 367.70 ms, p95 2068.95 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 317.49 ms, p95 841.38 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
