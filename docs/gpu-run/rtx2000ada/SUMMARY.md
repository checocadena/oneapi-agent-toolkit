# GPU run summary (real tool output)

- nvcc: `/usr/local/cuda/bin/nvcc`
- SYCL compiler: `/opt/sycl/bin/clang++`

## fir

| step | status | summary |
|---|---|---|
| reference_build | `rc=0` | g++ -O2 -std=c++17 -I/workspace/repo/examples/fir/../common -I/workspace/repo/examples/fir /workspace/repo/examples/fir/fir_ref.cpp -o build/reference |
| cuda_build | `rc=0` | /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/workspace/repo/examples/common -I/workspace/repo/examples/fir /workspace/repo/examples/fir/fir.cu -o build/cuda |
| cuda_verify_vs_cpp | `ok` | PASS: 2/2 case(s) match (worst residual peak -135.0 dBFS, threshold -96.0 dBFS) |
| cuda_benchmark | `ok` | 10 run(s) each; none: median 336.73 ms, p95 404.30 ms |
| sycl_gpu_compile | `ok` | compiled 1 source(s) -> /workspace/gpu-run/work/fir/build/sycl_gpu (0 warning(s)) |
| sycl_verify_vs_cpp | `ok` | PASS: 2/2 case(s) match (worst residual peak -135.0 dBFS, threshold -96.0 dBFS) |
| sycl_verify_vs_cuda | `ok` | PASS: 2/2 case(s) match (bit-exact, threshold -96.0 dBFS) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 375.23 ms, p95 399.08 ms |
| cpp_benchmark | `ok` | 10 run(s) each; none: median 62.31 ms, p95 71.09 ms |

## fuzz

| step | status | summary |
|---|---|---|
| reference_build | `rc=0` | g++ -O2 -std=c++17 -I/workspace/repo/examples/fuzz/../common /workspace/repo/examples/fuzz/fuzz_ref.cpp -o build/reference |
| cuda_build | `rc=0` | /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/workspace/repo/examples/common /workspace/repo/examples/fuzz/fuzz.cu -o build/cuda |
| cuda_verify_vs_cpp | `ok` | PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS) |
| cuda_benchmark | `ok` | 10 run(s) each; none: median 347.76 ms, p95 493.95 ms |
| sycl_gpu_compile | `ok` | compiled 1 source(s) -> /workspace/gpu-run/work/fuzz/build/sycl_gpu (0 warning(s)) |
| sycl_verify_vs_cpp | `ok` | PASS: 2/2 case(s) match (worst residual peak -138.5 dBFS, threshold -96.0 dBFS) |
| sycl_verify_vs_cuda | `ok` | PASS: 2/2 case(s) match (bit-exact, threshold -96.0 dBFS) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 365.17 ms, p95 395.68 ms |
| cpp_benchmark | `ok` | 10 run(s) each; none: median 59.49 ms, p95 66.86 ms |

## saxpy

| step | status | summary |
|---|---|---|
| reference_build | `rc=0` | g++ -O2 -std=c++17 /workspace/repo/examples/saxpy/saxpy_ref.cpp -o build/reference |
| cuda_build | `rc=0` | /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native /workspace/repo/examples/saxpy/saxpy.cu -o build/cuda |
| cuda_verify_vs_cpp | `ok` | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06) |
| cuda_benchmark | `ok` | 10 run(s) each; none: median 357.74 ms, p95 402.51 ms |
| sycl_gpu_compile | `ok` | compiled 1 source(s) -> /workspace/gpu-run/work/saxpy/build/sycl_gpu (0 warning(s)) |
| sycl_verify_vs_cpp | `ok` | PASS: 1/1 case(s) match (worst max abs err 4.77e-07, atol 1e-06) |
| sycl_verify_vs_cuda | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 1e-06) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 363.79 ms, p95 550.12 ms |
| cpp_benchmark | `ok` | 10 run(s) each; none: median 58.66 ms, p95 79.98 ms |

## softmax_hecbench

| step | status | summary |
|---|---|---|
| reference_build | `rc=0` | g++ -O2 -std=c++17 -I/workspace/repo/examples/softmax_hecbench /workspace/repo/examples/softmax_hecbench/softmax_ref.cpp -o build/reference |
| cuda_build | `rc=0` | /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/workspace/repo/examples/softmax_hecbench /workspace/repo/examples/softmax_hecbench/softmax.cu -o build/cuda |
| cuda_verify_vs_cpp | `ok` | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| cuda_benchmark | `ok` | 10 run(s) each; none: median 329.19 ms, p95 383.48 ms |
| sycl_gpu_compile | `ok` | compiled 1 source(s) -> /workspace/gpu-run/work/softmax_hecbench/build/sycl_gpu (0 warning(s)) |
| sycl_verify_vs_cpp | `ok` | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| sycl_verify_vs_cuda | `ok` | PASS: 1/1 case(s) match (worst max abs err 7.45e-09, atol 1e-06) |
| sycl_benchmark | `ok` | 10 run(s) each; cuda:0: median 329.86 ms, p95 371.23 ms |
| cpp_benchmark | `ok` | 10 run(s) each; none: median 63.05 ms, p95 80.22 ms |
