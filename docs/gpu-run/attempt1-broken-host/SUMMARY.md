# GPU run summary (real tool output)

- nvcc: `/usr/local/cuda/bin/nvcc`
- SYCL compiler: `/opt/sycl/bin/clang++`

## fir

| step | status | summary |
|---|---|---|
| reference_build | `rc=0` | g++ -O2 -std=c++17 -I/workspace/repo/examples/fir/../common -I/workspace/repo/examples/fir /workspace/repo/examples/fir/fir_ref.cpp -o build/reference |
| cuda_build | `rc=0` | /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/workspace/repo/examples/common -I/workspace/repo/examples/fir /workspace/repo/examples/fir/fir.cu -o build/cuda |
| cuda_verify_vs_cpp | `fail` | candidate exited with code 1 |
| cuda_benchmark | `fail` | command failed on all 1 device(s) |
| sycl_gpu_compile | `ok` | compiled 1 source(s) -> /workspace/gpu-run/work/fir/build/sycl_gpu (0 warning(s)) |
| sycl_verify_vs_cpp | `fail` | candidate exited with code 1 |
| sycl_verify_vs_cuda | `error` | reference exited with code 1 |
| sycl_benchmark | `fail` | command failed on all 1 device(s) |
| cpp_benchmark | `ok` | 10 run(s) each; none: median 15.74 ms, p95 17.36 ms |

## fuzz

| step | status | summary |
|---|---|---|
| reference_build | `rc=0` | g++ -O2 -std=c++17 -I/workspace/repo/examples/fuzz/../common /workspace/repo/examples/fuzz/fuzz_ref.cpp -o build/reference |
| cuda_build | `rc=0` | /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/workspace/repo/examples/common /workspace/repo/examples/fuzz/fuzz.cu -o build/cuda |
| cuda_verify_vs_cpp | `fail` | candidate exited with code 1 |
| cuda_benchmark | `fail` | command failed on all 1 device(s) |
| sycl_gpu_compile | `ok` | compiled 1 source(s) -> /workspace/gpu-run/work/fuzz/build/sycl_gpu (0 warning(s)) |
| sycl_verify_vs_cpp | `fail` | candidate exited with code 1 |
| sycl_verify_vs_cuda | `error` | reference exited with code 1 |
| sycl_benchmark | `fail` | command failed on all 1 device(s) |
| cpp_benchmark | `ok` | 10 run(s) each; none: median 10.22 ms, p95 11.01 ms |

## saxpy

| step | status | summary |
|---|---|---|
| reference_build | `rc=0` | g++ -O2 -std=c++17 /workspace/repo/examples/saxpy/saxpy_ref.cpp -o build/reference |
| cuda_build | `rc=0` | /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native /workspace/repo/examples/saxpy/saxpy.cu -o build/cuda |
| cuda_verify_vs_cpp | `fail` | candidate exited with code -11 |
| cuda_benchmark | `fail` | command failed on all 1 device(s) |
| sycl_gpu_compile | `ok` | compiled 1 source(s) -> /workspace/gpu-run/work/saxpy/build/sycl_gpu (0 warning(s)) |
| sycl_verify_vs_cpp | `fail` | candidate exited with code 1 |
| sycl_verify_vs_cuda | `error` | reference exited with code -11 |
| sycl_benchmark | `fail` | command failed on all 1 device(s) |
| cpp_benchmark | `ok` | 10 run(s) each; none: median 21.03 ms, p95 21.72 ms |

## softmax_hecbench

| step | status | summary |
|---|---|---|
| reference_build | `rc=0` | g++ -O2 -std=c++17 -I/workspace/repo/examples/softmax_hecbench /workspace/repo/examples/softmax_hecbench/softmax_ref.cpp -o build/reference |
| cuda_build | `rc=0` | /usr/local/cuda/bin/nvcc -O2 -std=c++17 -arch=native -I/workspace/repo/examples/softmax_hecbench /workspace/repo/examples/softmax_hecbench/softmax.cu -o build/cuda |
| cuda_verify_vs_cpp | `fail` | FAIL: 0/1 case(s) match; first failure: 443667 of 524288 element(s) outside tolerance |
| cuda_benchmark | `ok` | 10 run(s) each; none: median 19.62 ms, p95 20.88 ms |
| sycl_gpu_compile | `ok` | compiled 1 source(s) -> /workspace/gpu-run/work/softmax_hecbench/build/sycl_gpu (0 warning(s)) |
| sycl_verify_vs_cpp | `fail` | candidate exited with code 1 |
| sycl_verify_vs_cuda | `fail` | candidate exited with code 1 |
| sycl_benchmark | `fail` | command failed on all 1 device(s) |
| cpp_benchmark | `ok` | 10 run(s) each; none: median 14.05 ms, p95 18.11 ms |
