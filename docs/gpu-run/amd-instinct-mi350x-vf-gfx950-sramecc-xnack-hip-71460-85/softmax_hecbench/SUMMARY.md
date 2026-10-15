# GPU test: softmax_hecbench

- **Result:** `ok`: PASS on AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (hip:0, amd): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 4.69e-07, atol 1e-06)
- **Device:** AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (`hip:0`), vendor `amd`
- **SYCL target flags:** `-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx950`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 -I/tmp/tmp.H4U7IjIsE5/task /tmp/tmp.H4U7IjIsE5/task/softmax_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /opt/rocm/bin/hipcc -O2 -std=c++17 --offload-arch=gfx950 -I/tmp/tmp.H4U7IjIsE5/task /tmp/tmp.H4U7IjIsE5/out/work/softmax.hip.cpp -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.25e-07, atol 1e-06) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.H4U7IjIsE5/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 9.69e-08, atol 1e-06) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 4.69e-07, atol 1e-06) |
| sycl_benchmark | `ok` | 10 run(s) each; hip:0: median 183.52 ms, p95 184.10 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 155.96 ms, p95 160.09 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
