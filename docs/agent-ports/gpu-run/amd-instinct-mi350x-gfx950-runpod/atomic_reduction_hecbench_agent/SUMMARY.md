# GPU test: atomic_reduction_hecbench_agent

- **Result:** `ok`: PASS on gfx950:sramecc+:xnack- gfx950:sramecc+:xnack- [HIP 70051.83] (hip:0, amd): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 0, atol 0); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 0, atol 0)
- **Device:** gfx950:sramecc+:xnack- gfx950:sramecc+:xnack- [HIP 70051.83] (`hip:0`), vendor `amd`
- **SYCL target flags:** `-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx950`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 -I/tmp/tmp.B3ftncYlWn/task /tmp/tmp.B3ftncYlWn/task/reduction_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /opt/rocm/bin/hipcc -O2 -std=c++17 --offload-arch=gfx950 -I/tmp/tmp.B3ftncYlWn/task /tmp/tmp.B3ftncYlWn/out/work/reduction.hip.cpp -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.B3ftncYlWn/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 0, atol 0) |
| sycl_benchmark | `ok` | 10 run(s) each; hip:0: median 347.54 ms, p95 368.79 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 306.25 ms, p95 322.23 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
