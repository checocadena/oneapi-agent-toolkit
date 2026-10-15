# GPU test: adam_hecbench

- **Result:** `ok`: PASS on AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (hip:0, amd): sycl_verify_vs_reference: PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05); sycl_verify_vs_native: PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 1e-05)
- **Device:** AMD Instinct MI350X VF gfx950:sramecc+:xnack- [HIP 71460.85] (`hip:0`), vendor `amd`
- **SYCL target flags:** `-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfx950`

| step | status | summary |
|---|---|---|
| reference_build | `ok` | exit 0: g++ -O2 -std=c++17 -I/tmp/tmp.PLYv3KrhJb/task /tmp/tmp.PLYv3KrhJb/task/adam_ref.cpp -o build/reference |
| native_build | `ok` | exit 0: /opt/rocm/bin/hipcc -O2 -std=c++17 --offload-arch=gfx950 -I/tmp/tmp.PLYv3KrhJb/task /tmp/tmp.PLYv3KrhJb/out/work/adam.hip.cpp -o build/native |
| native_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) |
| sycl_compile | `ok` | compiled 1 source(s) -> /tmp/tmp.PLYv3KrhJb/out/work/build/sycl (0 warning(s)) |
| sycl_verify_vs_reference | `ok` | PASS: 1/1 case(s) match (worst max abs err 2.38e-07, atol 1e-05) |
| sycl_verify_vs_native | `ok` | PASS: 1/1 case(s) match (worst max abs err 5.96e-08, atol 1e-05) |
| sycl_benchmark | `ok` | 10 run(s) each; hip:0: median 204.76 ms, p95 206.70 ms |
| native_benchmark | `ok` | 10 run(s) each; none: median 169.90 ms, p95 173.25 ms |

Limitations:
- Benchmarks are single-process wall times plus program-reported kernel times; treat them as rough.
