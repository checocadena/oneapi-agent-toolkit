# Examples

Each example contains:

- a plain C++ **reference** (the ground truth for `verify`),
- the **CUDA** version (the input to the migration),
- a hand-written **idiomatic SYCL** port in `sycl/` (the target end state, with no dpct helpers),
- a `task.json` for the agent loop (`oneapi-agent loop --config examples/<name>/task.json`).

| Example | What it is | Inputs | verify mode |
|---------|------------|--------|-------------|
| `fuzz/` | tanh soft-clip fuzz waveshaper (drive + gain) | WAV | audio null test, ≤ -96 dBFS |
| `fir/` | 64-tap windowed-sinc low-pass FIR; `__constant__` taps plus a `__shared__` tile | WAV | audio null test, ≤ -96 dBFS |
| `saxpy/` | `y = a*x + y`, the HPC baseline | generated in-program | array, atol 1e-6 |
| `softmax_hecbench/` | HeCBench `softmax-cuda`: two kernels, one with cooperative-groups warp reductions | generated in-program | array, atol 1e-6 |
| `gemm_cublas/` | cuBLAS `cublasSgemm` (column-major, alpha/beta) → oneMath `oneapi::math::blas::column_major::gemm` | generated in-program | array, atol 2e-3 (worst-case float32 dot-product bound; see `task.json`) |

`gemm_cublas` needs two libraries:
- oneMKL SYCL, for the raw c2s output, which calls `oneapi::mkl::`.
- oneMath, for the idiomatic port.

Install both with `WITH_ONEMATH=1 scripts/install-toolchain.sh`. `scripts/oneapi-env.sh` sets `ONEMKL_PREFIX` and `ONEMATH_PREFIX`, which its `task.json` flags expand.

The audio inputs are generated procedurally (`python examples/gen_inputs.py`):
- a log sine sweep from 20 Hz to 20 kHz plus -30 dBFS white noise, mono float32 at 48 kHz;
- a stereo PCM16 version at 44.1 kHz.

No audio files are committed.

## Running

```bash
source scripts/oneapi-env.sh           # or: source /opt/intel/oneapi/setvars.sh
python examples/gen_inputs.py
python examples/run_pipeline.py        # every example, real tools → examples/RESULTS.md
oneapi-agent loop --dry-run --config examples/saxpy/task.json   # scripted model, real tools
```

## What the real pipeline found

`examples/RESULTS.md` holds the actual output. Highlights:

- **The raw SYCLomatic output of `softmax_hecbench` compiles but fails `verify`:** 22,156 of 524,288 elements differ by up to 4.1e-6 (3,473 float32 ULP).
  - **Cause:** c2s mapped CUDA `expf` to `sycl::native::exp`, a reduced-precision builtin, and emitted **no DPCT warning** for it.
  - **Fix check:** replacing it with `sycl::exp` in an otherwise unchanged copy brings max abs error down to 9.3e-8, a PASS.
  - This is the "last mile" in miniature: the migration looked clean, and only behavioural verification caught the problem. `scan_helpers` now reports `sycl::native::` math as a `native_math` advisory.
- The migrated `softmax` also carries `[[sycl::reqd_sub_group_size(32)]]`, a CUDA warp-size assumption. The idiomatic port uses one work-group per slice with `sycl::reduce_over_group`, which works for any sub-group size.
- For `fuzz`, `fir` and `saxpy`, the raw c2s output already verified against the reference (fuzz -138.5 dBFS, fir -135.0 dBFS residual peaks; saxpy max abs err 4.8e-7). It still carries 22, 14 and 11 dpct helper uses respectively; the hand-written ports remove all of them and verify identically.

## Benchmark caveats (read before quoting numbers)

The only SYCL device in the build sandbox is the **OpenCL CPU device** (Intel Xeon @ 2.80GHz, 4 vCPU). No GPU was available.

`benchmark` reports **whole-process wall time**.
- For the SYCL builds that includes SYCL runtime start-up and JIT compilation of SPIR-V for the CPU device, about 200 ms here.
- The C++ references are native binaries with none of that overhead.

The programs' own `kernel_ms` (captured with `report_regex`) is the fairer kernel number. For `softmax`, `kernel_ms` is timed with `std::chrono` around the first launch, so it also includes JIT compilation.

These numbers say nothing about GPU performance. AOT compilation (`-fsycl-targets=spir64_x86_64`) and repeated runs are on the roadmap.

## Licensing and provenance

- `softmax_hecbench/` adapts `src/softmax-cuda/main.cu` from [HeCBench](https://github.com/zjin-lcf/HeCBench) (commit `f0c817aa`).
  - Copyright (c) 2020 Argonne National Laboratory and 2020- Oak Ridge National Laboratory (Zheming Jin), under the BSD 3-Clause License. See `softmax_hecbench/LICENSE.HeCBench`.
  - The kernels are unchanged. Input generation is a deterministic LCG instead of `rand()`, results are written to a file, and the argument order changed.
- All other examples were written for this repository under its MIT license.
