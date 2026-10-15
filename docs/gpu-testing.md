# Testing a port on NVIDIA, AMD and Intel GPUs

`oneapi-agent gpu-test` verifies a SYCL port on whatever GPU the machine has.

| Vendor | SYCL backend | Ahead-of-time target | Original CUDA built with |
|---|---|---|---|
| NVIDIA | `cuda` | `nvptx64-nvidia-cuda` | `nvcc` |
| AMD | `hip` | `amd_gpu_<gfx arch>` (from `rocminfo`) | `hipify-perl` + `hipcc` |
| Intel | `level_zero` (or `opencl` GPU) | `spir64` | none; the C++ reference is the ground truth |

It builds the task's C++ reference and, where the vendor allows, the original CUDA. It then compiles the port for the GPU, verifies it against both, and benchmarks it on that one device. Every step is a real tool call, and the results go to `results/*.json`, `SUMMARY.md` and `summary.json`.

```bash
oneapi-agent gpu-test examples/saxpy/task.json --out gputest-saxpy        # first GPU found
oneapi-agent gpu-test my/task.json --vendor amd                           # or pick a vendor
oneapi-agent gpu-test my/task.json --device level_zero:1                  # or an exact device
```

If `sycl-ls` lists no GPU, the result is `toolchain_missing`, together with the device list. The tool never falls back to a CPU silently; `--allow-cpu` exists for testing only.
Agents get the same command as the `gpu_test` MCP tool.

## A GPU this machine cannot reach: bundles

```bash
oneapi-agent bundle my/task.json            # -> gputest-<name>.sh, one self-contained file
```

The bundle carries everything the run needs:
- the task, its sources, include dirs and inputs, with their relative layout preserved;
- this toolkit's own source.

The GPU machine needs only a SYCL compiler and Python 3. numpy is installed if it is missing. Run it there in either way:

```bash
bash gputest-<name>.sh                      # SSH shell, CI job, batch script
```
```python
!bash gputest-<name>.sh                     # a JupyterLab notebook cell (works from a phone)
```

It prints the summary and writes `gputest-results-<name>-<host>.tar.gz`. Bring that file back with:

```bash
oneapi-agent ingest gputest-results-<name>-<host>.tar.gz    # -> docs/gpu-run/<vendor>-<device>/<task>/
```

`ingest` rejects archives that contain absolute paths or `..` entries.

## Where to get each GPU

| Vendor | Automatable today | How |
|---|---|---|
| NVIDIA | yes | Runpod via its REST/MCP API: create a pod, run, fetch the results over the pod's HTTPS proxy, delete. Done by hand for [docs/gpu-run/rtx2000ada](gpu-run/rtx2000ada/SUMMARY.md); automation is planned. |
| AMD | yes, when stock exists | Runpod MI300X / MI350X with the ROCm image; `scripts/amd-gpu-run.sh`. Stock was `NONE` on 2026-10-15. |
| Intel | **no public API found** | See below; use a bundle. |

**Intel, as of 2026-10-15.**

The old Intel Tiber AI Cloud API has been shut down:
- Its hosts (`us-region-{1..5}-sdk-api.cloud.intel.com`) no longer resolve.
- Its Terraform provider, `intel/terraform-provider-intelcloud`, is archived.

cloud.intel.com now runs "Intel Cloud Services":
- You reserve instances in its web console, and approval can take 2–3 business days.
- Reservations last at most 30 days.
- SSH goes through a bastion and then asks for an instance password.
- Its [docs](https://cloud.intel.com/docs/index.html) describe no API or CLI.
- Its JupyterLab sessions work from a browser, which makes the bundle route the practical one.

A self-hosted Intel Arc or Flex machine registered as a GitHub Actions runner is the only fully automatic option we know of.

The original Intel DevCloud for oneAPI was retired on 2024-10-31.
