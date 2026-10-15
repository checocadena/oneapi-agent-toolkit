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

## Renting a GPU automatically: `remote-test`

```bash
export RUNPOD_API_KEY=...        # or VAST_API_KEY / VULTR_API_KEY
oneapi-agent remote-test examples/softmax_hecbench/task.json --provider runpod --vendor nvidia
```

`remote-test` runs the whole job in one command:
1. Builds a bundle.
2. Rents the cheapest GPU in stock, trying the next one if a card sells out.
3. Starts a small agent on the machine and uploads the bundle.
4. Waits for the run and imports the results into `docs/gpu-run/`.
5. **Always deletes the machine and checks that it is gone**, including when something failed.

Spending is capped by `--max-minutes` (45 by default) and `--max-usd` ($2 by default), using the provider's own prices.

**Login.**
- Each provider reads one API key, from `RUNPOD_API_KEY`, `VAST_API_KEY`, `VULTR_API_KEY` or `OCEAN_API_KEY`.
- The key is only sent to that provider's API. It is never written to disk or into the machine.
- The machine's agent gets a fresh random token for each run.
- Runpod is reached through its HTTPS proxy.
- Vast.ai, Vultr and DigitalOcean machines are reached on a bare IP. There the agent serves TLS with a throwaway self-signed certificate made for the run (this needs `openssl`), and the client trusts only that certificate.

| Provider | GPUs used | Status (2026-10-15) |
|---|---|---|
| Runpod | NVIDIA (RTX A4500 / 2000 Ada / A4000 / …), AMD MI300X | **Verified**: softmax on an RTX 2000 Ada, PASS against the C++ reference and the original CUDA, about 1 min, about $0.005 ([results](gpu-run/nvidia-rtx-2000-ada-generation/)). AMD stock: none. |
| Vast.ai | NVIDIA (RTX A4000 / 3090 / A5000 / 4090) | Create and delete verified live (RTX 3090); a full run not yet completed. No AMD offers listed. |
| Vultr | NVIDIA A16 / A40 Cloud GPU | Create and delete verified live (A16); a full run not yet completed. AMD Instinct plans are listed but not deployable on demand. Note: the API key's access control must allow the caller's IP. |
| DigitalOcean | **AMD Instinct MI300X** (single GPU, `gpu-mi300x1-192gb`, ~$2.59/h, ROCm image `gpu-amd-base`) | Implemented from the API docs; unit-tested, not yet run live. New accounts may need a support request to enable AMD GPUs. Key: `OCEAN_API_KEY` or `DIGITALOCEAN_TOKEN`. |
| Intel | none to rent | No public API: use a bundle (below). |

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
