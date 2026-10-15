# Testing a port on NVIDIA, AMD and Intel GPUs

`oneapi-agent gpu-test` verifies a SYCL port on whatever GPU the machine has.

| Vendor | SYCL backend | Ahead-of-time target | Original CUDA built with |
|---|---|---|---|
| NVIDIA | `cuda` | `nvptx64-nvidia-cuda` | `nvcc` |
| AMD | `hip` | `amdgcn-amd-amdhsa` + `--offload-arch=<gfx arch>` (from `rocminfo`; works for gfx950, which has no `amd_gpu_*` alias) | `hipify-perl` + `hipcc` |
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
oneapi-agent login runpod        # once; or export RUNPOD_API_KEY=... (VAST_API_KEY / VULTR_API_KEY / OCEAN_API_KEY)
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
- `oneapi-agent login <provider>` (`runpod`, `vast`, `vultr` or `digitalocean`) asks for the API key with a hidden prompt, or reads it from stdin when piped (`op read ... | oneapi-agent login vast`), and saves it in the OS password store: macOS Keychain, Windows Credential Manager or Linux Secret Service. It needs the `keyring` extra: `pip install 'oneapi-agent-toolkit[keyring]'`. `oneapi-agent logout <provider>` removes it.
- A key is looked up in this order: the environment variable (`RUNPOD_API_KEY`, `VAST_API_KEY`, `VULTR_API_KEY`, `OCEAN_API_KEY` or `DIGITALOCEAN_TOKEN`), then the password store. So CI and anything that exports a variable keep working, and a variable overrides a saved key.
- `oneapi-agent providers` shows, for each provider, whether a key was found and where (variable name or `keyring`, never the key), and makes one read-only API call to check the provider accepts it. It also shows account facts the API returns: Runpod balance and spend limit, Vast.ai credit, Vultr balance and pending charges, DigitalOcean account status and droplet limit. It never creates or deletes anything.
- Coding agents use the read-only MCP tool `providers_status` (same report). No MCP tool takes a key. When a key is missing, `remote-test` and `providers_status` return the provider, the console URL where the key is created, and `oneapi-agent login <provider>` for the user to run in their own terminal. The agent never sees or handles the key.
- The key is only sent to that provider's API. It is never written to disk in plain text, printed, or sent into the machine.
- Browser sign-in (OAuth) is not supported. [provider-oauth.md](provider-oauth.md) records what each provider offers.

**In CI (GitHub Actions).**

```bash
oneapi-agent ci setup --task examples/softmax_hecbench/task.json --provider runpod --max-usd 2 --max-minutes 45
```

- Writes `.github/workflows/gpu-test.yml`. It runs `remote-test` on pull requests labelled `gpu-test` and on manual dispatch, with the `--max-usd` and `--max-minutes` caps. It uploads the imported results (`gpu-run/`) as the `gpu-test-results` artifact.
- Pull requests from forks never run it. Fork PRs get no secrets anyway, and the job checks that the PR comes from the same repository.
- Only the `remote-test` step sees the key secrets. Dispatch inputs reach the shell through environment variables, not pasted into the script.
- Then, for each key found on this machine (environment or `login`), it asks before storing the key as a repository secret with `gh secret set <ENV_NAME>`. The value goes to `gh` on stdin, never on a command line or in output. `--yes` stores all of them without asking, and `--no-secrets` skips this step. Nothing is stored when `gh` is missing or not logged in, or when there is no terminal to ask on.
- The workflow installs this toolkit from GitHub `main` by default. Pass `--install` to pin a tag or commit.
- The machine's agent gets a fresh random token for each run.
- Runpod is reached through its HTTPS proxy.
- Vast.ai, Vultr and DigitalOcean machines are reached on a bare IP. There the agent serves TLS with a throwaway self-signed certificate made for the run (this needs `openssl`), and the client trusts only that certificate.

| Provider | GPUs used | Status (2026-10-15) |
|---|---|---|
| Runpod | NVIDIA (RTX A4500 / 2000 Ada / A4000 / …), AMD MI300X | **Verified**: softmax on an RTX 2000 Ada, PASS against the C++ reference and the original CUDA, about 1 min, about $0.005 ([results](gpu-run/nvidia-rtx-2000-ada-generation/)). AMD stock: none. |
| Vast.ai | NVIDIA (RTX A4000 / 3090 / A5000 / 4090) | **Verified**: softmax on an RTX A4000 ($0.12/h), PASS against the C++ reference (9.69e-8) and the original CUDA (7.45e-9), machine deletion confirmed, run from a laptop. Hosts that cannot pull the image are replaced automatically. No AMD offers listed. |
| Vultr | NVIDIA A16 / A40 Cloud GPU | **Verified**: softmax on an A16, PASS against the C++ reference (9.69e-8) and the original CUDA (7.45e-9), machine deletion confirmed, run from a laptop. AMD Instinct plans are listed but not deployable on demand. The API key's access control must allow the caller's IP. |
| DigitalOcean | **AMD Instinct** single GPUs: MI300X/MI325X/MI350X/MI355X (from $2.59/h; spot MI350X $3.50/h), ROCm image `gpu-amd-base` | **Verified**: softmax on an MI350X (gfx950, spot $3.50/h), PASS against the C++ reference (9.69e-8) and the original CUDA, hipified and built with `hipcc` (4.66e-7), with this repo's HIP build of intel/llvm (see "The AMD SYCL toolchain" below). About 2 min, about $0.12, deletion confirmed ([results](gpu-run/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/softmax_hecbench/SUMMARY.md)). The image's ROCm (7.14, `/opt/rocm/core-7.14`) ships no HIP headers; setup fetches the matching `amdrocm-runtime-dev7.14` package for `hipcc`. GPU images require an SSH key; the tool attaches one. Key: `OCEAN_API_KEY` or `DIGITALOCEAN_TOKEN`. |
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

## The AMD SYCL toolchain

intel/llvm publishes no build with the HIP backend: its nightly workflow configures with `--cuda` only. This repo therefore builds one itself:

- [`scripts/build-sycl-hip.sh`](../scripts/build-sycl-hip.sh) builds a pinned intel/llvm commit (the `nightly-2026-10-14` tag) with `buildbot/configure.py --hip` against ROCm 7.0, on Ubuntu 22.04 x86_64. It then compiles and links a SYCL program for gfx942 (MI300X) and gfx950 (MI350X/MI355X); no GPU is needed for that.
- [`.github/workflows/sycl-hip-toolchain.yml`](../.github/workflows/sycl-hip-toolchain.yml) runs it on a standard GitHub runner and publishes `sycl-hip-linux-x86_64.tar.gz`, with its sha256, `BUILD-INFO.txt` and the adapter's `ldd`, as the release `sycl-hip-nightly-2026-10-14-rocm7.0`.
- `remote-test --vendor amd` downloads that tarball to `/opt/sycl` on the machine and checks its sha256 (`DEFAULT_SYCL` in `remote.py`). `--sycl-url`/`--sycl-sha256` choose another tarball; `--image` alone trusts the image's own compiler.

To rebuild it locally: `docker run --rm -v "$PWD:/src:ro" -v /big/disk:/work ubuntu:22.04 bash /src/scripts/build-sycl-hip.sh /work`.

Requirements and limits:
- The GPU host needs **ROCm 7.x**, because the HIP adapter links `libamdhip64.so.7`. DigitalOcean's `gpu-amd-base` has ROCm 7.14 (HIP 7.14.60), and the adapter loaded there; the Runpod AMD default image is now `rocm/dev-ubuntu-22.04:7.0.2-complete`.
- AMD code is compiled with `-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=gfxXXX`. intel/llvm's `amd_gpu_*` shorthand stops at gfx942, but the AMDGPU backend accepts gfx950.
- Two intel/llvm naming bugs are worked around in the script: `configure.py --hip` names the AMD libclc runtime `amdgcn-amd-amdhsa-llvm` while CMake requires `amdgcn-amd-amdhsa`, and the driver looks for libspirv under `amdgpu-amd-amdhsa/` while it is installed under `amdgcn-amd-amdhsa/` (the tarball ships a link).
- The first published tarball (sha256 `e9bf82ee…c8fc3`, 280 MB) was built on an 8-core DigitalOcean droplet (about 1 h, about $0.25) with the same script; the 4-core CI runner takes about 2 h.
- `sycl::ext::oneapi::experimental::architecture` has no gfx950 value, so code that branches on the device architecture will not recognise an MI350X.

Alternatives considered (2026-10-15):
- AdaptiveCpp publishes no binaries, so it would also be a source build, plus a second compile and device-discovery path in `gpu_test`.
- Codeplay's oneAPI for AMD GPUs plugin: its download page loads, but every API behind it returns 404.

## Same port, four GPUs, two vendors, four rental services

Every run is fully automated with `remote-test`, the machine deletion is confirmed, and every result is graded the same way:

| GPU | Service | SYCL vs C++ reference | SYCL vs original CUDA on the same GPU |
|---|---|---|---|
| RTX 2000 Ada | Runpod | 9.69e-8 | 7.45e-9 |
| A16 | Vultr | 9.69e-8 | 7.45e-9 |
| RTX A4000 | Vast.ai | 9.69e-8 | 7.45e-9 |
| AMD Instinct MI350X | DigitalOcean | 9.69e-8 ([results](gpu-run/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/softmax_hecbench/SUMMARY.md)) | 4.69e-7 |

The values are max abs error on HeCBench softmax, atol 1e-6. On the MI350X the original CUDA, run as HIP, is itself 5.25e-7 from the C++ reference, so the SYCL-vs-original gap there comes from the HIP build's math, not from the port. The NVIDIA rows were measured with the previous softmax port (a 32-wide work-group per slice and an out-of-order queue); the current port (below) has so far been run on the MI350X only. An earlier manual run on Runpod (an RTX 2000 Ada, [docs/gpu-run/rtx2000ada](gpu-run/rtx2000ada/SUMMARY.md)) gave the same softmax numbers.

## Six HeCBench programs on an AMD MI350X

Each program covers a different GPU pattern. The CUDA kernels are HeCBench's, unchanged (commit `f0c817aa`); the SYCL ports are in [`examples/`](../examples/README.md). Each row is one `remote-test` run on DigitalOcean (2026-10-15/16), with droplet deletion confirmed:

| Program | Pattern | SYCL vs C++ reference | SYCL vs original CUDA (as HIP) | Tolerance |
|---|---|---|---|---|
| [softmax](gpu-run/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/softmax_hecbench/SUMMARY.md) | sub-group reductions | 9.69e-8 | 4.69e-7 | atol 1e-6 |
| [stencil1d](gpu-run/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/stencil1d_hecbench/SUMMARY.md) | shared-memory tile with halo, barrier | 0 | 0 | exact |
| [matrix_rotate](gpu-run/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/matrix_rotate_hecbench/SUMMARY.md) | in-place 2D index math | 0 | 0 | exact |
| [atomic_reduction](gpu-run/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/atomic_reduction_hecbench/SUMMARY.md) | grid-stride loads + device atomics, 4 block sizes | 0 | 0 | exact |
| [scan](gpu-run/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/scan_hecbench/SUMMARY.md) | Blelloch scan in shared memory, many barriers | 0 | 0 | exact |
| [adam](gpu-run/amd-instinct-mi350x-vf-gfx950-sramecc-xnack-hip-71460-85/adam_hecbench/SUMMARY.md) | Adam optimizer, `pow`/`sqrt` | 2.38e-7 | 5.96e-8 | atol 1e-5 + rtol 1e-4 |

All five new ports compiled and passed on their first run. They have not been run on NVIDIA yet.

**Speed.** Speed is reported, not graded. Most programs here run one short kernel, so a run's wall time is dominated by process and runtime start-up, and `kernel_ms` with one launch includes loading the kernel. To measure the launch cost, softmax was also run with 1000 launches per run (cost per launch, MI350X):

| softmax variant | per launch |
|---|---|
| previous SYCL port: 32-wide work-group per slice, out-of-order queue | 114.6 us |
| one sub-group per slice | 103.4 us |
| one sub-group per slice, in-order queue (the current port) | 24.4 us |
| original CUDA, hipified | 8.9 us |

So the large gap was the out-of-order queue's dependency tracking on every submission; CUDA's default stream is in-order, which makes an in-order queue the faithful translation. Of the new ports, atomic_reduction, scan and adam use in-order queues; stencil1d and matrix_rotate (one launch per run) were verified with the default out-of-order queue. The remaining gap (about 2.7x per launch here) is the SYCL runtime's own submission cost through the HIP adapter, not the kernel.

## Problems found on real clouds, and what the tool does about them

| Symptom | Cause | Handling |
|---|---|---|
| `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` at launch (Vultr A16) | the SYCL compiler emits newer PTX than the host driver can JIT | compile for the GPU's exact compute capability (`nvidia_gpu_sm_XY`, read from `nvidia-smi`) |
| no results from the run, `HOME: unbound variable` (Vultr) | cloud-init starts the agent without `HOME` | `HOME` defaults to `/root` |
| no results from the run, numpy missing (Ubuntu 24.04) | no pip, and PEP 668 blocks pip installs | `apt-get install python3-numpy`, then pip `--break-system-packages` |
| HTTP 403, `error code: 1010` (Runpod proxy) | Cloudflare blocks Python's default user agent | every request sends its own User-Agent |
| `will exceed your GPU limit` (DigitalOcean) | new accounts start with 0 GPUs | reported as-is; ask DigitalOcean support for a limit |
| `does not use root passwords, please use an SSH key` (DigitalOcean) | GPU images need an SSH key | the account's keys, or a throwaway key made and removed for the run |
| `sycl-ls`: "No platforms found" on an AMD MI350X (DigitalOcean) | intel/llvm prebuilts (v7.1.0 and the nightlies to 2026-10-14) are configured with `--cuda` only: no `libur_adapter_hip`, no amdgcn `libspirv`; Codeplay's AMD plugin download API is gone | `--vendor amd` installs intel/llvm built with `--hip` by this repo's CI (below); without any HIP toolchain it stops with `toolchain_missing` before renting; the setup log is kept in the result |
| `deleted ... NOT CONFIRMED` although the droplet was gone (DigitalOcean) | droplet deletion is asynchronous; one immediate check still saw it | deletion is polled for up to 2 minutes before it is reported unconfirmed |
| native build on DigitalOcean AMD: `'hip/hip_runtime.h' file not found` | `gpu-amd-base` ships the ROCm 7.14 runtime and `hipcc`, but not the HIP development headers | reported as a limitation; the SYCL port is still verified against the C++ reference |
| hipified CUDA: `'cooperative_groups/reduce.h' file not found` (ROCm 7.14) | `hipify-perl` leaves that include unchanged; HIP keeps `cg::reduce` (ROCm 7+) at `hip/cooperative_groups/hip_reduce.h` | the include is pointed at the HIP header the host has; with none, the native build is skipped with that reason |
| the first GPU sells out between the stock check and the create | live markets | the next candidate is tried |
| a Vast instance stuck in `loading`: `TLS handshake timeout` from Docker Hub | that host's network | the machine is deleted, its host is skipped, and another is tried; any machine silent past `--boot-minutes` is replaced the same way |

A run that crashes still returns an archive with the error and the run log.
