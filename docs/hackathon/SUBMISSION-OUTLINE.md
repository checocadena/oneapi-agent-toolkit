# Submission outline (placeholder until the real fields are published)

Everything in brackets is filled in only from measured, committed results. Nothing here is
a result yet.

## Project name
oneapi-agent-toolkit: Muse ports GPU code off CUDA, and proves it

## One-line summary
An open-source toolkit where Meta's Muse models port NVIDIA CUDA code to the open SYCL
standard, and every port is proven correct by checks the model never sees, then run on real
NVIDIA and AMD GPUs.

## Problem (about 80 words)
Most AI code runs on NVIDIA's CUDA, so moving it to other hardware means rewriting GPU code
by hand. Coding agents can help, but they report success they never checked: in our earlier
measurements an agent's "finished" port compiled cleanly and got 332,679 of 524,288 outputs
wrong. Porting needs an agent that can't fake it.

## What it does (about 120 words)
- Five tools that return evidence instead of verdicts: migrate, compile, scan_helpers,
  verify, benchmark.
- An agent loop that only accepts "done" when verification passes and no migration helpers
  remain, on the same freshly built source.
- [Meta feature from FEATURE-OPTIONS.md, as built.]
- GPU checks on rented NVIDIA and AMD machines that are always deleted afterwards.

## How it uses Meta's models
- Muse [Spark model id] is the agent in the loop, through the Meta Model API
  (`--provider meta`).
- [Option A: measured comparison, Muse with vs without the checks: X/Y vs X/Y.]
- [Option B/C: Muse plans the port of a whole repository / ports N kernels of llm.c.]

## What existed before the hackathon (disclosure)
The toolkit, its Claude-based measurements and the GPU pipeline existed before Oct 26
(v0.3.0, public since Oct 2026). Built during the hackathon: [list, with commit range].
*(Adjust to what the Official Rules require.)*

## Tech stack
Python 3.11+; Meta Model API (Muse); SYCLomatic, Intel oneAPI DPC++ (icpx), oneMath; MCP;
rented GPUs on Runpod, Vast.ai, Vultr, DigitalOcean; GitHub Actions CI with the real
toolchain.

## Links
- Repository: https://github.com/checocadena/oneapi-agent-toolkit
- Demo video: [link]
- Replays: https://checocadena.github.io/oneapi-agent-toolkit/docs/replay/index.html

## Demo video script (about 3 minutes; adjust to the required length)

| Time | Show | Say |
|---|---|---|
| 0:00 to 0:20 | A CUDA kernel next to an AMD and an Intel logo | "Most AI code only runs on NVIDIA. Moving it means rewriting GPU code, and AI agents say they did it when they didn't." |
| 0:20 to 0:45 | The replay of the failing run: every step green, then the hidden check: 332,679 wrong | "Here an agent declared success. The hidden check disagreed." |
| 0:45 to 1:45 | Terminal: `oneapi-agent loop --provider meta --config ...`; the trace as Muse migrates, fails to compile, fixes, verifies, reaches zero helpers | "Now Muse does the port, and the toolkit makes it prove each step." |
| 1:45 to 2:20 | [Meta feature: comparison chart, or the repository plan, or the llm.c kernels] | [measured numbers only] |
| 2:20 to 2:45 | `remote-test` on a rented NVIDIA and AMD GPU: PASS, machine deleted | "Then it runs on real NVIDIA and AMD hardware, against the original CUDA." |
| 2:45 to 3:00 | The repository page | "Open source, with every trace behind every number." |

Recording notes: record terminal parts with real runs (no mock output in the video unless it
is labelled MOCK); keep a backup recording made the day before.
