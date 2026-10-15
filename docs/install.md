# Installing and registering oneapi-agent

## 1. Install the Python package

```bash
git clone https://github.com/checocadena/oneapi-agent-toolkit
cd oneapi-agent-toolkit
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[all]"        # or: pip install -e ".[mcp]" for just the server
oneapi-agent --summary doctor  # reports which toolchain pieces were found
```

Python 3.11+ is required. The package works without the oneAPI toolchain.
Tools that need a missing binary return `status: "toolchain_missing"` together with the exact command that could not run.

## 2. (Optional but recommended) install the toolchain

| Piece | Used by | Official route | Route verified in this repo's CI-less sandbox |
|-------|---------|----------------|-----------------------------------------------|
| `icpx` (DPC++) + a SYCL device runtime | compile, benchmark, verify (running the candidate) | Intel oneAPI Base Toolkit (apt/yum/offline installer) | conda-forge `dpcpp_linux-64` + `intel-opencl-rt` |
| `c2s` (SYCLomatic) | migrate | Intel oneAPI Base Toolkit (`intel-oneapi-dpcpp-ct`) | GitHub release `oneapi-src/SYCLomatic` `linux_release.tgz` |
| CUDA headers | migrate (c2s parses `.cu`) | CUDA toolkit | conda-forge `cuda-cudart-dev` etc. at **12.3** |
| `sycl-ls` | benchmark device discovery | ships with icpx | ships with icpx |

`scripts/install-toolchain.sh [/opt]` reproduces the verified route.
Then run `source scripts/oneapi-env.sh` to put everything on `PATH` and `LD_LIBRARY_PATH`.
If you use the official installer, run `source /opt/intel/oneapi/setvars.sh` instead.

Discovery and overrides (environment variables):

| Variable | Meaning |
|----------|---------|
| `ONEAPI_AGENT_ICPX`, `ONEAPI_AGENT_C2S`, `ONEAPI_AGENT_SYCL_LS` | explicit binary paths (otherwise `PATH`, then `$ONEAPI_ROOT/.../bin`) |
| `ONEAPI_AGENT_CUDA_INCLUDE` | directory containing `cuda_runtime.h` (else `$CUDA_PATH/include`, `/usr/local/cuda/include`) |
| `ONEAPI_AGENT_DPCT_INCLUDE` | directory containing `dpct/dpct.hpp` (else `<c2s>/../include`) |

The MCP server inherits its environment from the agent that launches it.
Put these variables in the agent's MCP config (examples below) or source the env script before starting the agent.

## 3. Register the MCP server

The quick way: source your toolchain environment first, then

```bash
oneapi-agent setup --dry-run   # prints every file/command it would touch
oneapi-agent setup             # all detected agents; or --agent claude|codex|cursor|gemini (repeatable)
```

It runs `claude mcp add -s user` for Claude Code (and copies the `oneapi-port` skill to `~/.claude/skills/`),
and merges an `oneapi-agent` entry into `~/.codex/config.toml`, `~/.cursor/mcp.json` and `~/.gemini/settings.json`.
Edited files are backed up as `<file>.bak`, re-running replaces the entry instead of duplicating it,
and `ONEAPI_AGENT_API_KEY` is never written into any config. The manual equivalents follow.

The server speaks MCP over stdio: `oneapi-agent serve` (equivalently `python -m oneapi_agent.server`).
Use the absolute path of the venv's `oneapi-agent` if it is not on `PATH`.

### Claude Code

```bash
claude mcp add oneapi-agent \
  -e ONEAPI_AGENT_CUDA_INCLUDE=/opt/cuda-headers/targets/x86_64-linux/include \
  -e LD_LIBRARY_PATH=/opt/dpcpp/lib \
  -e PATH=/opt/dpcpp/bin:/opt/syclomatic/bin:/usr/bin:/bin \
  -- /path/to/oneapi-agent-toolkit/.venv/bin/oneapi-agent serve
claude mcp list
```

Alternatively, commit a project-scoped `.mcp.json`:

```json
{
  "mcpServers": {
    "oneapi-agent": {
      "command": "/path/to/oneapi-agent-toolkit/.venv/bin/oneapi-agent",
      "args": ["serve"],
      "env": {
        "PATH": "/opt/dpcpp/bin:/opt/syclomatic/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/opt/dpcpp/lib",
        "ONEAPI_AGENT_CUDA_INCLUDE": "/opt/cuda-headers/targets/x86_64-linux/include"
      }
    }
  }
}
```

To install the skill, copy or symlink `skills/oneapi-port` to `.claude/skills/oneapi-port` (per project) or `~/.claude/skills/oneapi-port` (per user).
Claude Code loads it whenever a task matches the description in its frontmatter.

### OpenAI Codex CLI

`~/.codex/config.toml`:

```toml
[mcp_servers.oneapi-agent]
command = "/path/to/oneapi-agent-toolkit/.venv/bin/oneapi-agent"
args = ["serve"]

[mcp_servers.oneapi-agent.env]
PATH = "/opt/dpcpp/bin:/opt/syclomatic/bin:/usr/bin:/bin"
LD_LIBRARY_PATH = "/opt/dpcpp/lib"
ONEAPI_AGENT_CUDA_INCLUDE = "/opt/cuda-headers/targets/x86_64-linux/include"
```

Codex has no native skill format here.
Reference the procedure from your repo's `AGENTS.md`, for example: "For CUDA→SYCL work, follow `skills/oneapi-port/SKILL.md` exactly."

### Cursor

Create `.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` globally:

```json
{
  "mcpServers": {
    "oneapi-agent": {
      "command": "/path/to/oneapi-agent-toolkit/.venv/bin/oneapi-agent",
      "args": ["serve"],
      "env": {
        "PATH": "/opt/dpcpp/bin:/opt/syclomatic/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/opt/dpcpp/lib",
        "ONEAPI_AGENT_CUDA_INCLUDE": "/opt/cuda-headers/targets/x86_64-linux/include"
      }
    }
  }
}
```

To load the procedure as a rule, add `.cursor/rules/oneapi-port.mdc` containing the body of `skills/oneapi-port/SKILL.md`, with `alwaysApply: false` and a description that mentions CUDA/SYCL.

## 4. Smoke test

```bash
oneapi-agent --summary doctor
oneapi-agent --summary scan-helpers tests/fixtures/reduce.dp.cpp   # exits 1: helpers remain
oneapi-agent loop --dry-run --mock-tools                           # no toolchain or API key needed
```
