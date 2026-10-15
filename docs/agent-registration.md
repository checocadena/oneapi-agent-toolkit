# Agent registration: verified against the real clients

`oneapi-agent setup` writes the `oneapi-agent` MCP server into each agent's config. Claude Code
was already verified through its own CLI. This page records the check of Codex, Gemini CLI and
Cursor against the real clients: each one was installed, then listed or connected to the server.
Run on 2026-10-15, Linux x86_64.

| Client | Version | Result |
|---|---|---|
| Codex CLI (`@openai/codex`) | `codex-cli 0.162.1` | config parsed, entry shown `enabled` (Codex's `mcp list` does not connect) |
| Gemini CLI (`@google/gemini-cli`) | `0.64.0` | `Connected` |
| Cursor CLI (`cursor-agent`) | `2026.10.14-941492e` | `ready`, all 6 tools listed |

No format changes were needed: all three clients accepted what setup writes.

## How it was run

The CLIs were installed into a scratch prefix, not globally:

```sh
npm install --prefix "$S/clis" @openai/codex @google/gemini-cli
# Cursor: the package https://cursor.com/install downloads, unpacked in place of running the script
curl -fsSL -o cursor-agent.tgz \
  https://downloads.cursor.com/lab/2026.10.14-941492e/linux/x64/agent-cli-package.tar.gz
tar --strip-components=1 -xzf cursor-agent.tgz -C "$S/cursor-pkg"
```

Every command below ran with `HOME` set to a new, empty directory, those CLIs on `PATH`, and no
API keys in the environment. No client was logged in.

```sh
PYTHONPATH=src .venv/bin/python -m oneapi_agent.cli setup --agent codex --agent gemini --agent cursor
```

```
  "summary": "registered oneapi-agent with: codex, cursor, gemini",
```

### Codex

```sh
codex mcp list
codex mcp get oneapi-agent
```

```
Name          Command                                           Args                    Env         Cwd  Status   Auth
oneapi-agent  <repo>/.venv/bin/python  -m oneapi_agent.server  PATH=*****  -    enabled  Unsupported
```

```
oneapi-agent
  enabled: true
  transport: stdio
  command: <repo>/.venv/bin/python
  args: -m oneapi_agent.server
```

`Auth: Unsupported` is what Codex shows for any stdio server. Codex has no command that
connects to a server without a logged-in session, so for Codex the check covers config parsing
only. Gemini and Cursor started the same command and connected to it.

### Gemini CLI

```sh
gemini mcp list
```

When run from a folder that is not trusted, Gemini reads the entry but does not start it:

```
Warning: MCP servers are configured but disabled because this folder is untrusted.
○ oneapi-agent: <repo>/.venv/bin/python -m oneapi_agent.server (stdio) - Disabled
```

After the working folder was trusted (`~/.gemini/trustedFolders.json` containing
`{"<folder>": "TRUST_FOLDER"}`, or by answering Gemini's trust prompt), the server connected:

```
✓ oneapi-agent: <repo>/.venv/bin/python -m oneapi_agent.server (stdio) - Connected
```

Folder trust is the user's own security decision, so `setup` does not change it.

### Cursor

```sh
cursor-agent mcp list
cursor-agent mcp list-tools oneapi-agent
```

```
oneapi-agent: ready
```

```
Tools for oneapi-agent (6):
- benchmark (cmd, runs, devices, warmup, cwd, report_regex)
- compile (sources, flags, output, cwd, add_dpct_include)
- doctor ()
- migrate (cuda_path, out_dir, in_root, extra_args)
- scan_helpers (path)
- verify (reference_cmd, candidate_cmd, inputs, tolerance, rtol, mode, threshold_dbfs, max_ulp, cwd)
```

## Idempotency under the real clients

- Setup was run twice more with the same environment. Every action reported `"changed": false`,
  and `sha256sum -c` confirmed that `config.toml`, `settings.json` and `mcp.json` were
  byte-identical.
- The clients then added a server of their own: `codex mcp add other-srv -- /bin/true` and
  `gemini mcp add -s user other-srv /bin/true`. Setup was run twice more with a changed
  environment (`ONEAPI_AGENT_REGTEST=1`). The first run reported `"changed": true` and the
  second `"changed": false`. `config.toml` still held exactly one `[mcp_servers.oneapi-agent]`,
  and each client still worked:
  - `codex mcp list` showed `oneapi-agent` with `ONEAPI_AGENT_REGTEST=*****, PATH=*****` next
    to `other-srv`.
  - `gemini mcp list` showed `✓ oneapi-agent ... - Connected` next to the client-added
    `other-srv`.
  - `cursor-agent mcp list` showed `oneapi-agent: ready`.
- An entry written by Codex itself (`codex mcp add oneapi-agent --env FOO=bar -- /usr/bin/python3 -m x`)
  was replaced by setup (`"replaced_existing": true`). Afterwards `codex mcp list` showed one
  `oneapi-agent` entry with the setup command and only `PATH=*****`, and `other-srv` was kept.
