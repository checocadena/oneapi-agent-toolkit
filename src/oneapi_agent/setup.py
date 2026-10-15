"""``oneapi-agent setup``: register the MCP server (and the skill) with coding agents.

One command instead of hand-editing configs. Supported agents and what is written:

  claude  ``claude mcp add -s user oneapi-agent ...`` (its own CLI) + the oneapi-port skill
          copied to ``~/.claude/skills/oneapi-port``
  codex   ``[mcp_servers.oneapi-agent]`` in ``~/.codex/config.toml``
  cursor  ``mcpServers.oneapi-agent`` in ``~/.cursor/mcp.json``
  gemini  ``mcpServers.oneapi-agent`` in ``~/.gemini/settings.json``

The server is launched as ``<this python> -m oneapi_agent.server``, so it does not depend on
PATH. Toolchain variables present when setup runs (PATH, LD_LIBRARY_PATH, ONEAPI_AGENT_*, ...)
are copied into the server's environment: run setup after sourcing your oneAPI environment.
Every edited file is backed up first (``<file>.bak``), and re-running setup replaces the
existing entry instead of duplicating it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .result import ToolResult
from .toolchain import run_command

SERVER_NAME = "oneapi-agent"
AGENTS = ("claude", "codex", "cursor", "gemini")
_ENV_KEYS = ("PATH", "LD_LIBRARY_PATH", "ONEAPI_ROOT", "CUDA_PATH")
_ENV_PREFIX = "ONEAPI_AGENT_"
_SECRET_KEYS = ("ONEAPI_AGENT_API_KEY",)  # never copied into agent configs


def server_command() -> list[str]:
    return [sys.executable, "-m", "oneapi_agent.server"]


def server_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    out = {k: env[k] for k in _ENV_KEYS if env.get(k)}
    out.update({k: v for k, v in env.items()
                if k.startswith(_ENV_PREFIX) and k not in _SECRET_KEYS and v})
    return out


def skill_source() -> Path | None:
    # src/oneapi_agent/skill is a symlink to skills/oneapi-port in a git checkout; wheels
    # carry a real copy (package data), so `pip install oneapi-agent-toolkit` gets it too.
    p = Path(__file__).resolve().parent / "skill"
    return p if (p / "SKILL.md").is_file() else None


def detect_agents(home: Path) -> list[str]:
    found = []
    if shutil.which("claude") or (home / ".claude").exists():
        found.append("claude")
    if shutil.which("codex") or (home / ".codex").exists():
        found.append("codex")
    if (home / ".cursor").exists():
        found.append("cursor")
    if shutil.which("gemini") or (home / ".gemini").exists():
        found.append("gemini")
    return found


def _backup(path: Path, dry_run: bool) -> str | None:
    if not path.exists():
        return None
    bak = path.with_name(path.name + ".bak")
    if not dry_run:
        shutil.copy2(path, bak)
    return str(bak)


def _merge_json(path: Path, entry: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if path.exists():
        text = path.read_text().strip()
        if text:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                return {"path": str(path), "status": "fail",
                        "reason": f"existing file is not valid JSON ({exc}); left untouched"}
    servers = data.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        return {"path": str(path), "status": "fail", "reason": "mcpServers is not an object"}
    changed = servers.get(SERVER_NAME) != entry
    action: dict[str, Any] = {"path": str(path), "status": "ok", "changed": changed,
                              "replaced_existing": SERVER_NAME in servers}
    if changed:
        action["backup"] = _backup(path, dry_run)
        servers[SERVER_NAME] = entry
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + "\n")
    return action


def _toml_str(s: str) -> str:
    return json.dumps(s)  # a JSON string literal is a valid TOML basic string


def _codex_block(cmd: list[str], env: dict[str, str]) -> str:
    lines = [f"[mcp_servers.{SERVER_NAME}]", f"command = {_toml_str(cmd[0])}",
             "args = [" + ", ".join(_toml_str(a) for a in cmd[1:]) + "]"]
    if env:
        lines += ["", f"[mcp_servers.{SERVER_NAME}.env]"]
        lines += [f"{k} = {_toml_str(v)}" for k, v in sorted(env.items())]
    return "\n".join(lines) + "\n"


def _strip_codex_section(text: str) -> str:
    """Drop our own [mcp_servers.oneapi-agent] / [...env] tables, keep everything else."""
    out, skipping = [], False
    ours = (f"[mcp_servers.{SERVER_NAME}]", f"[mcp_servers.{SERVER_NAME}.env]")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            skipping = stripped in ours
        if not skipping:
            out.append(line)
    return "\n".join(out).rstrip() + ("\n" if out else "")


def setup_codex(home: Path, cmd: list[str], env: dict[str, str], dry_run: bool) -> dict[str, Any]:
    path = home / ".codex" / "config.toml"
    old = path.read_text() if path.exists() else ""
    base = _strip_codex_section(old)
    new = (base + ("\n" if base else "") + _codex_block(cmd, env))
    try:
        import tomllib

        tomllib.loads(new)
    except Exception as exc:  # never write a file the agent cannot parse
        return {"agent": "codex", "path": str(path), "status": "fail",
                "reason": f"resulting TOML would not parse ({exc}); left untouched"}
    action = {"agent": "codex", "path": str(path), "status": "ok", "changed": new != old,
              "replaced_existing": f"[mcp_servers.{SERVER_NAME}]" in old}
    if new != old:
        action["backup"] = _backup(path, dry_run)
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new)
    return action


def setup_claude(cmd: list[str], env: dict[str, str], dry_run: bool,
                 home: Path | None = None) -> dict[str, Any]:
    # The name goes before -e: -e is variadic and would swallow a name placed after it.
    argv = ["claude", "mcp", "add", "-s", "user", SERVER_NAME]
    for k, v in sorted(env.items()):
        argv += ["-e", f"{k}={v}"]
    argv += ["--", *cmd]
    action: dict[str, Any] = {"agent": "claude", "command": argv}
    if not shutil.which("claude"):
        action.update(status="skipped",
                      reason="`claude` CLI not on PATH; run the command above once it is installed")
        return action
    if dry_run:
        action.update(status="ok", changed=True, dry_run=True)
        return action
    # The claude CLI keeps user-scope servers under $HOME: point it at the same home as the
    # other agents' files, so `setup(home=...)` never touches the real user's config.
    cli_env = {"HOME": str(home)} if home is not None else None
    run_command(["claude", "mcp", "remove", "-s", "user", SERVER_NAME], timeout=60,
                env=cli_env)  # idempotent
    res = run_command(argv, timeout=60, env=cli_env)
    action.update(status="ok" if res.returncode == 0 else "fail", changed=res.returncode == 0,
                  returncode=res.returncode, stderr=res.stderr[-1000:])
    return action


def install_claude_skill(home: Path, dry_run: bool) -> dict[str, Any]:
    src = skill_source()
    dest = home / ".claude" / "skills" / "oneapi-port"
    if src is None:
        return {"agent": "claude", "skill": "oneapi-port", "status": "skipped",
                "reason": "skill files not found in the installed package"}
    action = {"agent": "claude", "skill": "oneapi-port", "path": str(dest), "status": "ok",
              "changed": True, "replaced_existing": dest.exists()}
    if not dry_run:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    return action


def setup(agents: Sequence[str] | None = None, *, dry_run: bool = False, skill: bool = True,
          home: Path | None = None, environ: dict[str, str] | None = None) -> ToolResult:
    home = home or Path.home()
    chosen = list(agents) if agents else detect_agents(home)
    bad = [a for a in chosen if a not in AGENTS]
    if bad:
        return ToolResult("setup", "error", f"unknown agent(s): {bad}; choose from {AGENTS}")
    if not chosen:
        return ToolResult("setup", "error",
                          "no supported agent detected; pass --agent claude|codex|cursor|gemini",
                          {"looked_for": list(AGENTS), "home": str(home)})
    cmd, env = server_command(), server_env(environ)
    entry = {"command": cmd[0], "args": cmd[1:], "env": env}
    actions: list[dict[str, Any]] = []
    for agent in chosen:
        if agent == "claude":
            actions.append(setup_claude(cmd, env, dry_run, home))
            if skill:
                actions.append(install_claude_skill(home, dry_run))
        elif agent == "codex":
            actions.append(setup_codex(home, cmd, env, dry_run))
        elif agent == "cursor":
            actions.append({"agent": "cursor", **_merge_json(home / ".cursor" / "mcp.json", entry,
                                                             dry_run)})
        elif agent == "gemini":
            actions.append({"agent": "gemini",
                            **_merge_json(home / ".gemini" / "settings.json", entry, dry_run)})
    failed = [a for a in actions if a.get("status") == "fail"]
    done = [a for a in actions if a.get("status") == "ok"]
    limitations = [
        "Restart each agent (or reload its MCP servers) to pick up the new server.",
        "The server environment is a snapshot of PATH/LD_LIBRARY_PATH/ONEAPI_AGENT_* at setup "
        "time; re-run setup after changing your toolchain.",
        "Codex, Cursor and Gemini have no native skill format here: point them at "
        "skills/oneapi-port/SKILL.md from your AGENTS.md or rules.",
    ]
    if dry_run:
        limitations.insert(0, "DRY RUN: nothing was written.")
    evidence = {"agents": chosen, "server_command": cmd, "env_keys": sorted(env), "actions": actions,
                "dry_run": dry_run}
    names = ", ".join(sorted({a["agent"] for a in done}))
    if failed:
        return ToolResult("setup", "fail",
                          f"{len(failed)} step(s) failed: "
                          + "; ".join(f"{a['agent']}: {a.get('reason') or a.get('stderr', '')[:120]}"
                                      for a in failed), evidence, limitations)
    if not done:
        return ToolResult("setup", "fail", "nothing was registered (every step was skipped)",
                          evidence, limitations)
    prefix = "would register" if dry_run else "registered"
    return ToolResult("setup", "ok", f"{prefix} {SERVER_NAME} with: {names}", evidence, limitations)
