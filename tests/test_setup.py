"""`oneapi-agent setup` against a throwaway home directory."""

from __future__ import annotations

import json
import sys
import tomllib

import pytest

import oneapi_agent.setup as st

ENV = {"PATH": "/opt/dpcpp/bin:/usr/bin", "LD_LIBRARY_PATH": "/opt/dpcpp/lib",
       "ONEAPI_AGENT_CUDA_INCLUDE": "/opt/cuda/include", "ONEAPI_AGENT_API_KEY": "sk-secret",
       "UNRELATED": "x"}


@pytest.fixture
def no_claude_cli(monkeypatch):
    real_which = st.shutil.which
    monkeypatch.setattr(st.shutil, "which", lambda n: None if n == "claude" else real_which(n))


def test_server_env_copies_toolchain_vars_but_never_the_api_key():
    env = st.server_env(ENV)
    assert env == {"PATH": ENV["PATH"], "LD_LIBRARY_PATH": ENV["LD_LIBRARY_PATH"],
                   "ONEAPI_AGENT_CUDA_INCLUDE": "/opt/cuda/include"}
    assert st.server_command() == [sys.executable, "-m", "oneapi_agent.server"]


def test_cursor_and_gemini_merge_without_clobbering(tmp_path):
    cursor = tmp_path / ".cursor" / "mcp.json"
    cursor.parent.mkdir()
    cursor.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "keep": 1}))
    r = st.setup(["cursor", "gemini"], home=tmp_path, environ=ENV)
    assert r.status == "ok", r.summary
    data = json.loads(cursor.read_text())
    assert data["keep"] == 1 and "other" in data["mcpServers"]
    entry = data["mcpServers"]["oneapi-agent"]
    assert entry["args"] == ["-m", "oneapi_agent.server"]
    assert "ONEAPI_AGENT_API_KEY" not in entry["env"]
    assert (tmp_path / ".cursor" / "mcp.json.bak").exists()
    gem = json.loads((tmp_path / ".gemini" / "settings.json").read_text())
    assert gem["mcpServers"]["oneapi-agent"] == entry


def test_rerun_is_idempotent(tmp_path):
    st.setup(["cursor"], home=tmp_path, environ=ENV)
    first = (tmp_path / ".cursor" / "mcp.json").read_text()
    r = st.setup(["cursor"], home=tmp_path, environ=ENV)
    assert (tmp_path / ".cursor" / "mcp.json").read_text() == first
    assert r.evidence["actions"][0]["changed"] is False


def test_invalid_json_is_left_untouched(tmp_path):
    p = tmp_path / ".cursor" / "mcp.json"
    p.parent.mkdir()
    p.write_text("{ not json")
    r = st.setup(["cursor"], home=tmp_path, environ=ENV)
    assert r.status == "fail" and p.read_text() == "{ not json"


def test_codex_toml_replaces_only_our_tables(tmp_path):
    p = tmp_path / ".codex" / "config.toml"
    p.parent.mkdir()
    p.write_text('model = "x"\n\n[mcp_servers.other]\ncommand = "o"\n\n'
                 '[mcp_servers.oneapi-agent]\ncommand = "stale"\n\n[mcp_servers.oneapi-agent.env]\nA = "1"\n')
    assert st.setup(["codex"], home=tmp_path, environ=ENV).status == "ok"
    data = tomllib.loads(p.read_text())
    assert data["model"] == "x" and data["mcp_servers"]["other"]["command"] == "o"
    ours = data["mcp_servers"]["oneapi-agent"]
    assert ours["command"] == sys.executable and ours["args"] == ["-m", "oneapi_agent.server"]
    assert ours["env"]["LD_LIBRARY_PATH"] == "/opt/dpcpp/lib" and "A" not in ours["env"]
    st.setup(["codex"], home=tmp_path, environ=ENV)  # second run: still one table
    assert p.read_text().count("[mcp_servers.oneapi-agent]") == 1


def test_claude_without_cli_is_skipped_but_skill_installs(tmp_path, no_claude_cli):
    r = st.setup(["claude"], home=tmp_path, environ=ENV)
    acts = {(a["agent"], "skill" in a): a for a in r.evidence["actions"]}
    assert acts[("claude", False)]["status"] == "skipped"
    argv = acts[("claude", False)]["command"]
    # name before the variadic -e flags (verified against the real CLI, which rejects the reverse)
    assert argv[:6] == ["claude", "mcp", "add", "-s", "user", "oneapi-agent"]
    assert argv.index("--") > max(i for i, a in enumerate(argv) if a == "-e")
    assert (tmp_path / ".claude" / "skills" / "oneapi-port" / "SKILL.md").is_file()
    assert r.status == "ok"


def test_dry_run_writes_nothing(tmp_path, no_claude_cli):
    r = st.setup(["cursor", "codex", "claude"], home=tmp_path, environ=ENV, dry_run=True)
    assert r.status == "ok" and r.summary.startswith("would register")
    assert list(tmp_path.iterdir()) == []


def test_no_agents_detected_and_unknown_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(st.shutil, "which", lambda n: None)
    assert st.setup(home=tmp_path, environ=ENV).status == "error"
    assert st.setup(["vim"], home=tmp_path, environ=ENV).status == "error"


def test_claude_cli_runs_with_the_given_home(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(st.shutil, "which", lambda n: "/usr/bin/claude" if n == "claude" else None)

    class R:
        returncode, stderr = 0, ""

    monkeypatch.setattr(st, "run_command", lambda argv, **kw: calls.append(kw.get("env")) or R())
    assert st.setup(["claude"], home=tmp_path, environ=ENV, skill=False).status == "ok"
    assert calls and all(e == {"HOME": str(tmp_path)} for e in calls)
