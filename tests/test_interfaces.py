"""CLI and MCP server surfaces."""

from __future__ import annotations

import asyncio
import json
import subprocess

import pytest
from conftest import PY, REPO

from oneapi_agent.cli import main as cli_main


def test_cli_scan_helpers_exit_code_and_json(fixtures, capsys):
    rc = cli_main(["scan-helpers", str(fixtures / "reduce.dp.cpp")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["status"] == "fail" and out["evidence"]["total"] == 16


def test_cli_summary_mode(fixtures, capsys):
    rc = cli_main(["--summary", "scan-helpers", str(fixtures / "reduce.dp.cpp")])
    assert rc == 1
    assert capsys.readouterr().out.startswith("[scan_helpers] fail:")


def test_cli_doctor_runs(capsys):
    rc = cli_main(["doctor"])
    out = json.loads(capsys.readouterr().out)
    assert out["tool"] == "doctor" and rc in (0, 2)
    assert set(out["evidence"]) >= {"icpx", "c2s", "sycl-ls"}


def test_cli_verify(capsys):
    rc = cli_main(["verify", "--ref", f"{PY} -c 'print(1, 2)'", "--cand", f"{PY} -c 'print(1, 2)'"])
    assert rc == 0 and json.loads(capsys.readouterr().out)["status"] == "ok"


def test_cli_loop_dry_run_mock_tools(tmp_path):
    proc = subprocess.run(
        [PY, "-m", "oneapi_agent.cli", "loop", "--dry-run", "--mock-tools",
         "--config", str(REPO / "examples" / "saxpy" / "task.json"),
         "--runs-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["stop_reason"] == "success"
    assert (tmp_path / out["run_dir"].split("/")[-1] / "report.md").is_file()


def test_mcp_server_lists_and_calls_tools(fixtures):
    pytest.importorskip("mcp")
    from oneapi_agent.server import mcp

    async def go():
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        res = await mcp.call_tool("scan_helpers", {"path": str(fixtures / "reduce.dp.cpp")})
        return names, res

    names, res = asyncio.run(go())
    assert {"migrate", "compile", "scan_helpers", "verify", "benchmark", "doctor"} <= names
    # The result shape differs between mcp 1.x (tuple/list) and 2.x (CallToolResult);
    # find the JSON text in either.
    text = str(res)
    assert '"status": "fail"' in text or "'status': 'fail'" in text


def test_mcp_tool_descriptions_mention_evidence():
    pytest.importorskip("mcp")
    from oneapi_agent.server import mcp

    tools = asyncio.run(mcp.list_tools())
    for t in tools:
        if t.name != "doctor":
            assert "status" in (t.description or ""), t.name
