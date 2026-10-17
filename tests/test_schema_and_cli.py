"""The published ToolResult schema matches the code and real CLI output; first-run CLI errors."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from oneapi_agent.result import STATUSES, tool_result_json_schema

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "docs" / "tool-result-schema.json"


def cli(*args: str, env: dict[str, str] | None = None, cwd: Path = REPO):
    full = {**os.environ, "PYTHONPATH": str(REPO / "src"), **(env or {})}
    return subprocess.run([sys.executable, "-m", "oneapi_agent.cli", *args], cwd=cwd, env=full,
                          capture_output=True, text=True, timeout=120)


def conforms(obj: dict, schema: dict) -> list[str]:
    """Minimal check of the schema's own rules (no jsonschema dependency)."""
    problems = [f"missing {k}" for k in schema["required"] if k not in obj]
    problems += [f"extra {k}" for k in obj if k not in schema["properties"]]
    py = {"string": str, "object": dict, "array": list}
    for k, spec in schema["properties"].items():
        if k in obj and not isinstance(obj[k], py[spec["type"]]):
            problems.append(f"{k} is not {spec['type']}")
    if obj.get("status") not in schema["properties"]["status"]["enum"]:
        problems.append(f"status {obj.get('status')!r}")
    return problems


def test_published_schema_is_generated_from_the_code():
    assert json.loads(SCHEMA.read_text()) == tool_result_json_schema(), (
        "regenerate docs/tool-result-schema.json from oneapi_agent.result.tool_result_json_schema")
    assert tool_result_json_schema()["properties"]["status"]["enum"] == list(STATUSES)


def test_real_cli_output_conforms_to_the_schema(tmp_path):
    schema = json.loads(SCHEMA.read_text())
    (tmp_path / "k.cpp").write_text("#include <dpct/dpct.hpp>\nint main(){ dpct::get_default_queue(); }\n")
    for args in (("doctor",), ("scan-helpers", str(tmp_path))):
        p = cli(*args)
        result = json.loads(p.stdout)
        assert conforms(result, schema) == [], (args, result)
        assert p.returncode == {"ok": 0, "fail": 1}.get(result["status"], 2)


def test_real_model_run_without_a_key_fails_before_any_work(tmp_path):
    env = {"ONEAPI_AGENT_API_KEY": "", "ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""}
    p = cli("loop", "--config", str(REPO / "examples/saxpy/task.json"),
            "--runs-dir", str(tmp_path / "runs"), env=env)
    assert p.returncode != 0
    assert "no API key" in p.stderr and "Traceback" not in p.stderr
    assert not (tmp_path / "runs").exists()  # nothing was started


def test_setup_with_no_agent_says_what_to_do(tmp_path):
    env = {"HOME": str(tmp_path), "PATH": os.path.dirname(sys.executable)}
    p = cli("--summary", "setup", "--dry-run", env=env)
    assert p.returncode == 2 and "pass --agent NAME" in p.stdout
    p = cli("--summary", "setup", "--dry-run", "--agent", "claude", env=env)
    assert p.returncode == 0 and "would register" in p.stdout
