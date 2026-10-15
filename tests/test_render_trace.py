"""scripts/render_trace.py: replay page from a tiny synthetic trace."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = "/home/user/oneapi-agent-toolkit/runs/20260101T000000Z"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_trace", REPO / "scripts" / "render_trace.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_trace"] = mod
    spec.loader.exec_module(mod)
    return mod


rt = _load()


def res(tool: str, status: str, summary: str, **ev: Any) -> dict[str, Any]:
    return {"tool": tool, "status": status, "summary": summary, "evidence": ev, "limitations": []}


def _records() -> list[dict[str, Any]]:
    v1 = "int main() {\n  return 1;\n}\n"
    v2 = "int main() {\n  return 0; // <script>alert(1)</script>\n}\n"
    return [
        {"type": "start", "model": "m", "run_dir": RUN_DIR,
         "config": {"name": "demo", "goal": "Port x.cu </title><script>bad()</script>",
                    "cuda_path": "/home/user/oneapi-agent-toolkit/examples/x/x.cu"}},
        {"type": "model_response", "iteration": 1, "stop_reason": "tool_use", "content": [
            {"type": "text", "text": "I will write <script>alert('x')</script> & go."},
            {"type": "tool_use", "id": "a", "name": "write_file", "input": {}}]},
        {"type": "tool_call", "iteration": 1, "name": "write_file",
         "input": {"path": "migrated/x.cpp", "content": v1}},
        {"type": "tool_result", "iteration": 1, "name": "write_file",
         "input": {"path": "migrated/x.cpp", "content": v1}, "source_hash": "h1",
         "result": res("write_file", "ok", "wrote migrated/x.cpp",
                       path=f"{RUN_DIR}/work/migrated/x.cpp")},
        {"type": "model_response", "iteration": 2, "stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "b", "name": "write_file", "input": {}}]},
        {"type": "tool_call", "iteration": 2, "name": "write_file",
         "input": {"path": "migrated/x.cpp", "content": v2}},
        {"type": "tool_result", "iteration": 2, "name": "write_file",
         "input": {"path": "migrated/x.cpp", "content": v2}, "source_hash": "h2",
         "result": res("write_file", "ok", "wrote migrated/x.cpp")},
        {"type": "tool_call", "iteration": 2, "name": "compile", "input": {}},
        {"type": "tool_result", "iteration": 2, "name": "compile", "input": {}, "source_hash": "h2",
         "result": res("compile", "fail", "compile failed",
                       command="/opt/dpcpp/bin/icpx /root/.cache/x " +
                               "/home/user/oneapi-agent-toolkit/examples/x/x.h")},
        {"type": "evaluation", "compiled": True, "verify_status": "fail",
         "verify_summary": "FAIL: 0/1 case(s) match", "helpers": 0, "advisories": []},
        {"type": "stop", "stop_reason": "model_finished", "stop_detail": "d", "iterations": 2,
         "usage": {"input_tokens": 10, "output_tokens": 5}, "cost_usd": 0.01, "model": "m",
         "baseline": True, "run_dir": RUN_DIR},
    ]


@pytest.fixture
def page() -> str:
    return str(rt.render_html(_records(), title="T <b>"))


def test_renders_steps_header_and_hidden_check(page: str) -> None:
    assert page.startswith("<!doctype html>")
    assert page.count('class="step"') == 4  # 2 writes, compile, hidden check
    assert "Hidden check the agent never saw" in page
    assert "FAIL: 0/1 case(s) match" in page
    assert "model_finished" in page and "$0.0100" in page
    assert 'class="badge fail"' in page


def test_escapes_trace_content(page: str) -> None:
    assert "<script>alert" not in page
    assert "<script>bad()" not in page
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in page
    assert "T &lt;b&gt;" in page
    # the only script element is the page's own
    assert page.count("<script>") == 1


def test_write_file_diff_against_previous_version(page: str) -> None:
    assert '<span class="dd">-  return 1;</span>' in page
    assert '<span class="da">+  return 0; // &lt;script&gt;alert(1)&lt;/script&gt;</span>' in page
    assert "--- a/migrated/x.cpp" in page
    assert "previous version in this trace" in page


def test_absolute_paths_stripped(page: str) -> None:
    assert "/home/user" not in page
    assert "/root/" not in page
    assert "examples/x/x.h" in page and ".cache/x" in page
    assert "&quot;path&quot;: &quot;migrated/x.cpp&quot;" in page  # evidence path, work-relative


def test_path_stripper_worktree_and_bare_root() -> None:
    s = rt.PathStripper(None)
    assert s("/home/user/oneapi-agent-toolkit/.claude/worktrees/w1/src/a.py") == "src/a.py"
    assert s("cwd /home/user/oneapi-agent-toolkit") == "cwd ."


def test_cli_and_site(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in _records()) + "\n")
    out = tmp_path / "out" / "x.html"
    assert rt.main([str(trace), "-o", str(out), "--title", "Demo"]) == 0
    assert "<title>Demo</title>" in out.read_text()
    manifest = tmp_path / "runs.json"
    manifest.write_text(json.dumps({"title": "Idx", "runs": [
        {"slug": "a", "trace": "trace.jsonl", "title": "A", "compare": True},
        {"slug": "b", "trace": "trace.jsonl", "title": "B", "compare": True}]}))
    assert rt.main(["--site", str(manifest)]) == 0
    index = (tmp_path / "index.html").read_text()
    assert 'href="a.html"' in index and "Side by side" in index
    assert (tmp_path / "b.html").is_file()
