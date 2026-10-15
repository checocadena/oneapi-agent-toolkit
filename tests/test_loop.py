"""Agent loop in dry-run mode: scripted model + mock tools (no API key, no toolchain)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oneapi_agent.loop import (
    AgentLoop,
    LoopConfig,
    MockToolbox,
    ScriptedModel,
    Toolbox,
    dry_run_script,
    estimate_cost,
    failure_signature,
    mock_queues,
    tool_schemas,
)
from oneapi_agent.result import ToolResult


@pytest.fixture
def cfg(tmp_path: Path) -> LoopConfig:
    cu = tmp_path / "k.cu"
    cu.write_text("__global__ void k(float* x){ x[threadIdx.x] = 1.f; }\nint main(){}\n")
    return LoopConfig(goal="port k.cu", cuda_path=str(cu), reference_cmd="ref {output}",
                      candidate_cmd="cand {output}", max_iterations=12, name="unit")


def make_loop(cfg, tmp_path, script, queues, **kw):
    return AgentLoop(cfg, ScriptedModel(script, **kw), runs_dir=tmp_path / "runs",
                     toolbox_factory=lambda c, w: MockToolbox(c, w, queues), run_name="r")


def read_trace(loop: AgentLoop) -> list[dict]:
    return [json.loads(line) for line in loop.trace_path.read_text().splitlines()]


def test_success_path(cfg, tmp_path):
    loop = make_loop(cfg, tmp_path, dry_run_script("success", cfg, "int main(){}\n"),
                     mock_queues("success"))
    out = loop.run()
    assert out["stop_reason"] == "success"
    assert out["best"] is not None and out["best"]["helpers"] == 0
    assert Path(out["best"]["path"], "migrated", "k.dp.cpp").is_file()
    # the deliberately broken edit failed to compile and was then fixed
    hist = [(e.name, e.result.status) for e in loop.state.history]
    assert ("compile", "fail") in hist
    trace = read_trace(loop)
    kinds = {t["type"] for t in trace}
    assert {"start", "model_response", "tool_call", "tool_result", "best_snapshot", "stop"} <= kinds
    report = Path(out["report"]).read_text()
    assert "`success`" in report
    assert "fixed by iteration" in report
    assert "MOCK RESULT" in report
    # harness ran the final benchmark on success
    assert loop.state.history[-1].name == "benchmark"


def test_stuck_path(cfg, tmp_path):
    loop = make_loop(cfg, tmp_path, dry_run_script("stuck", cfg), mock_queues("stuck"))
    out = loop.run()
    assert out["stop_reason"] == "stuck"
    assert out["iterations"] == 3
    assert "undeclared_var" in out["stop_detail"]
    assert out["best"] is None
    assert "**not fixed**" in Path(out["report"]).read_text()


def test_max_iterations_path(cfg, tmp_path):
    cfg.max_iterations = 5
    loop = make_loop(cfg, tmp_path, dry_run_script("max_iterations", cfg),
                     mock_queues("max_iterations"))
    out = loop.run()
    assert out["stop_reason"] == "max_iterations"
    assert out["iterations"] == 5
    assert out["best"] is None


def test_max_iterations_keeps_best_passing(cfg, tmp_path):
    """Verify passes once (helpers remain), later edits break it: best snapshot survives."""
    cfg.max_iterations = 4
    script = [
        [("write_file", {"path": "p.cpp", "content": "dpct::get_current_device();\n"}),
         ("compile", {"sources": ["p.cpp"]}), ("verify", {})],
        [("write_file", {"path": "p.cpp", "content": "// broken\n"}),
         ("compile", {"sources": ["p.cpp"]})],
        [("write_file", {"path": "p.cpp", "content": "// broken 2\n"}),
         ("compile", {"sources": ["p.cpp"]})],
        [("write_file", {"path": "p.cpp", "content": "// broken 3\n"}),
         ("compile", {"sources": ["p.cpp"]})],
    ]
    ok = ToolResult("compile", "ok", "ok", {"diagnostics": []})
    queues = {
        "compile": [ok] + [ToolResult("compile", "fail", f"e{i}", {"diagnostics": [
            {"file": "p.cpp", "line": i, "severity": "error", "message": f"e{i}"}]})
            for i in range(5)],
        "verify": [ToolResult("verify", "ok", "PASS (mock)", {"cases": [{"passed": True}]})],
    }
    loop = make_loop(cfg, tmp_path, script, queues)
    out = loop.run()
    assert out["stop_reason"] == "max_iterations"
    assert out["best"]["iteration"] == 1 and out["best"]["helpers"] == 1
    assert (Path(out["best"]["path"]) / "p.cpp").read_text() == "dpct::get_current_device();\n"


def test_verify_on_stale_binary_does_not_count(cfg, tmp_path):
    """compile ok -> edit -> verify ok: binary is stale, so no success and no snapshot."""
    script = [
        [("write_file", {"path": "p.cpp", "content": "int a;\n"}),
         ("compile", {"sources": ["p.cpp"]})],
        [("write_file", {"path": "p.cpp", "content": "int b;\n"}),
         ("verify", {}), ("scan_helpers", {})],
        "done",
        "done",
    ]
    queues = {"compile": [ToolResult("compile", "ok", "ok")],
              "verify": [ToolResult("verify", "ok", "PASS (mock)")],
              "scan_helpers": [ToolResult("scan_helpers", "ok", "none")]}
    out = make_loop(cfg, tmp_path, script, queues).run()
    assert out["stop_reason"] == "model_stopped"
    assert out["best"] is None


def test_finish_is_rejected_without_evidence(cfg, tmp_path):
    script = [[("finish", {"summary": "trust me, it works"})], "ok", "ok"]
    loop = make_loop(cfg, tmp_path, script, {})
    out = loop.run()
    assert out["stop_reason"] == "model_stopped"
    finish = [t for t in read_trace(loop) if t["type"] == "tool_result" and t["name"] == "finish"]
    assert finish[0]["result"]["status"] == "fail"
    assert "rejected" in finish[0]["result"]["summary"]


def test_token_budget_stop(cfg, tmp_path):
    cfg.max_total_tokens = 5000
    loop = make_loop(cfg, tmp_path, dry_run_script("max_iterations", cfg),
                     mock_queues("max_iterations"), tokens_per_turn=(3000, 0))
    out = loop.run()
    assert out["stop_reason"] == "budget"
    assert out["iterations"] == 2
    assert "token budget" in out["stop_detail"]


def test_cost_estimate_and_cost_budget(cfg, tmp_path):
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1_000_000}
    assert estimate_cost("claude-sonnet-5-5", usage) == pytest.approx(2.0 + 1.0 + 0.2)
    assert estimate_cost("unknown-model", usage) is None


def test_toolbox_confines_paths(cfg, tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    tb = Toolbox(cfg, work)
    r = tb("write_file", {"path": "../escape.txt", "content": "x"})
    assert r.status == "error" and "escapes" in r.summary
    assert not (tmp_path / "escape.txt").exists()
    assert tb("read_file", {"path": "/etc/passwd"}).status == "error"
    assert tb("read_file", {"path": cfg.cuda_path}).status == "ok"  # task source is readable
    assert tb("nope", {}).status == "error"


def test_failure_signature():
    assert failure_signature(ToolResult("compile", "ok", "")) is None
    assert failure_signature(ToolResult("scan_helpers", "fail", "3 remain")) is None
    sig = failure_signature(ToolResult("compile", "fail", "x", {"diagnostics": [
        {"file": "/a/b/k.cpp", "line": 3, "severity": "error", "message": "boom"}]}))
    assert sig == "compile:k.cpp:3:boom"
    vs = failure_signature(ToolResult("verify", "fail", "x", {"cases": [
        {"passed": False, "reason": "2 of 9 element(s) outside tolerance"}]}))
    assert vs == "verify:2 of 9 element(s) outside tolerance"


def test_task_file_loading(tmp_path):
    (tmp_path / "a.cu").write_text("")
    (tmp_path / "in.wav").write_bytes(b"")
    (tmp_path / "task.json").write_text(json.dumps({
        "goal": "g", "cuda_path": "a.cu", "reference_cmd": "r", "candidate_cmd": "c",
        "inputs": ["in.wav"], "reference_build": ["g++", "{task_dir}/ref.cpp"],
        "include_dirs": ["."], "mode": "audio"}))
    c = LoopConfig.from_task_file(tmp_path / "task.json", max_iterations=3)
    assert c.cuda_path == str((tmp_path / "a.cu").resolve())
    assert c.inputs == [str((tmp_path / "in.wav").resolve())]
    assert c.reference_build == ["g++", f"{tmp_path.resolve()}/ref.cpp"]
    assert c.max_iterations == 3 and c.mode == "audio"


def test_task_file_expands_env_in_base_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("ONEMATH_PREFIX", "/x/onemath")
    (tmp_path / "a.cu").write_text("")
    (tmp_path / "task.json").write_text(json.dumps({
        "goal": "g", "cuda_path": "a.cu", "reference_cmd": "r", "candidate_cmd": "c",
        "base_flags": ["-O2", "-I${ONEMATH_PREFIX}/include", "-lonemath"]}))
    c = LoopConfig.from_task_file(tmp_path / "task.json")
    assert c.base_flags == ["-O2", "-I/x/onemath/include", "-lonemath"]


def test_repo_example_task_files_load():
    root = Path(__file__).resolve().parents[1] / "examples"
    tasks = sorted(root.glob("*/task.json"))
    assert len(tasks) == 5
    for t in tasks:
        c = LoopConfig.from_task_file(t)
        assert Path(c.cuda_path).is_file()
        assert (t.parent / "sycl" / (Path(c.cuda_path).stem + ".cpp")).is_file()


def test_client_prefers_oneapi_agent_key(monkeypatch):
    from oneapi_agent.loop import make_anthropic_client

    calls = []

    class FakeSDK:
        @staticmethod
        def Anthropic(**kw):  # noqa: N802 - mirrors the SDK class name
            calls.append(kw)
            return kw

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://host-internal.invalid")
    monkeypatch.setenv("ONEAPI_AGENT_API_KEY", "k-test")
    monkeypatch.delenv("ONEAPI_AGENT_BASE_URL", raising=False)
    assert make_anthropic_client(FakeSDK) == {"api_key": "k-test",
                                              "base_url": "https://api.anthropic.com"}
    monkeypatch.delenv("ONEAPI_AGENT_API_KEY")
    assert make_anthropic_client(FakeSDK) == {}  # falls back to the SDK's own lookup


def test_baseline_hides_checks_and_ends_on_finish(cfg, tmp_path):
    cfg.baseline = True
    names = {t["name"] for t in tool_schemas(baseline=True)}
    assert not names & {"scan_helpers", "verify", "benchmark"} and "compile" in names
    script = [[("verify", {})],  # not offered, but a model could still try
              [("finish", {"summary": "done"})]]
    loop = make_loop(cfg, tmp_path, script, {})
    out = loop.run()
    assert out["stop_reason"] == "model_finished" and out["baseline"] is True
    assert loop.state.history[0].result.status == "error"  # verify refused, not run
    assert "Verify compares" not in loop.messages[0]["content"]
    assert "verify-passed" not in loop._state_note()


def test_evaluate_reports_never_compiled(cfg, tmp_path):
    cfg.baseline, cfg.evaluate = True, True
    out = make_loop(cfg, tmp_path, [[("finish", {"summary": "done"})]], {}).run()
    assert out["evaluation"]["compiled"] is False
    assert out["evaluation"]["note"] == "the model never compiled"
