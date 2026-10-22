"""Model providers: selection, the OpenAI-compatible (Meta) client, and the loop over it.

Everything here runs offline: the Meta path uses ``ScriptedTransport``, a fake server that
answers with real chat-completion JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oneapi_agent.cli import main
from oneapi_agent.loop import (
    AgentLoop,
    LoopConfig,
    MockToolbox,
    dry_run_script,
    mock_queues,
    preflight_real_model,
)
from oneapi_agent.models import (
    OpenAICompatModel,
    ScriptedModel,
    ScriptedTransport,
    from_openai_response,
    make_model,
    resolve_provider,
    to_openai_messages,
    to_openai_tools,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path: Path) -> LoopConfig:
    cu = tmp_path / "k.cu"
    cu.write_text("__global__ void k(float* x){ x[threadIdx.x] = 1.f; }\nint main(){}\n")
    return LoopConfig(goal="port k.cu", cuda_path=str(cu), reference_cmd="ref {output}",
                      candidate_cmd="cand {output}", max_iterations=12, name="unit",
                      model="muse-spark-1.3", provider="meta")


def test_provider_resolution_order():
    assert resolve_provider("claude-sonnet-5-5", environ={}) == "anthropic"
    assert resolve_provider("muse-spark-1.3", environ={}) == "meta"
    assert resolve_provider("some-model", environ={}) == "anthropic"
    assert resolve_provider("claude-x", environ={"ONEAPI_AGENT_PROVIDER": "meta"}) == "meta"
    assert resolve_provider("muse-x", "anthropic", {"ONEAPI_AGENT_PROVIDER": "meta"}) == \
        "anthropic"
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_provider("x", "openrouter", {})


def test_history_translates_to_openai_messages():
    history = [
        {"role": "user", "content": "port it"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "migrating"},
            {"type": "tool_use", "id": "t1", "name": "migrate", "input": {"path": "k.cu"}},
            {"type": "tool_use", "id": "t2", "name": "list_files", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": '{"status": "ok"}'},
            {"type": "tool_result", "tool_use_id": "t2", "content": "[]", "is_error": True},
            {"type": "text", "text": "state: iteration 1"}]},
    ]
    out = to_openai_messages("be careful", history)
    assert out[0] == {"role": "system", "content": "be careful"}
    assert out[1] == {"role": "user", "content": "port it"}
    assert out[2]["role"] == "assistant" and out[2]["content"] == "migrating"
    assert [c["function"]["name"] for c in out[2]["tool_calls"]] == ["migrate", "list_files"]
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"path": "k.cu"}
    # tool results first, in order, then the loop's state note as a user message
    assert [(m["role"], m.get("tool_call_id")) for m in out[3:]] == \
        [("tool", "t1"), ("tool", "t2"), ("user", None)]
    assert out[5]["content"] == "state: iteration 1"


def test_tool_schemas_translate():
    t = to_openai_tools([{"name": "verify", "description": "d",
                          "input_schema": {"type": "object", "properties": {}}}])
    assert t == [{"type": "function", "function": {
        "name": "verify", "description": "d", "parameters": {"type": "object", "properties": {}}}}]


def test_response_translates_tool_calls_usage_and_stop_reasons():
    r = from_openai_response({
        "model": "muse-spark-1.3",
        "choices": [{"finish_reason": "stop", "message": {
            "content": "calling", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "compile", "arguments": '{"sources": ["a.cpp"]}'}},
                {"id": "c2", "type": "function",
                 "function": {"name": "verify", "arguments": "{not json"}}]}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 50,
                  "prompt_tokens_details": {"cached_tokens": 400}}})
    assert r.stop_reason == "tool_use"  # tool calls win over a "stop" finish reason
    assert [u["name"] for u in r.tool_uses] == ["compile", "verify"]
    assert r.tool_uses[0]["input"] == {"sources": ["a.cpp"]}
    assert r.tool_uses[1]["input"] == {"_unparsed_arguments": "{not json"}
    assert r.usage == {"input_tokens": 600, "output_tokens": 50,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 400}
    assert r.text == "calling" and r.model == "muse-spark-1.3"
    assert from_openai_response({"choices": [{"finish_reason": "length", "message": {
        "content": "x"}}]}).stop_reason == "max_tokens"
    assert from_openai_response({"choices": [{"finish_reason": "content_filter", "message": {
        "content": None}}]}).stop_reason == "refusal"


def test_client_posts_to_base_url_with_bearer_key(monkeypatch):
    monkeypatch.setenv("ONEAPI_AGENT_META_BASE_URL", "https://meta.example/v1/")
    seen = {}

    def transport(url, headers, body):
        seen.update(url=url, headers=headers, body=body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}

    m = OpenAICompatModel("muse-spark-1.3", api_key="sk-test", transport=transport)
    r = m.create(system="s", messages=[{"role": "user", "content": "q"}], tools=[])
    assert seen["url"] == "https://meta.example/v1/chat/completions"
    assert seen["headers"] == {"Authorization": "Bearer sk-test"}
    assert seen["body"]["model"] == "muse-spark-1.3" and "tools" not in seen["body"]
    assert r.text == "hi" and r.model == "muse-spark-1.3"


def test_client_reads_meta_key_from_env(monkeypatch):
    monkeypatch.delenv("ONEAPI_AGENT_META_API_KEY", raising=False)
    monkeypatch.setenv("META_API_KEY", "sk-env")
    assert OpenAICompatModel("muse-spark-1.3", transport=lambda *a: {}).api_key == "sk-env"


def test_make_model_dry_run_picks_the_provider_path():
    script = [[("list_files", {})]]
    assert isinstance(make_model("anthropic", "claude-sonnet-5-5", dry_run_script=script),
                      ScriptedModel)
    meta = make_model("meta", "muse-spark-1.3", dry_run_script=script)
    assert isinstance(meta, OpenAICompatModel) and meta.is_mock
    assert isinstance(meta.transport, ScriptedTransport)


def test_loop_reaches_success_through_the_meta_client(cfg, tmp_path):
    transport = ScriptedTransport(dry_run_script("success", cfg, "int main(){}\n"))
    model = OpenAICompatModel("muse-spark-1.3", api_key="", transport=transport)
    model.is_mock = True
    loop = AgentLoop(cfg, model, runs_dir=tmp_path / "runs", run_name="r",
                     toolbox_factory=lambda c, w: MockToolbox(c, w, mock_queues("success")))
    out = loop.run()
    assert out["stop_reason"] == "success"
    assert loop.state.cost_usd == 0.0
    # every request after the first carries the previous turn's tool results
    url, body = transport.requests[1]
    assert url.endswith("/chat/completions") and body["model"] == "muse-spark-1.3"
    assert {t["function"]["name"] for t in body["tools"]} >= {"migrate", "verify", "finish"}
    assert any(m["role"] == "tool" for m in body["messages"])
    start = json.loads(loop.trace_path.read_text().splitlines()[0])
    assert start["config"]["provider"] == "meta" and start["model"] == "muse-spark-1.3"


def test_preflight_checks_the_providers_own_key():
    with pytest.raises(SystemExit, match="ONEAPI_AGENT_META_API_KEY"):
        preflight_real_model("muse-spark-1.3", {"ANTHROPIC_API_KEY": "k",
                                                "ONEAPI_AGENT_ALLOW_UNPRICED": "1"},
                             provider="meta")
    preflight_real_model("muse-spark-1.3", {"META_API_KEY": "k", "ONEAPI_AGENT_PRICE_IN": "1",
                                            "ONEAPI_AGENT_PRICE_OUT": "4"}, provider="meta")
    with pytest.raises(SystemExit, match="no price known"):
        preflight_real_model("muse-spark-1.3", {"META_API_KEY": "k"}, provider="meta")


def test_cli_dry_run_with_meta_provider_needs_no_key(tmp_path, monkeypatch, capsys):
    for k in ("META_API_KEY", "ONEAPI_AGENT_META_API_KEY", "ONEAPI_AGENT_MODEL",
              "ONEAPI_AGENT_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    code = main(["loop", "--dry-run", "--mock-tools", "--provider", "meta",
                 "--config", str(REPO / "examples" / "saxpy" / "task.json"),
                 "--runs-dir", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["stop_reason"] == "success"
    trace = Path(out["run_dir"], "trace.jsonl").read_text().splitlines()
    assert json.loads(trace[0])["model"] == "muse-spark-1.3"
