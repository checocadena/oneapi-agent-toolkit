"""Pricing for any model, so the per-run USD cap always applies; real-model preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oneapi_agent.loop import (
    PRICE_ENV,
    AgentLoop,
    LoopConfig,
    MockToolbox,
    ScriptedModel,
    dry_run_script,
    estimate_cost,
    mock_queues,
    model_pricing,
    preflight_real_model,
)

USAGE = {"input_tokens": 1_000_000, "output_tokens": 100_000,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1_000_000}


@pytest.fixture(autouse=True)
def _no_price_env(monkeypatch):
    for k in (*PRICE_ENV, "ONEAPI_AGENT_ALLOW_UNPRICED"):
        monkeypatch.delenv(k, raising=False)


def test_table_prices_are_unchanged():
    p = model_pricing("claude-sonnet-5-5")
    assert (p.input, p.output, p.source) == (2.0, 10.0, "built-in table")
    assert estimate_cost("claude-sonnet-5-5", USAGE) == pytest.approx(2.0 + 1.0 + 0.2)


def test_unknown_model_has_no_price_and_says_how_to_fix_it():
    with pytest.raises(ValueError, match="ONEAPI_AGENT_PRICE_IN"):
        model_pricing("some-open-model")
    assert estimate_cost("some-open-model", USAGE) is None


def test_env_override_prices_any_model():
    env = {"ONEAPI_AGENT_PRICE_IN": "1.25", "ONEAPI_AGENT_PRICE_OUT": "4.25"}
    p = model_pricing("some-open-model", env)
    assert (p.input, p.output, p.cache_read, p.source) == (1.25, 4.25, 1.25, "env override")
    # cached input defaults to the full input price (conservative); it can be set explicitly
    assert estimate_cost("some-open-model", USAGE, env) == pytest.approx(1.25 + 0.425 + 1.25)
    env["ONEAPI_AGENT_PRICE_CACHED_IN"] = "0.15"
    assert estimate_cost("some-open-model", USAGE, env) == pytest.approx(1.25 + 0.425 + 0.15)


@pytest.mark.parametrize("env, msg", [
    ({"ONEAPI_AGENT_PRICE_IN": "1"}, "set both"),
    ({"ONEAPI_AGENT_PRICE_IN": "x", "ONEAPI_AGENT_PRICE_OUT": "1"}, "must be numbers"),
    ({"ONEAPI_AGENT_PRICE_IN": "-1", "ONEAPI_AGENT_PRICE_OUT": "1"}, "negative"),
])
def test_bad_override_is_an_error_not_a_free_run(env, msg):
    with pytest.raises(ValueError, match=msg):
        model_pricing("claude-sonnet-5-5", env)


def test_preflight_needs_an_api_key():
    with pytest.raises(SystemExit, match="no API key"):
        preflight_real_model("claude-sonnet-5-5", {})
    preflight_real_model("claude-sonnet-5-5", {"ONEAPI_AGENT_API_KEY": "k"})


def test_preflight_refuses_an_unpriced_model_unless_allowed():
    key = {"ONEAPI_AGENT_API_KEY": "k"}
    with pytest.raises(SystemExit, match="no price known"):
        preflight_real_model("some-open-model", key)
    said: list[str] = []
    preflight_real_model("some-open-model", {**key, "ONEAPI_AGENT_ALLOW_UNPRICED": "1"},
                         warn=said.append)
    assert said and "only by tokens" in said[0]
    preflight_real_model("some-open-model", {**key, "ONEAPI_AGENT_PRICE_IN": "1",
                                             "ONEAPI_AGENT_PRICE_OUT": "4"})
    # the opt-in does not excuse a malformed override
    with pytest.raises(SystemExit, match="set both"):
        preflight_real_model("some-open-model", {**key, "ONEAPI_AGENT_ALLOW_UNPRICED": "1",
                                                 "ONEAPI_AGENT_PRICE_IN": "1"})


class PricedFake:
    """A non-scripted model name driving the scripted turns (ScriptedModel itself is free)."""

    def __init__(self, inner: ScriptedModel, name: str) -> None:
        self.inner, self.name = inner, name

    def create(self, **kw):
        return self.inner.create(**kw)


def _loop(tmp_path: Path, name: str) -> AgentLoop:
    cu = tmp_path / "k.cu"
    cu.write_text("__global__ void k(float* x){ x[threadIdx.x] = 1.f; }\nint main(){}\n")
    cfg = LoopConfig(goal="port k.cu", cuda_path=str(cu), reference_cmd="ref {output}",
                     candidate_cmd="cand {output}", max_iterations=12, name="unit",
                     max_cost_usd=0.5, max_total_tokens=10**9)
    model = PricedFake(ScriptedModel(dry_run_script("max_iterations", cfg),
                                     tokens_per_turn=(200_000, 0)), name)
    queues = mock_queues("max_iterations")
    return AgentLoop(cfg, model, runs_dir=tmp_path / "runs",
                     toolbox_factory=lambda c, w: MockToolbox(c, w, queues), run_name="r")


def test_cost_cap_stops_an_overridden_model(tmp_path, monkeypatch):
    monkeypatch.setenv("ONEAPI_AGENT_PRICE_IN", "1.25")
    monkeypatch.setenv("ONEAPI_AGENT_PRICE_OUT", "4.25")
    out = _loop(tmp_path, "some-open-model").run()
    # 200k input tokens per turn at $1.25/M = $0.25 per turn; cap $0.50 -> stop after turn 3
    assert out["stop_reason"] == "budget" and "cost budget" in out["stop_detail"]
    assert out["iterations"] == 3


def test_unpriced_run_is_flagged_in_the_trace(tmp_path, capsys):
    loop = _loop(tmp_path, "some-open-model")
    loop.cfg.max_iterations = 2
    loop.run()
    warnings = [json.loads(x) for x in loop.trace_path.read_text().splitlines()
                if json.loads(x)["type"] == "warning"]
    assert len(warnings) == 1 and "cannot apply" in warnings[0]["message"]
    assert "WARNING: no price known" in capsys.readouterr().err
