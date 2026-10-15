"""remote_test control flow with a fake Runpod API (no network, no GPU)."""

from __future__ import annotations

from pathlib import Path

from oneapi_agent import remote

REPO = Path(__file__).resolve().parents[1]


class FakeAPI:
    def __init__(self, price=None, fail_create=False):
        self.pods, self.price, self.fail_create, self.deleted = {}, price, fail_create, []

    def gpu_price(self, gpu, cloud):
        return self.price

    def create_pod(self, body):
        if self.fail_create:
            raise RuntimeError("no stock")
        self.pods["p1"] = body
        return {"id": "p1"}

    def get_pod(self, pid):
        return self.pods.get(pid)

    def delete_pod(self, pid):
        self.deleted.append(pid)
        self.pods.pop(pid, None)


def test_pod_is_deleted_when_the_agent_never_answers(tmp_path):
    api = FakeAPI()
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), api=api, poll_s=0.01,
                           max_minutes=0.01, into=str(tmp_path), log=lambda m: None,
                           agent_url=lambda pid: "http://127.0.0.1:9")  # nothing listens there
    assert r.status == "error" and "TimeoutError" in r.summary
    assert api.deleted == ["p1"] and r.evidence["pod_deleted"] is True


def test_create_body_carries_token_agent_and_port(monkeypatch):
    seen = {}
    api = FakeAPI()
    orig = api.create_pod

    def capture(body):
        seen.update(body)
        return orig(body)

    monkeypatch.setattr(api, "create_pod", capture)
    remote.remote_test(str(REPO / "examples/saxpy/task.json"), api=api, poll_s=0.01,
                       max_minutes=0.01, log=lambda m: None,
                       agent_url=lambda pid: "http://127.0.0.1:9")
    assert seen["ports"] == ["8000/http"] and len(seen["env"]["GT_TOKEN"]) >= 24
    assert seen["env"]["GT_AGENT"] and seen["gpu"]["minCudaVersion"] == "12.4"


def test_budget_too_small_for_the_gpu_price_refuses_before_creating():
    api = FakeAPI(price=6.0)
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), api=api, max_usd=0.5,
                           log=lambda m: None)
    assert r.status == "error" and "allows only" in r.summary and api.deleted == []
    assert api.pods == {}


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), log=lambda m: None)
    assert r.status == "toolchain_missing"
