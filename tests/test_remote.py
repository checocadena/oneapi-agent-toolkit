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
    assert r.status == "error" and "became ready" in r.summary
    assert api.deleted == ["p1"] and "stuck" in r.evidence["create_attempts"][-1]["error"]


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
    assert r.status == "error" and "fits max_usd" in r.summary and api.deleted == []
    assert api.pods == {}


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), log=lambda m: None)
    assert r.status == "toolchain_missing"


def test_falls_back_to_the_next_gpu_when_one_is_out_of_stock(tmp_path):
    import io
    import urllib.error

    api = FakeAPI()
    tried = []
    orig = api.create_pod

    def flaky(body):
        tried.append(body["gpu"]["id"])
        if len(tried) == 1:
            raise urllib.error.HTTPError("u", 400, "no stock", {}, io.BytesIO(b"no instances"))
        return orig(body)

    api.create_pod = flaky
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), api=api, poll_s=0.01,
                           max_minutes=0.01, into=str(tmp_path), log=lambda m: None,
                           agent_url=lambda pid: "http://127.0.0.1:9")
    assert len(tried) == 2
    assert "HTTP 400" in r.evidence["create_attempts"][0]["error"] and api.deleted == ["p1"]
    assert r.evidence["create_attempts"][1]["gpu"] == tried[1]



class StuckThenSlowProvider(remote.Provider):
    """First machine reports a broken host; the replacement never answers (boot timeout)."""

    name = "fake"

    def __init__(self):
        self.created, self.deleted, self.bad = [], [], []

    def candidates(self, vendor, gpu_type):
        return ["G"]

    def price(self, gpu):
        return None

    def create(self, gpu, *, vendor, image, env):
        mid = f"m{len(self.created) + 1}"
        self.created.append(mid)
        return mid

    def agent_url(self, machine_id):
        if machine_id == "m1":
            raise remote.MachineStuck("host cannot pull the image: TLS handshake timeout")
        return "http://127.0.0.1:9"

    def delete(self, machine_id):
        self.deleted.append(machine_id)

    def exists(self, machine_id):
        return machine_id not in self.deleted

    def mark_bad(self, machine_id):
        self.bad.append(machine_id)


def test_stuck_machines_are_replaced_then_given_up(tmp_path):
    prov = StuckThenSlowProvider()
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), provider=prov, poll_s=0.01,
                           boot_minutes=0.002, max_stuck=2, max_minutes=5, into=str(tmp_path),
                           log=lambda m: None)
    assert prov.created == ["m1", "m2"] and prov.deleted == ["m1", "m2"] and prov.bad == ["m1", "m2"]
    reasons = [a["error"] for a in r.evidence["create_attempts"]]
    assert "TLS handshake timeout" in reasons[0] and "did not answer within" in reasons[1]
    assert r.status == "error" and "2 stuck machine(s) replaced" in r.summary


def test_amd_without_a_hip_toolchain_image_refuses_before_renting(tmp_path):
    prov = StuckThenSlowProvider()
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), provider=prov, vendor="amd",
                           poll_s=0.01, boot_minutes=0.002, max_stuck=1, max_minutes=5,
                           into=str(tmp_path), log=lambda m: None)
    assert r.status == "toolchain_missing" and "HIP" in r.summary
    assert prov.created == [] and "--image" in r.limitations[0]


class SlowDeleteProvider(StuckThenSlowProvider):
    """Like a DigitalOcean droplet: still listed for a few checks after the DELETE."""

    def __init__(self):
        super().__init__()
        self.checks_after_delete = 0

    def agent_url(self, machine_id):
        raise remote.MachineStuck("broken host")

    def exists(self, machine_id):
        if machine_id not in self.deleted:
            return True
        self.checks_after_delete += 1
        return self.checks_after_delete < 3


def test_asynchronous_delete_is_waited_for_and_confirmed(tmp_path):
    prov = SlowDeleteProvider()
    logs = []
    remote.remote_test(str(REPO / "examples/saxpy/task.json"), provider=prov, poll_s=0.01,
                       max_stuck=1, max_minutes=5, into=str(tmp_path), log=logs.append)
    assert prov.deleted == ["m1"] and prov.checks_after_delete == 3
    assert any("deleted fake machine m1: confirmed" in m for m in logs), logs
