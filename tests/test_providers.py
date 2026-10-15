"""Vast.ai / Vultr providers against fake APIs, and the agent's pinned-TLS path for real."""

from __future__ import annotations

import base64
import io
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import pytest

from oneapi_agent import remote

REPO = Path(__file__).resolve().parents[1]


class FakeJSON:
    """Records requests; answers from a routing table of (method, path-prefix) -> response."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def req(self, method, path, body=None):
        self.calls.append((method, path, body))
        for (m, prefix), resp in self.routes.items():
            if m == method and path.startswith(prefix):
                if isinstance(resp, Exception):
                    raise resp
                return resp(body) if callable(resp) else resp
        raise AssertionError(f"unexpected {method} {path}")


def http_error(code):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b"gone"))


# ---------------------------------------------------------------- Vast.ai

def test_vast_rents_cheapest_offer_with_port_mapping_and_skips_taken_offers():
    offers = {"offers": [{"id": 11, "dph_total": 0.12}, {"id": 12, "dph_total": 0.15}]}
    calls = {"n": 0}

    def rent(body):
        calls["n"] += 1
        return {"success": True, "new_contract": 777}

    api = FakeJSON({("POST", "/bundles/"): offers, ("PUT", "/asks/11/"): http_error(410),
                    ("PUT", "/asks/12/"): rent})
    v = remote.VastProvider("k", api=api)
    assert v.price("RTX A4000") == 0.12
    mid = v.create("RTX A4000", vendor="nvidia", image=None, env={"GT_TOKEN": "t"})
    assert mid == "777"
    search = next(b for m, p, b in api.calls if p == "/bundles/")
    assert search["gpu_name"] == {"in": ["RTX A4000"]} and search["direct_port_count"] == {"gte": 1}
    put = next(b for m, p, b in api.calls if p == "/asks/12/")
    assert put["env"]["-p 8000:8000"] == "1" and put["env"]["GT_TOKEN"] == "t"
    assert put["runtype"] == "args" and put["args"][:2] == ["bash", "-c"]


def test_vast_no_offers_is_unavailable():
    v = remote.VastProvider("k", api=FakeJSON({("POST", "/bundles/"): {"offers": []}}))
    with pytest.raises(remote.Unavailable):
        v.create("RTX 4090", vendor="nvidia", image=None, env={})
    assert remote.VastProvider("k", api=FakeJSON({})).candidates("amd", None) == []


def test_vast_agent_url_from_mapped_port_and_dead_instance_raises():
    running = {"instances": {"actual_status": "running", "public_ipaddr": "1.2.3.4\n",
                             "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41234"}]}}}
    v = remote.VastProvider("k", api=FakeJSON({("GET", "/instances/9/"): running}))
    assert v.agent_url("9") == "https://1.2.3.4:41234"
    loading = {"instances": {"actual_status": "loading", "ports": {}}}
    assert remote.VastProvider("k", api=FakeJSON({("GET", "/instances/9/"): loading})).agent_url("9") is None
    dead = {"instances": {"actual_status": "exited", "status_msg": "image pull failed"}}
    with pytest.raises(RuntimeError, match="exited"):
        remote.VastProvider("k", api=FakeJSON({("GET", "/instances/9/"): dead})).agent_url("9")
    gone = remote.VastProvider("k", api=FakeJSON({("GET", "/instances/9/"): {"instances": None}}))
    assert gone.exists("9") is False


# ---------------------------------------------------------------- Vultr

PLANS = {"plans": [{"id": "vcg-a16-6c-60g-16vram", "hourly_cost": 0.471, "locations": ["ewr"]},
                   {"id": "vcg-a40-24c-120g-48vram", "hourly_cost": 1.712, "locations": []}]}


def test_vultr_creates_vm_with_cloud_init_that_starts_the_agent():
    created = {}

    def create(body):
        created.update(body)
        return {"instance": {"id": "abc-123"}}

    api = FakeJSON({("GET", "/plans"): PLANS, ("POST", "/instances"): create})
    v = remote.VultrProvider("k", api=api)
    assert v.price("vcg-a16-6c-60g-16vram") == 0.471
    mid = v.create("vcg-a16-6c-60g-16vram", vendor="nvidia", image=None,
                   env={"GT_TOKEN": "tok", "GT_AGENT": "QUJD"})
    assert mid == "abc-123" and created["region"] == "ewr" and created["os_id"] == 2284
    script = base64.b64decode(created["user_data"]).decode()
    assert script.startswith("#!/bin/bash") and "export GT_TOKEN=tok" in script
    assert "ufw allow 8000/tcp" in script and "gt_agent.py" in script
    with pytest.raises(remote.Unavailable, match="no deployable locations"):
        v.create("vcg-a40-24c-120g-48vram", vendor="nvidia", image=None, env={})


def test_vultr_agent_url_waits_for_active_and_an_ip():
    pending = {"instance": {"status": "pending", "main_ip": "0.0.0.0"}}
    active = {"instance": {"status": "active", "main_ip": "5.6.7.8"}}
    assert remote.VultrProvider("k", api=FakeJSON({("GET", "/instances/x"): pending})).agent_url("x") is None
    assert remote.VultrProvider("k", api=FakeJSON({("GET", "/instances/x"): active})).agent_url("x") \
        == "https://5.6.7.8:8000"
    gone = remote.VultrProvider("k", api=FakeJSON({("GET", "/instances/x"): http_error(404)}))
    assert gone.exists("x") is False


def test_missing_provider_key_is_reported(monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), provider="vast",
                           log=lambda m: None)
    assert r.status == "toolchain_missing" and "VAST_API_KEY" in r.summary


# ---------------------------------------------------------------- pinned TLS, for real

class LocalAgentProvider(remote.Provider):
    """Runs the real POD_AGENT as a local subprocess; reached over pinned TLS like Vast/Vultr."""

    name, tls = "local", True

    def __init__(self, workdir: Path):
        self.workdir, self.proc, self.port = workdir, None, None

    def candidates(self, vendor, gpu_type):
        return ["local-cpu"]

    def price(self, gpu):
        return None

    def create(self, gpu, *, vendor, image, env):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        agent = self.workdir / "agent.py"
        agent.write_text(base64.b64decode(env["GT_AGENT"]).decode())
        full = {**__import__("os").environ, **env, "GT_DIR": str(self.workdir / "m"),
                "GT_PORT": str(self.port), "GT_NO_SETUP": "1"}
        self.proc = subprocess.Popen([sys.executable, str(agent)], env=full,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "m1"

    def agent_url(self, machine_id):
        return f"https://127.0.0.1:{self.port}"

    def delete(self, machine_id):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(10)
            self.proc = None

    def exists(self, machine_id):
        return self.proc is not None


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")
def test_end_to_end_over_pinned_tls(tmp_path):
    prov = LocalAgentProvider(tmp_path)
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), provider=prov, poll_s=0.5,
                           max_minutes=10, into=str(tmp_path / "into"), log=lambda m: None)
    # No GPU here: the remote gpu_test reports that honestly, and the archive still comes back.
    assert r.status in ("toolchain_missing", "ok", "fail"), r.summary
    assert r.evidence["agent_url"].startswith("https://") and r.evidence["pod_deleted"] is True
    assert list((tmp_path / "into").rglob("summary.json"))


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")
def test_agent_rejects_unpinned_and_wrong_certificates(tmp_path):
    pair = remote.make_tls_pair(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    other_pair = remote.make_tls_pair(other)
    assert pair and other_pair
    prov = LocalAgentProvider(tmp_path)
    env = {"GT_TOKEN": "tok", "GT_AGENT": base64.b64encode(remote.POD_AGENT.encode()).decode(),
           "GT_TLS_CERT": base64.b64encode(pair[0].read_bytes()).decode(),
           "GT_TLS_KEY": base64.b64encode(pair[1].read_bytes()).decode()}
    prov.create("x", vendor="nvidia", image=None, env=env)
    try:
        url = prov.agent_url("m1") + "/status"
        for _ in range(50):
            code, _ = remote._agent_call(url, "tok", context=remote.pinned_context(pair[0]))
            if code == 200:
                break
            time.sleep(0.2)
        assert code == 200
        assert remote._agent_call(url, "bad", context=remote.pinned_context(pair[0]))[0] == 403
        assert remote._agent_call(url, "tok", context=remote.pinned_context(other_pair[0]))[0] == 0
        assert remote._agent_call(url, "tok")[0] == 0  # system CAs do not trust it
    finally:
        prov.delete("m1")
