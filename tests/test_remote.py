"""remote_test control flow with a fake Runpod API (no network, no GPU)."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

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
        self.created, self.deleted, self.bad, self.envs = [], [], [], []

    def candidates(self, vendor, gpu_type):
        return ["G"]

    def price(self, gpu):
        return None

    def create(self, gpu, *, vendor, image, env):
        self.envs.append(env)
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


def _amd_run(tmp_path, **kw):
    prov = StuckThenSlowProvider()
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), provider=prov,
                           poll_s=0.01, boot_minutes=0.002, max_stuck=1, max_minutes=5,
                           into=str(tmp_path), log=lambda m: None, **kw)
    return prov, r


def test_amd_without_any_hip_toolchain_refuses_before_renting(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "DEFAULT_SYCL", {})
    prov, r = _amd_run(tmp_path, vendor="amd")
    assert r.status == "toolchain_missing" and "HIP" in r.summary
    assert prov.created == [] and "--sycl-url" in r.limitations[0]


def test_amd_installs_the_published_hip_toolchain_by_default(tmp_path):
    prov, r = _amd_run(tmp_path, vendor="amd")
    url, sha = remote.DEFAULT_SYCL["amd"]
    assert prov.created and all(e["GT_SYCL_URL"] == url for e in prov.envs)
    assert url.startswith("https://github.com/checocadena/oneapi-agent-toolkit/releases/download/")
    assert all(e["GT_SYCL_SHA256"] == sha for e in prov.envs) and len(sha) == 64
    assert r.evidence["sycl_url"] == url


def test_an_image_brings_its_own_compiler_and_sycl_url_overrides(tmp_path):
    prov, _ = _amd_run(tmp_path, vendor="amd", image="my/hip-sycl:1")
    assert prov.created and all("GT_SYCL_URL" not in e for e in prov.envs)
    prov, _ = _amd_run(tmp_path, vendor="nvidia", sycl_url="https://x/t.tar.gz",
                       sycl_sha256="ab" * 32)
    assert prov.envs[0]["GT_SYCL_URL"] == "https://x/t.tar.gz"
    assert prov.envs[0]["GT_SYCL_SHA256"] == "ab" * 32


def _setup_url_block(prefix):
    """The agent's GT_SYCL_URL install step, pointed at a scratch prefix instead of /opt/sycl."""
    setup = remote.POD_AGENT.split('SETUP = r"""', 1)[1].split('"""', 1)[0]
    block = setup[setup.index('if [ -n "${GT_SYCL_URL:-}" ]'):setup.index("elif ")]
    return block.replace("/opt/sycl", str(prefix / "sycl")).replace(
        "/tmp/sycl.tar.gz", str(prefix / "dl.tar.gz")) + "fi\n"


@pytest.mark.skipif(not (shutil.which("sha256sum") and shutil.which("curl")),
                    reason="needs sha256sum and curl")
def test_setup_installs_the_sycl_tarball_and_checks_its_sha256(tmp_path):
    src = tmp_path / "src"
    (src / "bin").mkdir(parents=True)
    (src / "bin" / "clang++").write_text("#!/bin/sh\n")
    (src / "BUILD-INFO.txt").write_text("intel/llvm commit: test\n")
    tar = tmp_path / "t.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for f in src.rglob("*"):
            tf.add(f, arcname=str(f.relative_to(src)))
    good = hashlib.sha256(tar.read_bytes()).hexdigest()
    script = _setup_url_block(tmp_path)

    def run(sha):
        env = {**os.environ, "GT_SYCL_URL": tar.as_uri(), "GT_SYCL_SHA256": sha}
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

    bad = run("0" * 64)
    assert bad.returncode != 0 and not (tmp_path / "sycl" / "bin" / "clang++").exists()
    ok = run(good)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert (tmp_path / "sycl" / "bin" / "clang++").exists() and "intel/llvm commit" in ok.stdout


def _hip_headers_setup(tmp_path, *, headers=False, runtime="7.14.0-3",
                       madison=("7.14.1-0", "7.14.0-3")):
    """Run the agent's HIP-headers step against a fake ROCm core-7.14 install and stub apt."""
    setup = remote.POD_AGENT.split('SETUP = r"""', 1)[1].split('"""', 1)[0]
    block = setup[setup.index("# HIP headers for the native"):setup.index("fi  # end HIP headers")]
    etc = tmp_path / "etc"
    (etc / "apt" / "sources.list.d").mkdir(parents=True)
    (etc / "os-release").write_text('ID=ubuntu\nVERSION_ID="24.04"\n')
    script = (block.replace("/etc/os-release", str(etc / "os-release"))
              .replace("/etc/apt", str(etc / "apt"))
              .replace("/opt/rocm/.info", str(tmp_path / "none")) + "fi\n")
    root = tmp_path / "opt" / "rocm" / "core-7.14"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "hipcc").write_text("#!/bin/sh\n")
    if headers:
        (root / "include" / "hip").mkdir(parents=True)
        (root / "include" / "hip" / "hip_runtime.h").write_text("")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "hipcc").symlink_to(root / "bin" / "hipcc")  # like /opt/rocm/bin -> core-7.14
    log = tmp_path / "calls.log"
    rows = "".join(f"echo 'amdrocm-runtime-dev7.14 | {v} | https://repo.amd.com x'; "
                   for v in madison)
    stubs = {
        "dpkg-query": f'[ -n "{runtime}" ] && printf %s "{runtime}" || exit 1',
        "apt-cache": f'[ "$1" = madison ] && {{ {rows}:; }}',
        "apt-get": '[ "$1" = download ] && touch "${2%%=*}_${2#*=}_amd64.deb"; :',
        "dpkg-deb": ('d="$3/opt/rocm/core-7.14/include/hip"; mkdir -p "$d" && '
                     'touch "$d/hip_runtime.h"'),
        "curl": 'touch "$5"',  # -fsSL --retry 3 -o FILE URL
    }
    for name, body in stubs.items():
        f = bin_ / name
        f.write_text(f'#!/bin/bash\necho "{name} $*" >> "{log}"\n{body}\n')
        f.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"}
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    calls = log.read_text() if log.exists() else ""
    return out, calls, root, etc


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_setup_unpacks_hip_headers_matching_the_installed_runtime(tmp_path):
    out, calls, root, etc = _hip_headers_setup(tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert (root / "include" / "hip" / "hip_runtime.h").exists()
    # The exact runtime version, downloaded rather than installed (no 2 GB amdrocm-llvm-dev).
    assert "apt-get download amdrocm-runtime-dev7.14=7.14.0-3" in calls
    assert "apt-get install" not in calls
    assert "unpacked amdrocm-runtime-dev7.14=7.14.0-3" in out.stdout
    src = (etc / "apt" / "sources.list.d" / "gt-amdrocm.list").read_text()
    assert "packages-multi-arch/ubuntu2404 stable main" in src and "gt-amdrocm.asc" in src
    assert (etc / "apt" / "keyrings" / "gt-amdrocm.asc").exists()


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_setup_leaves_existing_hip_headers_alone(tmp_path):
    out, calls, _, _ = _hip_headers_setup(tmp_path, headers=True)
    assert out.returncode == 0 and calls == ""
    assert "HIP headers: " in out.stdout and "missing" not in out.stdout


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_setup_says_so_when_no_matching_hip_headers_exist(tmp_path):
    out, calls, root, _ = _hip_headers_setup(tmp_path, madison=("7.14.1-0",))
    assert "NOT INSTALLED: no amdrocm-runtime-dev7.14 matching the installed runtime (7.14.0-3)" \
        in out.stdout
    assert "still missing" in out.stdout and "apt-get download" not in calls
    assert not (root / "include").exists()
    out, calls, _, _ = _hip_headers_setup(tmp_path / "u", runtime="")
    assert "version unknown" in out.stdout and "apt-get download" not in calls


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


class KeyLeakProvider(SlowDeleteProvider):
    """Machine deletion works, but the provider's post-run cleanup cannot remove its key."""

    def __init__(self):
        super().__init__()
        self.cleanup_calls = []

    def cleanup(self, poll_s=5):
        self.cleanup_calls.append(list(self.deleted))
        return False


def test_cleanup_runs_after_deletion_and_a_leftover_key_is_reported(tmp_path):
    prov = KeyLeakProvider()
    logs = []
    r = remote.remote_test(str(REPO / "examples/saxpy/task.json"), provider=prov, poll_s=0.01,
                           max_stuck=1, max_minutes=5, into=str(tmp_path), log=logs.append)
    assert prov.cleanup_calls == [["m1"]]  # after the machine was deleted
    assert any("SSH key" in lim for lim in r.limitations), r.limitations
    assert any("SSH key NOT removed" in m for m in logs)
