"""``remote_test``: run ``gpu_test`` on a rented Runpod GPU (NVIDIA or AMD), then delete it.

Flow (no SSH; everything goes through Runpod's HTTPS proxy):

1. ``make_bundle`` packs the task and this package into one ``.sh``.
2. A pod is created through the Runpod REST API (``https://api.runpod.io/v2``), with a small
   stdlib HTTP agent (``POD_AGENT`` below) as its start command. The agent installs a SYCL
   compiler if the image has none (an intel/llvm release, which carries the CUDA and HIP
   adapters) and listens on port 8000, guarded by a random per-run token.
3. The bundle is uploaded to ``https://<pod>-8000.proxy.runpod.net/bundle``; the agent runs it
   and serves ``/status`` and ``/results``.
4. The results archive is downloaded and imported with ``ingest_results``.
5. The pod is deleted in a ``finally`` block, whatever happened, and the delete is verified.

Spending limits: ``max_minutes`` (wall clock, from create to delete) and ``max_usd`` (checked
against the GPU's catalog price before creating, and enforced as a time limit).
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import shlex
import shutil
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .bundle import ingest_results, make_bundle
from .result import ToolResult
from .toolchain import run_command

API = "https://api.runpod.io/v2"
# Runpod's proxy sits behind Cloudflare, which rejects Python's default "Python-urllib" agent
# (HTTP 403, "error code: 1010"), so every request names itself.
USER_AGENT = "oneapi-agent-toolkit (+https://github.com/checocadena/oneapi-agent-toolkit)"
# Tried in order until one is in stock: cheap cards first (Runpod catalog ids).
DEFAULT_GPUS = {
    "nvidia": ["NVIDIA RTX A4500", "NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX A4000",
               "NVIDIA RTX A5000", "NVIDIA RTX 4000 Ada Generation", "NVIDIA L4",
               "NVIDIA GeForce RTX 4090", "NVIDIA A40", "NVIDIA RTX A6000"],
    "amd": ["AMD Instinct MI300X OAM"],
}
DEFAULT_IMAGE = {
    "nvidia": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",  # devel: has nvcc
    "amd": "runpod/pytorch:2.4.0-py3.10-rocm6.1.0-ubuntu22.04",  # has hipcc, hipify-perl
}

# Runs inside the pod. Standard library only; started as `python3 /gt_agent.py`.
POD_AGENT = r'''
import http.server, json, os, pathlib, subprocess, threading, time
TOKEN = os.environ["GT_TOKEN"]
W = pathlib.Path(os.environ.get("GT_DIR", "/workspace/gputest")); W.mkdir(parents=True, exist_ok=True)
S = {"phase": "setup", "rc": None, "started": time.time()}
SETUP = r"""
set -x
export DEBIAN_FRONTEND=noninteractive
command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl; }
python3 -c 'import numpy' 2>/dev/null || { apt-get update -qq && apt-get install -y -qq python3-numpy; } \
  || python3 -m pip install -q --break-system-packages numpy || python3 -m pip install -q numpy
if ! command -v icpx >/dev/null && [ ! -x /opt/sycl/bin/clang++ ]; then
  URL="$(curl -fsSL 'https://api.github.com/repos/intel/llvm/releases?per_page=30' | python3 -c '
import json,sys
for r in json.load(sys.stdin):
    for a in r.get("assets", []):
        n = a["name"]
        if n.endswith(".tar.gz") and "linux" in n and "windows" not in n:
            print(a["browser_download_url"]); sys.exit()')"
  echo "intel/llvm: $URL"
  mkdir -p /opt/sycl && curl -fsSL "$URL" | tar xz -C /opt/sycl
fi
export PATH="/opt/sycl/bin:/usr/local/cuda/bin:/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/sycl/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi || rocm-smi || true
sycl-ls
"""
ENV = dict(os.environ)
ENV["PATH"] = "/opt/sycl/bin:/usr/local/cuda/bin:/opt/rocm/bin:" + ENV.get("PATH", "")
ENV["LD_LIBRARY_PATH"] = "/opt/sycl/lib:/opt/rocm/lib:" + ENV.get("LD_LIBRARY_PATH", "")
ENV.setdefault("SYCL_CXX", "/opt/sycl/bin/clang++")  # dropped after setup if not installed

def setup():
    if os.environ.get("GT_NO_SETUP"):  # tests: use whatever toolchain is already on PATH
        S["phase"], S["setup_rc"] = "ready", 0
        return
    with open(W / "setup.log", "w") as log:
        rc = subprocess.call(["bash", "-c", SETUP], stdout=log, stderr=subprocess.STDOUT, env=ENV)
    if not os.path.exists("/opt/sycl/bin/clang++"):
        ENV.pop("SYCL_CXX", None)
    S["phase"] = "ready" if rc == 0 else "setup_failed"
    S["setup_rc"] = rc

def run(args):
    while S["phase"] == "setup":
        time.sleep(2)
    # Run even if setup failed: gpu_test then reports what is missing (toolchain_missing) in a
    # results archive, which is more useful than no archive at all.
    S["phase"] = "running"
    with open(W / "run.log", "w") as log:
        S["rc"] = subprocess.call(["bash", str(W / "bundle.sh"), *args], cwd=W, stdout=log,
                                  stderr=subprocess.STDOUT, env=ENV)
    S["phase"] = "done"

def tail(name, n=4000):
    p = W / name
    return p.read_text(errors="replace")[-n:] if p.exists() else ""

class H(http.server.BaseHTTPRequestHandler):
    def _ok(self):
        if self.headers.get("X-Token") != TOKEN:
            self.send_response(403); self.end_headers(); return False
        return True
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if not self._ok(): return
        if self.path == "/status":
            body = dict(S, setup_tail=tail("setup.log"), run_tail=tail("run.log"))
            self._send(200, json.dumps(body).encode())
        elif self.path == "/results":
            files = sorted(W.glob("gputest-results-*.tar.gz"))
            if not files: self._send(404, b"{}"); return
            self._send(200, files[-1].read_bytes(), "application/gzip")
        else:
            self._send(404, b"{}")
    def do_PUT(self):
        if not self._ok(): return
        if self.path.split("?")[0] != "/bundle" or S["phase"] not in ("setup", "ready"):
            self._send(409, b'{"error": "not accepting a bundle now"}'); return
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > 60 * 1024 * 1024:
            self._send(413, b"{}"); return
        (W / "bundle.sh").write_bytes(self.rfile.read(n))
        args = json.loads(self.headers.get("X-Args", "[]"))
        S["phase"] = S["phase"] if S["phase"] == "setup" else "ready"
        threading.Thread(target=run, args=(args,), daemon=True).start()
        self._send(202, b'{"accepted": true}')
    def log_message(self, *a):
        pass

threading.Thread(target=setup, daemon=True).start()
srv = http.server.ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("GT_PORT", "8000"))), H)
if os.environ.get("GT_TLS_CERT"):  # direct-IP providers: TLS with a per-run cert the client pins
    import base64, ssl
    (W / "tls.crt").write_bytes(base64.b64decode(os.environ["GT_TLS_CERT"]))
    (W / "tls.key").write_bytes(base64.b64decode(os.environ["GT_TLS_KEY"]))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(W / "tls.crt", W / "tls.key")
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
srv.serve_forever()
'''


class RunpodAPI:
    """Minimal Runpod REST v2 client (pods + GPU catalog)."""

    def __init__(self, api_key: str, base: str = API) -> None:
        self.key, self.base = api_key, base

    def _req(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
            "User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else None

    def gpu_price(self, gpu_id: str, cloud: str) -> float | None:
        try:
            data = self._req("GET", "/catalog/gpus")
        except (urllib.error.URLError, ValueError):
            return None
        for g in data.get("gpus", data) if isinstance(data, dict) else data:
            if g.get("id") == gpu_id:
                price = (g.get("price") or {}).get(cloud.lower())
                return float(price) if price else None
        return None

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", "/pods", body)

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        try:
            return self._req("GET", f"/pods/{pod_id}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def delete_pod(self, pod_id: str) -> None:
        try:
            self._req("DELETE", f"/pods/{pod_id}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise


def make_tls_pair(directory: Path) -> tuple[Path, Path] | None:
    """A throwaway self-signed cert + key for one run (None if openssl is unavailable)."""
    crt, key = directory / "agent.crt", directory / "agent.key"
    res = run_command(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt",
                       "ec_paramgen_curve:prime256v1", "-nodes", "-days", "2",
                       "-subj", "/CN=oneapi-gputest-agent", "-keyout", str(key), "-out", str(crt)],
                      timeout=60)
    return (crt, key) if res.returncode == 0 and crt.exists() else None


def pinned_context(cert: Path) -> ssl.SSLContext:
    """Trust exactly this certificate (it is self-signed for a bare IP, so no hostname check)."""
    ctx = ssl.create_default_context(cafile=str(cert))
    ctx.check_hostname = False
    return ctx


def _agent_call(url: str, token: str, method: str = "GET", data: bytes | None = None,
                headers: dict[str, str] | None = None, timeout: float = 60,
                context: ssl.SSLContext | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"X-Token": token, "User-Agent": USER_AGENT,
                                          **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, b""


class Unavailable(Exception):
    """The provider could not create this GPU right now (no stock, quota): try the next one."""


START_CMD = 'echo "$GT_AGENT" | base64 -d > /gt_agent.py && exec python3 /gt_agent.py'


class Provider:
    """What remote_test needs from a GPU rental service. Subclasses implement each method."""

    name = "provider"
    key_env = ""
    tls = False  # True for providers reached on a bare IP: the agent then serves pinned TLS

    def candidates(self, vendor: str, gpu_type: str | None) -> list[str]:
        raise NotImplementedError

    def price(self, gpu: str) -> float | None:
        """USD per hour, or None if unknown."""
        raise NotImplementedError

    def create(self, gpu: str, *, vendor: str, image: str | None, env: dict[str, str]) -> str:
        """Start a machine that runs START_CMD with ``env``; return its id. Raise Unavailable."""
        raise NotImplementedError

    def agent_url(self, machine_id: str) -> str | None:
        """Base URL of the agent on port 8000, or None while it is not known yet."""
        raise NotImplementedError

    def delete(self, machine_id: str) -> None:
        raise NotImplementedError

    def exists(self, machine_id: str) -> bool:
        raise NotImplementedError


class RunpodProvider(Provider):
    name, key_env = "runpod", "RUNPOD_API_KEY"

    def __init__(self, api: RunpodAPI, cloud: str = "SECURE") -> None:
        self.api, self.cloud = api, cloud

    def candidates(self, vendor: str, gpu_type: str | None) -> list[str]:
        return [gpu_type] if gpu_type else DEFAULT_GPUS[vendor]

    def price(self, gpu: str) -> float | None:
        return self.api.gpu_price(gpu, self.cloud)

    def create(self, gpu: str, *, vendor: str, image: str | None, env: dict[str, str]) -> str:
        body: dict[str, Any] = {
            "name": f"oneapi-gputest-{secrets.token_hex(3)}", "cloud": self.cloud,
            "gpu": {"id": gpu, "count": 1}, "image": image or DEFAULT_IMAGE[vendor],
            "disk": 60, "ports": ["8000/http"], "env": env,
            "entrypoint": ["bash", "-c"], "cmd": [START_CMD],
        }
        if vendor == "nvidia":
            body["gpu"]["minCudaVersion"] = "12.4"
        try:
            return str(self.api.create_pod(body)["id"])
        except urllib.error.HTTPError as e:
            raise Unavailable(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e

    def agent_url(self, machine_id: str) -> str | None:
        return f"https://{machine_id}-8000.proxy.runpod.net"

    def delete(self, machine_id: str) -> None:
        self.api.delete_pod(machine_id)

    def exists(self, machine_id: str) -> bool:
        return self.api.get_pod(machine_id) is not None


class _JSONAPI:
    """Bearer-token JSON client shared by the Vast and Vultr providers."""

    def __init__(self, base: str, key: str) -> None:
        self.base, self.key = base, key

    def req(self, method: str, path: str, body: Any = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
            "Accept": "application/json", "User-Agent": USER_AGENT})
        with urllib.request.urlopen(r, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None


class VastProvider(Provider):
    """Vast.ai marketplace (docs.vast.ai; endpoints as used by the official vast CLI).

    Offers are searched per GPU name (cheapest verified, rentable, reliable host with a direct
    port), rented with ``PUT /api/v0/asks/{offer}/`` in ``args`` mode with ``-p 8000:8000``,
    and reached on ``public_ipaddr`` + the host port mapped to 8000 (plain TCP, so the agent
    serves TLS with a per-run pinned certificate). As of 2026-10-15 Vast lists no AMD offers.
    """

    name, key_env, tls = "vast", "VAST_API_KEY", True
    GPUS = {"nvidia": ["RTX A4000", "RTX 3090", "RTX A5000", "RTX 4090", "RTX 4080"], "amd": []}
    IMAGE = {"nvidia": "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"}  # python3 + nvcc

    def __init__(self, key: str, api: _JSONAPI | None = None) -> None:
        self.api = api or _JSONAPI("https://console.vast.ai/api/v0", key)

    def candidates(self, vendor: str, gpu_type: str | None) -> list[str]:
        return [gpu_type] if gpu_type else self.GPUS.get(vendor, [])

    def _offers(self, gpu: str) -> list[dict[str, Any]]:
        q = {"gpu_name": {"in": [gpu]}, "num_gpus": {"eq": 1}, "rentable": {"eq": True},
             "rented": {"eq": False}, "verified": {"eq": True}, "reliability": {"gte": 0.98},
             "direct_port_count": {"gte": 1}, "type": "ondemand",
             "order": [["dph_total", "asc"]], "limit": 5}
        try:
            return list((self.api.req("POST", "/bundles/", q) or {}).get("offers") or [])
        except (urllib.error.URLError, ValueError):
            return []

    def price(self, gpu: str) -> float | None:
        offers = self._offers(gpu)
        return float(offers[0]["dph_total"]) if offers else None

    def create(self, gpu: str, *, vendor: str, image: str | None, env: dict[str, str]) -> str:
        offers = self._offers(gpu)
        if not offers:
            raise Unavailable("no rentable offer")
        body = {"client_id": "me", "image": image or self.IMAGE.get(vendor, self.IMAGE["nvidia"]),
                "disk": 40, "label": f"oneapi-gputest-{secrets.token_hex(3)}",
                "env": {"-p 8000:8000": "1", **env}, "runtype": "args",
                "args": ["bash", "-c", START_CMD]}
        for offer in offers:  # the cheapest can be taken between search and rent
            try:
                res = self.api.req("PUT", f"/asks/{offer['id']}/", body)
            except urllib.error.HTTPError as e:
                if e.code in (404, 410):
                    continue
                raise Unavailable(f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}") from e
            if res and res.get("success") and res.get("new_contract"):
                return str(res["new_contract"])
        raise Unavailable("every offer was taken before it could be rented")

    def _instance(self, mid: str) -> dict[str, Any] | None:
        try:
            data = self.api.req("GET", f"/instances/{mid}/")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        inst = (data or {}).get("instances")
        return inst if isinstance(inst, dict) and inst else None

    def agent_url(self, machine_id: str) -> str | None:
        inst = self._instance(machine_id)
        if not inst:
            return None
        if inst.get("actual_status") in ("exited", "unknown", "offline"):
            raise RuntimeError(f"vast instance is {inst.get('actual_status')}: "
                               f"{inst.get('status_msg', '')}"[:300])
        mapped = (inst.get("ports") or {}).get("8000/tcp") or []
        ip = inst.get("public_ipaddr")
        if inst.get("actual_status") == "running" and mapped and ip:
            return f"https://{ip.strip()}:{mapped[0]['HostPort']}"
        return None

    def delete(self, machine_id: str) -> None:
        try:
            self.api.req("DELETE", f"/instances/{machine_id}/", {})
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise

    def exists(self, machine_id: str) -> bool:
        return self._instance(machine_id) is not None


def cloud_init_script(env: dict[str, str]) -> str:
    """A cloud-init user-data shell script that starts the agent with ``env`` on port 8000."""
    exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items())
    return ("#!/bin/bash\n" + exports + "\n"
            "(command -v ufw >/dev/null && ufw allow 8000/tcp) || true\n"
            "mkdir -p /workspace && cd /workspace\n"
            "nohup bash -c " + shlex.quote(START_CMD) + " > /var/log/gputest-agent.log 2>&1 &\n")


class VultrProvider(Provider):
    """Vultr Cloud GPU (api.vultr.com/v2; fields as in the official govultr SDK / Terraform).

    A VM is created with ``POST /v2/instances`` (Ubuntu 24.04; Vultr's GPU plans boot GPU-enabled
    images with drivers) and a cloud-init ``user_data`` script that starts the agent; it is
    reached on ``main_ip:8000`` with pinned TLS. As of 2026-10-15 Vultr's AMD Instinct plans
    are listed but not deployable on demand, so only NVIDIA plans are candidates.
    """

    name, key_env, tls = "vultr", "VULTR_API_KEY", True
    GPUS = {"nvidia": ["vcg-a16-6c-60g-16vram", "vcg-a16-12c-120g-32vram",
                       "vcg-a40-24c-120g-48vram"], "amd": []}
    UBUNTU_2404 = 2284

    def __init__(self, key: str, api: _JSONAPI | None = None) -> None:
        self.api = api or _JSONAPI("https://api.vultr.com/v2", key)
        self._plans: dict[str, dict[str, Any]] | None = None

    def candidates(self, vendor: str, gpu_type: str | None) -> list[str]:
        return [gpu_type] if gpu_type else self.GPUS.get(vendor, [])

    def _plan(self, plan_id: str) -> dict[str, Any] | None:
        if self._plans is None:
            self._plans = {}
            for kind in ("vdm", "vcg"):
                try:
                    data = self.api.req("GET", f"/plans?type={kind}&per_page=500") or {}
                except (urllib.error.URLError, ValueError):
                    continue
                self._plans.update({pl["id"]: pl for pl in data.get("plans", [])})
        return self._plans.get(plan_id)

    def price(self, gpu: str) -> float | None:
        plan = self._plan(gpu)
        return float(plan["hourly_cost"]) if plan and plan.get("hourly_cost") else None

    def create(self, gpu: str, *, vendor: str, image: str | None, env: dict[str, str]) -> str:
        plan = self._plan(gpu)
        regions = (plan or {}).get("locations") or []
        if not regions:
            raise Unavailable("plan has no deployable locations right now")
        script = cloud_init_script(env)
        body = {"region": regions[0], "plan": gpu, "os_id": self.UBUNTU_2404,
                "label": f"oneapi-gputest-{secrets.token_hex(3)}",
                "user_data": base64.b64encode(script.encode()).decode()}
        try:
            res = self.api.req("POST", "/instances", body)
        except urllib.error.HTTPError as e:
            raise Unavailable(f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}") from e
        return str(res["instance"]["id"])

    def _instance(self, mid: str) -> dict[str, Any] | None:
        try:
            return (self.api.req("GET", f"/instances/{mid}") or {}).get("instance")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def agent_url(self, machine_id: str) -> str | None:
        inst = self._instance(machine_id)
        ip = (inst or {}).get("main_ip")
        if inst and inst.get("status") == "active" and ip and ip != "0.0.0.0":
            return f"https://{ip}:8000"
        return None

    def delete(self, machine_id: str) -> None:
        try:
            self.api.req("DELETE", f"/instances/{machine_id}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise

    def exists(self, machine_id: str) -> bool:
        return self._instance(machine_id) is not None


class DigitalOceanProvider(Provider):
    """DigitalOcean GPU Droplets (api.digitalocean.com/v2): the cheapest self-serve single AMD
    Instinct MI300X found (``gpu-mi300x1-192gb``, ~$2.59/h on 2026-10-15), booted from the
    ``gpu-amd-base`` image (Ubuntu + ROCm) with a cloud-init script that starts the agent.

    Candidates come from the live ``/v2/sizes`` list: available single-GPU sizes of the vendor,
    cheapest first. AMD GPU access may first need a support request on new accounts; the API's
    error is reported as the create failure. Reached on the droplet's public IPv4, port 8000,
    with pinned TLS.
    """

    name, key_env, tls = "digitalocean", "OCEAN_API_KEY", True
    IMAGE = {"amd": "gpu-amd-base"}
    AMD_MARKERS = ("mi300", "mi325", "mi350", "mi355")

    def __init__(self, key: str, api: _JSONAPI | None = None) -> None:
        self.api = api or _JSONAPI("https://api.digitalocean.com/v2", key)
        self._sizes: dict[str, dict[str, Any]] | None = None
        self._temp_key_id: int | None = None  # throwaway SSH key registered by this run

    def ssh_keys(self) -> list[Any]:
        """SSH keys for the droplet: DigitalOcean's GPU images refuse root passwords.

        ONEAPI_DO_SSH_KEYS (comma-separated ids/fingerprints) wins; otherwise the account's
        existing keys; otherwise a throwaway ed25519 key made and registered for this run only
        (deleted again with the droplet). The agent never needs SSH; the key just satisfies
        the image.
        """
        env = os.environ.get("ONEAPI_DO_SSH_KEYS", "").strip()
        if env:
            return [k.strip() for k in env.split(",") if k.strip()]
        try:
            existing = (self.api.req("GET", "/account/keys?per_page=200") or {}).get("ssh_keys")
        except (urllib.error.URLError, ValueError):
            existing = None
        if existing:
            return [k["id"] for k in existing]
        tmp = Path(tempfile.mkdtemp(prefix="do-key-"))
        res = run_command(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C",
                           "oneapi-gputest-temp", "-f", str(tmp / "k")], timeout=60)
        if res.returncode != 0:
            raise Unavailable("no SSH key on the account and ssh-keygen failed: "
                              + (res.stderr or res.stdout)[-200:])
        try:
            key = self.api.req("POST", "/account/keys", {
                "name": f"oneapi-gputest-temp-{secrets.token_hex(3)}",
                "public_key": (tmp / "k.pub").read_text().strip()})["ssh_key"]
        except urllib.error.HTTPError as e:
            raise Unavailable(f"could not register a temporary SSH key: HTTP {e.code}: "
                              f"{e.read().decode(errors='replace')[:300]}") from e
        finally:
            shutil.rmtree(tmp, ignore_errors=True)  # the private half is never needed
        self._temp_key_id = int(key["id"])
        return [key["id"]]

    def sizes(self) -> dict[str, dict[str, Any]]:
        if self._sizes is None:
            try:
                data = self.api.req("GET", "/sizes?per_page=200") or {}
            except (urllib.error.URLError, ValueError):
                data = {}
            self._sizes = {z["slug"]: z for z in data.get("sizes", [])}
        return self._sizes

    def candidates(self, vendor: str, gpu_type: str | None) -> list[str]:
        if gpu_type:
            return [gpu_type]
        out = []
        for slug, z in self.sizes().items():
            amd = any(m in slug for m in self.AMD_MARKERS)
            single = "x1-" in slug
            if slug.startswith("gpu-") and single and z.get("available") and z.get("regions") \
                    and amd == (vendor == "amd"):
                out.append(slug)
        return sorted(out, key=lambda sl: float(self.sizes()[sl].get("price_hourly") or 1e9))

    def price(self, gpu: str) -> float | None:
        z = self.sizes().get(gpu)
        return float(z["price_hourly"]) if z and z.get("price_hourly") else None

    def create(self, gpu: str, *, vendor: str, image: str | None, env: dict[str, str]) -> str:
        z = self.sizes().get(gpu) or {}
        regions = z.get("regions") or []
        if not regions:
            raise Unavailable("size has no regions with capacity right now")
        img = image or self.IMAGE.get(vendor)
        if not img:
            raise Unavailable(f"no default GPU image for {vendor}; pass --image")
        body = {"name": f"oneapi-gputest-{secrets.token_hex(3)}", "region": regions[0],
                "size": gpu, "image": img, "user_data": cloud_init_script(env),
                "ssh_keys": self.ssh_keys(), "tags": ["oneapi-gputest"]}
        try:
            res = self.api.req("POST", "/droplets", body)
        except urllib.error.HTTPError as e:
            raise Unavailable(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
        return str(res["droplet"]["id"])

    def _droplet(self, mid: str) -> dict[str, Any] | None:
        try:
            return (self.api.req("GET", f"/droplets/{mid}") or {}).get("droplet")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def agent_url(self, machine_id: str) -> str | None:
        d = self._droplet(machine_id)
        if not d or d.get("status") != "active":
            return None
        for net in (d.get("networks") or {}).get("v4", []):
            if net.get("type") == "public" and net.get("ip_address"):
                return f"https://{net['ip_address']}:8000"
        return None

    def delete(self, machine_id: str) -> None:
        try:
            self.api.req("DELETE", f"/droplets/{machine_id}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        self.cleanup()

    def cleanup(self) -> None:
        """Remove the throwaway SSH key, if this run registered one."""
        if self._temp_key_id is not None:
            try:
                self.api.req("DELETE", f"/account/keys/{self._temp_key_id}")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
            self._temp_key_id = None

    def exists(self, machine_id: str) -> bool:
        return self._droplet(machine_id) is not None


def make_provider(name: str, *, cloud: str = "SECURE", api_key: str | None = None) -> Provider:
    """Provider by name, keyed from its environment variable (or ``api_key``)."""
    key = api_key or os.environ.get(PROVIDER_KEY_ENV.get(name, ""), "")
    if name == "runpod":
        return RunpodProvider(RunpodAPI(key), cloud)
    if name == "vast":
        return VastProvider(key)
    if name == "vultr":
        return VultrProvider(key)
    if name == "digitalocean":
        return DigitalOceanProvider(key or os.environ.get("DIGITALOCEAN_TOKEN", ""))
    raise ValueError(f"unknown provider {name!r}")


PROVIDER_KEY_ENV = {"runpod": "RUNPOD_API_KEY", "vast": "VAST_API_KEY", "vultr": "VULTR_API_KEY",
                    "digitalocean": "OCEAN_API_KEY"}
# Also accepted: the name doctl uses.
PROVIDER_KEY_ALIASES = {"digitalocean": ("DIGITALOCEAN_TOKEN",)}


def remote_test(task_file: str, *, vendor: str = "nvidia", gpu_type: str | None = None,
                provider: str | Provider = "runpod", cloud: str = "SECURE",
                image: str | None = None, max_minutes: float = 45, max_usd: float = 2.0,
                into: str = "docs/gpu-run", keep_pod: bool = False, api_key: str | None = None,
                api: RunpodAPI | None = None, log: Callable[[str], None] = print,
                poll_s: float = 10, agent_url: Callable[[str], str] | None = None) -> ToolResult:
    """Rent a GPU, run ``gpu_test`` there through a bundle, import the results, delete it."""
    if isinstance(provider, Provider):
        prov = provider
    elif api is not None:  # tests / callers that bring their own Runpod client
        prov = RunpodProvider(api, cloud)
    else:
        env_name = PROVIDER_KEY_ENV.get(provider, "")
        if not env_name:
            return ToolResult("remote_test", "error", f"unknown provider {provider!r}; "
                              f"choose from {sorted(PROVIDER_KEY_ENV)}")
        if not (api_key or os.environ.get(env_name)
                or any(os.environ.get(a) for a in PROVIDER_KEY_ALIASES.get(provider, ()))):
            return ToolResult("remote_test", "toolchain_missing", f"{env_name} is not set", {},
                              [f"Create an API key in the {provider} console and export "
                               f"{env_name}."])
        prov = make_provider(provider, cloud=cloud, api_key=api_key)
    candidates = prov.candidates(vendor, gpu_type)
    affordable: list[tuple[str, float | None, float]] = []
    for g in candidates:
        p = prov.price(g)
        budget_minutes = max_usd / p * 60 if p else float("inf")
        if budget_minutes >= 10:  # a GPU the budget can't keep for 10 minutes is skipped
            affordable.append((g, p, min(max_minutes, budget_minutes)))
    if not affordable:
        return ToolResult("remote_test", "error",
                          f"no candidate GPU fits max_usd ${max_usd} for 10+ minutes",
                          {"candidates": candidates, "provider": prov.name})
    tmp = Path(tempfile.mkdtemp(prefix="remote-test-"))
    b = make_bundle(task_file, str(tmp / "bundle.sh"))
    if not b.ok:
        return b
    token = secrets.token_urlsafe(24)
    env = {"GT_TOKEN": token, "GT_AGENT": base64.b64encode(POD_AGENT.encode()).decode()}
    ctx: ssl.SSLContext | None = None
    tls_note: list[str] = []
    if prov.tls:
        pair = make_tls_pair(tmp)
        if pair:
            env["GT_TLS_CERT"] = base64.b64encode(pair[0].read_bytes()).decode()
            env["GT_TLS_KEY"] = base64.b64encode(pair[1].read_bytes()).decode()
            ctx = pinned_context(pair[0])
        else:
            return ToolResult("remote_test", "toolchain_missing",
                              f"openssl is needed to protect the connection to a {prov.name} "
                              "machine (it is reached on a bare IP)", {"provider": prov.name})
    t0 = time.time()
    mid: str | None = None
    ev: dict[str, Any] = {"provider": prov.name, "create_attempts": []}
    status: dict[str, Any] = {}
    result: ToolResult
    gpu, price = affordable[0][0], affordable[0][1]
    try:
        for cand in affordable:
            try:
                mid = prov.create(cand[0], vendor=vendor, image=image, env=env)
            except Unavailable as e:
                ev["create_attempts"].append({"gpu": cand[0], "error": str(e)})
                log(f"{cand[0]}: not created ({str(e)[:400]}); trying the next GPU")
                continue
            gpu, price, max_minutes = cand
            break
        if mid is None:
            getattr(prov, "cleanup", lambda: None)()
            return ToolResult("remote_test", "error",
                              f"no candidate GPU could be created on {prov.name} "
                              f"({len(affordable)} tried)", ev,
                              ["Usually means no stock right now; retry later or pass --gpu."])
        ev.update(gpu=gpu, price_per_hour=price, max_minutes=round(max_minutes, 1),
                  machine_id=mid)
        deadline = t0 + max_minutes * 60
        log(f"created {prov.name} machine {mid} ({gpu}, ${price}/h)")
        # Wait until the agent's address is known and it answers, then upload the bundle.
        url: str | None = None
        seen_codes: set[int] = set()
        while time.time() < deadline:
            url = agent_url(mid) if agent_url else prov.agent_url(mid)
            if url:
                code, _ = _agent_call(url + "/status", token, timeout=20, context=ctx)
                if code == 200:
                    break
                if code not in seen_codes:  # 0 = not reachable yet; 502/404 = still starting
                    seen_codes.add(code)
                    log(f"waiting for the agent (HTTP {code or 'no answer'})")
            time.sleep(poll_s)
        else:
            raise TimeoutError("the machine's agent never answered before the time limit")
        assert url is not None
        ev["agent_url"] = url
        code, resp = _agent_call(url + "/bundle", token, "PUT", (tmp / "bundle.sh").read_bytes(),
                                 {"X-Args": json.dumps(["--vendor", vendor])}, timeout=300,
                                 context=ctx)
        if code != 202:
            raise RuntimeError(f"bundle upload failed: HTTP {code} {resp[:200]!r}")
        log("bundle uploaded; waiting for the run")
        last = ""
        while time.time() < deadline:
            code, raw = _agent_call(url + "/status", token, timeout=30, context=ctx)
            if code == 200:
                status = json.loads(raw)
                if status.get("phase") != last:
                    last = status.get("phase", "")
                    log(f"agent phase: {last}")
                if status.get("phase") == "done":
                    break
            time.sleep(poll_s)
        else:
            raise TimeoutError(f"run did not finish within {max_minutes:.0f} min")
        code, archive = _agent_call(url + "/results", token, timeout=120, context=ctx)
        if code != 200:
            raise RuntimeError("run finished but produced no results archive. Run log tail: "
                               f"{status.get('run_tail', '')[-2000:]} || setup log tail: "
                               f"{status.get('setup_tail', '')[-800:]}")
        (tmp / "results.tar.gz").write_bytes(archive)
        ing = ingest_results(str(tmp / "results.tar.gz"), into)
        ev["ingest"] = ing.evidence
        remote_status = (ing.evidence.get("remote_result") or {}).get("status") or ing.status
        result = ToolResult("remote_test", remote_status, f"{gpu}: {ing.summary}", ev,
                            [*ing.limitations, *tls_note])
    except (TimeoutError, RuntimeError, urllib.error.URLError, KeyError, ValueError) as exc:
        ev["agent_status"] = {k: status.get(k) for k in ("phase", "rc", "setup_rc")}
        ev["setup_tail"] = status.get("setup_tail", "")[-3000:]
        ev["run_tail"] = status.get("run_tail", "")[-3000:]
        result = ToolResult("remote_test", "error", f"{type(exc).__name__}: {exc}"[:600], ev)
    finally:
        if mid and not keep_pod:
            prov.delete(mid)
            gone = not prov.exists(mid)
            ev["pod_deleted"] = gone
            log(f"deleted {prov.name} machine {mid}: {'confirmed' if gone else 'NOT CONFIRMED'}")
    minutes = (time.time() - t0) / 60
    ev["minutes"] = round(minutes, 1)
    if price is not None:
        ev["est_cost_usd"] = round(price * minutes / 60, 3)
    if mid and not keep_pod and not ev.get("pod_deleted"):
        result.limitations.append(f"{prov.name} machine {mid} deletion could not be confirmed: "
                                  "check the provider's console.")
    return result
