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
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .bundle import ingest_results, make_bundle
from .result import ToolResult

API = "https://api.runpod.io/v2"
DEFAULT_GPU = {"nvidia": "NVIDIA RTX A4500", "amd": "AMD Instinct MI300X OAM"}
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
python3 -c 'import numpy' 2>/dev/null || python3 -m pip install -q numpy
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
    with open(W / "setup.log", "w") as log:
        rc = subprocess.call(["bash", "-c", SETUP], stdout=log, stderr=subprocess.STDOUT, env=ENV)
    if not os.path.exists("/opt/sycl/bin/clang++"):
        ENV.pop("SYCL_CXX", None)
    S["phase"] = "ready" if rc == 0 else "setup_failed"
    S["setup_rc"] = rc

def run(args):
    while S["phase"] == "setup":
        time.sleep(2)
    if S["phase"] == "setup_failed":
        S["phase"] = "done"; S["rc"] = 99; return
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
http.server.ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("GT_PORT", "8000"))), H).serve_forever()
'''


class RunpodAPI:
    """Minimal Runpod REST v2 client (pods + GPU catalog)."""

    def __init__(self, api_key: str, base: str = API) -> None:
        self.key, self.base = api_key, base

    def _req(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
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


def _agent_call(url: str, token: str, method: str = "GET", data: bytes | None = None,
                headers: dict[str, str] | None = None, timeout: float = 60) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"X-Token": token, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, b""


def remote_test(task_file: str, *, vendor: str = "nvidia", gpu_type: str | None = None,
                cloud: str = "SECURE", image: str | None = None, max_minutes: float = 45,
                max_usd: float = 2.0, into: str = "docs/gpu-run", keep_pod: bool = False,
                api_key: str | None = None, api: RunpodAPI | None = None,
                log: Callable[[str], None] = print, poll_s: float = 10,
                agent_url: Callable[[str], str] | None = None) -> ToolResult:
    """Rent a GPU on Runpod, run ``gpu_test`` there through a bundle, import the results."""
    key = api_key or os.environ.get("RUNPOD_API_KEY", "")
    if api is None and not key:
        return ToolResult("remote_test", "toolchain_missing",
                          "RUNPOD_API_KEY is not set", {},
                          ["Create an API key in the Runpod console and export RUNPOD_API_KEY."])
    api = api or RunpodAPI(key)
    gpu = gpu_type or DEFAULT_GPU[vendor]
    price = api.gpu_price(gpu, cloud)
    if price is not None:
        max_minutes = min(max_minutes, max_usd / price * 60)
        if max_minutes < 10:
            return ToolResult("remote_test", "error",
                              f"{gpu} costs ${price}/h: max_usd ${max_usd} allows only "
                              f"{max_minutes:.0f} min", {"price_per_hour": price})
    tmp = Path(tempfile.mkdtemp(prefix="remote-test-"))
    b = make_bundle(task_file, str(tmp / "bundle.sh"))
    if not b.ok:
        return b
    token = secrets.token_urlsafe(24)
    body: dict[str, Any] = {
        "name": f"oneapi-gputest-{secrets.token_hex(3)}", "cloud": cloud,
        "gpu": {"id": gpu, "count": 1}, "image": image or DEFAULT_IMAGE[vendor],
        "disk": 60, "ports": ["8000/http"],
        "env": {"GT_TOKEN": token, "GT_AGENT": base64.b64encode(POD_AGENT.encode()).decode()},
        "entrypoint": ["bash", "-c"],
        "cmd": ["echo \"$GT_AGENT\" | base64 -d > /gt_agent.py && exec python3 /gt_agent.py"],
    }
    if vendor == "nvidia":
        body["gpu"]["minCudaVersion"] = "12.4"
    t0 = time.time()
    deadline = t0 + max_minutes * 60
    pod_id: str | None = None
    ev: dict[str, Any] = {"gpu": gpu, "cloud": cloud, "image": body["image"],
                          "price_per_hour": price, "max_minutes": round(max_minutes, 1)}
    status: dict[str, Any] = {}
    result: ToolResult
    try:
        try:
            pod = api.create_pod(body)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            return ToolResult("remote_test", "error", f"pod create failed: HTTP {e.code}: {msg}",
                              ev, ["Often means no stock for this GPU right now."])
        pod_id = pod["id"]
        ev["pod_id"] = pod_id
        log(f"created pod {pod_id} ({gpu}, ${price}/h)")
        url = agent_url(pod_id) if agent_url else f"https://{pod_id}-8000.proxy.runpod.net"
        # Wait for the agent to answer, then upload the bundle.
        while time.time() < deadline:
            code, _ = _agent_call(url + "/status", token, timeout=20)
            if code == 200:
                break
            time.sleep(poll_s)
        else:
            raise TimeoutError("the pod's agent never answered before the time limit")
        code, resp = _agent_call(url + "/bundle", token, "PUT", (tmp / "bundle.sh").read_bytes(),
                                 {"X-Args": json.dumps(["--vendor", vendor])}, timeout=300)
        if code != 202:
            raise RuntimeError(f"bundle upload failed: HTTP {code} {resp[:200]!r}")
        log("bundle uploaded; waiting for the run")
        last = ""
        while time.time() < deadline:
            code, raw = _agent_call(url + "/status", token, timeout=30)
            if code == 200:
                status = json.loads(raw)
                if status.get("phase") != last:
                    last = status.get("phase", "")
                    log(f"pod phase: {last}")
                if status.get("phase") == "done":
                    break
            time.sleep(poll_s)
        else:
            raise TimeoutError(f"run did not finish within {max_minutes:.0f} min")
        code, archive = _agent_call(url + "/results", token, timeout=120)
        if code != 200:
            raise RuntimeError("run finished but produced no results archive; "
                               f"setup log tail: {status.get('setup_tail', '')[-1500:]} "
                               f"run log tail: {status.get('run_tail', '')[-1500:]}")
        (tmp / "results.tar.gz").write_bytes(archive)
        ing = ingest_results(str(tmp / "results.tar.gz"), into)
        ev["ingest"] = ing.evidence
        remote_status = (ing.evidence.get("remote_result") or {}).get("status") or ing.status
        result = ToolResult("remote_test", remote_status, f"{gpu}: {ing.summary}", ev,
                            ing.limitations)
    except (TimeoutError, RuntimeError, urllib.error.URLError, KeyError, ValueError) as exc:
        ev["agent_status"] = {k: status.get(k) for k in ("phase", "rc", "setup_rc")}
        ev["setup_tail"] = status.get("setup_tail", "")[-3000:]
        ev["run_tail"] = status.get("run_tail", "")[-3000:]
        result = ToolResult("remote_test", "error", f"{type(exc).__name__}: {exc}"[:600], ev)
    finally:
        if pod_id and not keep_pod:
            api.delete_pod(pod_id)
            gone = api.get_pod(pod_id) is None
            ev["pod_deleted"] = gone
            log(f"deleted pod {pod_id}: {'confirmed' if gone else 'NOT CONFIRMED'}")
    minutes = (time.time() - t0) / 60
    ev["minutes"] = round(minutes, 1)
    if price is not None:
        ev["est_cost_usd"] = round(price * minutes / 60, 3)
    if pod_id and not keep_pod and not ev.get("pod_deleted"):
        result.limitations.append(f"Pod {pod_id} deletion could not be confirmed: check the "
                                  "Runpod console.")
    return result
