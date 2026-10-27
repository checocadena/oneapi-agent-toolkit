"""Portable GPU test bundles: test a port on a machine this tool cannot reach.

``make_bundle`` packs a task (its sources, reference, include dirs and inputs) plus this
package's own source into ONE self-extracting shell script. On any machine with a GPU, a SYCL
compiler and Python 3, ``bash <bundle>.sh`` runs ``gpu_test`` and writes a results archive.
``ingest_results`` brings that archive back as evidence.

Works where only a browser is available too: upload the file to a JupyterLab session and run
``!bash <bundle>.sh`` in a notebook cell.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Any

from .result import ToolResult
from .tools.gpu_test import load_task

PKG_DIR = Path(__file__).resolve().parent
_SKIP_DIRS = {"build", "__pycache__", ".git", "runs", "work", ".venv"}
MAX_BUNDLE_BYTES = 50 * 1024 * 1024

RUNNER = r'''#!/usr/bin/env bash
# oneapi-agent GPU test bundle: __NAME__
# Run on any machine with a GPU (NVIDIA, AMD or Intel), a SYCL compiler (icpx or intel/llvm
# clang++) and Python 3:   bash __FILE__ [extra gpu-test args, e.g. --vendor amd]
# Writes gputest-results-__NAME__-<host>.tar.gz next to this file and prints the summary.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
D="$(mktemp -d)"
sed '1,/^__PAYLOAD_BELOW__$/d' "$0" | base64 -d | tar xz -C "$D" || { echo "bundle extract failed"; exit 2; }
cd "$D"
export HOME="${HOME:-/root}"  # cloud-init and some agents start us without HOME
for f in /opt/intel/oneapi/setvars.sh "$HOME/intel/oneapi/setvars.sh"; do
  if [ -z "${ONEAPI_ROOT:-}" ] && [ -f "$f" ]; then . "$f" --force >/dev/null 2>&1 || true; fi
done
[ -d /opt/rocm/bin ] && export PATH="$PATH:/opt/rocm/bin"
ensure_numpy() {
  python3 -c "import numpy" 2>/dev/null && return 0
  # Distro Pythons (Ubuntu 24.04+) refuse plain `pip install` (PEP 668) and may lack pip:
  # prefer the distro package when root, then pip with the escape hatch.
  if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
    (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-numpy) \
      >/dev/null 2>&1
    python3 -c "import numpy" 2>/dev/null && return 0
  fi
  python3 -m pip install -q --user numpy 2>/dev/null \
    || python3 -m pip install -q --user --break-system-packages numpy 2>/dev/null \
    || pip install -q numpy 2>/dev/null
  python3 -c "import numpy" 2>/dev/null
}
mkdir -p "$D/out/results"
ensure_numpy || echo "WARNING: numpy is missing and could not be installed"
# Large task files (task "fetch") are downloaded here and checked against their sha256 instead
# of travelling in the bundle.
fetch_file() {  # <dest> <url> <sha256>
  [ -f "$1" ] && echo "$3  $1" | sha256sum -c - >/dev/null 2>&1 && return 0
  mkdir -p "$(dirname "$1")"
  curl -fsSL --retry 3 -o "$1.part" "$2" && echo "$3  $1.part" | sha256sum -c - >/dev/null \
    && mv "$1.part" "$1" && echo "fetched $1" \
    || { echo "FETCH FAILED: $2 -> $1"; rm -f "$1.part"; }
}
__FETCH__
# The task's own setup script (task "gpu_setup"), sourced so that what it exports (e.g. a
# library prefix used in base_flags) reaches gpu-test. Its output goes to setup.log.
__SETUP__
PYTHONPATH="$D/pkg" python3 -m oneapi_agent.cli --summary gpu-test "__TASK__" --out "$D/out" "$@" \
  2>&1 | tee "$D/out/run.log"
rc=${PIPESTATUS[0]}
if [ ! -f "$D/out/summary.json" ]; then  # crashed before writing a result: say so, with the log
  python3 - "$D/out" "$rc" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
log = (out / "run.log").read_text(errors="replace")[-6000:] if (out / "run.log").exists() else ""
summary = {"tool": "gpu_test", "status": "error",
           "summary": "gpu-test crashed before writing a result (exit %s)" % sys.argv[2],
           "evidence": {"run_log_tail": log}, "limitations": []}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
(out / "SUMMARY.md").write_text("# GPU test crashed\n\n```\n" + log + "\n```\n")
PY
fi
OUTF="$HERE/gputest-results-__NAME__-$(hostname | tr -cd 'A-Za-z0-9_-').tar.gz"
EXTRA=""; [ -f "$D/out/setup.log" ] && EXTRA="setup.log"
tar czf "$OUTF" -C "$D/out" summary.json SUMMARY.md results run.log $EXTRA
echo; cat "$D/out/SUMMARY.md"; echo; echo "results archive: $OUTF"
exit $rc
__PAYLOAD_BELOW__
'''


def _common_root(paths: list[Path]) -> Path:
    return Path(os.path.commonpath([str(p) for p in paths]))


def _add_tree(tar: tarfile.TarFile, src: Path, arcname: str,
              skip: frozenset[Path] = frozenset()) -> None:
    if src.is_file():
        tar.add(src, arcname=arcname)
        return
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in _SKIP_DIRS for part in rel.parts) or not p.is_file():
            continue
        if p.absolute() in skip:  # fetched on the GPU machine instead
            continue
        tar.add(p, arcname=f"{arcname}/{rel.as_posix()}")


def _fetch_entries(task: dict[str, Any]) -> list[dict[str, str]]:
    """The task's ``fetch`` list: ``{"path", "url", "sha256"}`` per large file, path relative
    to the task dir. Checked here so a bad entry fails at bundle time, not on the machine."""
    entries = task.get("fetch") or []
    for e in entries:
        if not (isinstance(e, dict) and {"path", "url", "sha256"} <= e.keys()):
            raise ValueError(f"fetch entry needs path, url and sha256: {e!r}")
        if Path(e["path"]).is_absolute() or ".." in Path(e["path"]).parts:
            raise ValueError(f"fetch path must stay inside the task dir: {e['path']!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", e["sha256"]) or not e["url"].startswith("https://"):
            raise ValueError(f"fetch entry needs an https url and a sha256 hex digest: {e!r}")
    return entries


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def make_bundle(task_file: str, output: str | None = None) -> ToolResult:
    """Pack ``task_file``'s task and this package into one runnable ``.sh`` file."""
    task = load_task(task_file)
    tdir = Path(task["task_dir"])
    try:
        fetch = _fetch_entries(task)
    except ValueError as exc:
        return ToolResult("bundle", "error", str(exc))
    fetched = frozenset((tdir / e["path"]).absolute() for e in fetch)
    setup = task.get("gpu_setup")
    if setup and (Path(setup).is_absolute() or ".." in Path(setup).parts):
        return ToolResult("bundle", "error", f"gpu_setup must be inside the task dir: {setup!r}")
    missing = [s for s in [task["cuda_path_abs"], *task["sycl_sources_abs"],
                           *[i for i in task["inputs_abs"] if Path(i).absolute() not in fetched],
                           *task["include_dirs_abs"], *([str(tdir / setup)] if setup else [])]
               if not Path(s).exists()]
    if missing:
        return ToolResult("bundle", "error", f"task references missing file(s): {missing}",
                          {"missing": missing},
                          ["Inputs generated by a script (e.g. examples/gen_inputs.py) must be "
                           "generated before bundling."])
    # Everything keeps its path relative to one common root, so "../common" style task paths
    # stay valid inside the bundle.
    parts = [tdir, *map(Path, task["include_dirs_abs"]),
             *[Path(i) for i in task["inputs_abs"] if Path(i).absolute() not in fetched]]
    root = _common_root([p if p.is_dir() else p.parent for p in parts])
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        added: set[str] = set()
        for p in parts:
            rel = p.relative_to(root).as_posix()
            arc = "task" if rel == "." else f"task/{rel}"
            if arc in added or any(arc.startswith(a + "/") for a in added):
                continue
            _add_tree(tar, p, arc, fetched)
            added.add(arc)
        _add_tree(tar, PKG_DIR, "pkg/oneapi_agent")
    data = buf.getvalue()
    if len(data) > MAX_BUNDLE_BYTES:
        return ToolResult("bundle", "error", f"bundle would be {len(data)} bytes "
                          f"(> {MAX_BUNDLE_BYTES}); move large inputs out of the task dir")
    name = re.sub(r"[^A-Za-z0-9_-]", "-", task["name"])
    out = Path(output or f"gputest-{name}.sh").resolve()
    task_arc = "task/" + (tdir / Path(task_file).name).relative_to(root).as_posix()
    tdir_arc = "$D/" + str(Path(task_arc).parent)
    fetch_sh = "\n".join(f'fetch_file "{tdir_arc}"/{_q(e["path"])} {_q(e["url"])} {e["sha256"]}'
                         for e in fetch)
    setup_sh = (f'{{ . "{tdir_arc}"/{_q(setup)}; }} > "$D/out/setup.log" 2>&1 '
                '|| echo "gpu_setup failed: see setup.log"' if setup else "")
    script = (RUNNER.replace("__NAME__", name).replace("__FILE__", out.name)
              .replace("__TASK__", task_arc).replace("__FETCH__", fetch_sh)
              .replace("__SETUP__", setup_sh))
    out.write_text(script + base64.encodebytes(data).decode())
    out.chmod(0o755)
    return ToolResult("bundle", "ok", f"wrote {out.name} ({out.stat().st_size // 1024} KiB): "
                      f"run `bash {out.name}` on a GPU machine",
                      {"bundle": str(out), "task": task["name"], "bytes": out.stat().st_size,
                       "root": str(root)},
                      ["The target machine needs a SYCL compiler (icpx, or intel/llvm clang++ "
                       "for NVIDIA/AMD), Python 3 and numpy (installed if missing)."])


def _safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    safe = []
    for m in tar.getmembers():
        p = Path(m.name)
        if p.is_absolute() or ".." in p.parts or not (m.isfile() or m.isdir()):
            raise ValueError(f"unsafe path in archive: {m.name}")
        safe.append(m)
    return safe


def ingest_results(archive: str, into: str = "docs/gpu-run") -> ToolResult:
    """Unpack a results archive from a bundle run into ``<into>/<vendor>-<device>/``."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = _safe_members(tar)
            names = {m.name for m in members}
            if "summary.json" not in names:
                return ToolResult("ingest", "error", "archive has no summary.json: not a "
                                  "gpu-test results archive", {"members": sorted(names)[:50]})
            raw = tar.extractfile("summary.json")
            summary: dict[str, Any] = json.loads(raw.read()) if raw else {}
            ev = summary.get("evidence", {})
            dev = ev.get("device") or {}
            vendor, name = str(ev.get("vendor", "unknown")), str(dev.get("name", "device"))
            label = name if name.lower().startswith(vendor.lower()) else f"{vendor}-{name}"
            slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:60]
            dest = Path(into) / slug / str(ev.get("task", "task"))
            dest.mkdir(parents=True, exist_ok=True)
            kw: dict[str, Any] = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tar.extractall(dest, members=members, **kw)  # noqa: S202 - members validated above
    except (OSError, tarfile.TarError, ValueError, json.JSONDecodeError) as exc:
        return ToolResult("ingest", "error", f"could not read {archive}: {exc}")
    return ToolResult("ingest", "ok" if summary.get("status") == "ok" else "fail",
                      f"{summary.get('status')}: {summary.get('summary', '')}"[:500],
                      {"dest": str(dest), "remote_result": summary},
                      ["This archive was produced on another machine; its evidence is as "
                       "trustworthy as that machine and the bundle that ran there."])
