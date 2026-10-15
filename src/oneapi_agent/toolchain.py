"""Locating oneAPI binaries and running external commands.

Every external command goes through a ``Runner`` so tests can substitute a fake
one and no tool ever pretends a command ran when it did not.

Binary lookup order (first hit wins):
  1. explicit env override (``ONEAPI_AGENT_ICPX``, ``ONEAPI_AGENT_C2S``, ``ONEAPI_AGENT_SYCL_LS``)
  2. ``$PATH``
  3. ``$ONEAPI_ROOT``/common install prefixes (``/opt/intel/oneapi/...``)
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG_LIMIT = 12_000  # characters of stdout/stderr kept in evidence (tail-biased)


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    not_found: bool = False
    cwd: str | None = None
    env_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def command(self) -> str:
        return shlex.join(self.argv)

    def evidence(self, limit: int = LOG_LIMIT) -> dict[str, Any]:
        """Compact, JSON-safe record of what actually ran."""
        out: dict[str, Any] = {
            "command": self.command,
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 6),
            "stdout": truncate(self.stdout, limit),
            "stderr": truncate(self.stderr, limit),
        }
        if self.cwd:
            out["cwd"] = self.cwd
        if self.env_overrides:
            out["env_overrides"] = dict(self.env_overrides)
        if self.timed_out:
            out["timed_out"] = True
        if self.not_found:
            out["not_found"] = True
        return out


Runner = Callable[..., CommandResult]


def truncate(text: str, limit: int = LOG_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} chars truncated]...\n{text[-tail:]}"


def run_command(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = 600.0,
    stdin: str | None = None,
) -> CommandResult:
    """Run ``argv`` (no shell) and capture everything. Never raises for process failures."""
    argv = [str(a) for a in argv]
    full_env = None
    if env:
        full_env = dict(os.environ)
        full_env.update(env)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=full_env,
            input=stdin,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            argv, None, "", str(exc), time.perf_counter() - start, not_found=True,
            cwd=str(cwd) if cwd else None, env_overrides=dict(env or {}),
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            argv,
            None,
            _as_text(exc.stdout),
            _as_text(exc.stderr),
            time.perf_counter() - start,
            timed_out=True,
            cwd=str(cwd) if cwd else None,
            env_overrides=dict(env or {}),
        )
    return CommandResult(
        argv,
        proc.returncode,
        proc.stdout,
        proc.stderr,
        time.perf_counter() - start,
        cwd=str(cwd) if cwd else None,
        env_overrides=dict(env or {}),
    )


def _as_text(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data


def split_command(cmd: str | Sequence[str]) -> list[str]:
    """Accept either a shell-like string or an argv list."""
    if isinstance(cmd, str):
        return shlex.split(cmd)
    return [str(c) for c in cmd]


# --------------------------------------------------------------------------- lookup

_ENV_OVERRIDES = {
    "icpx": "ONEAPI_AGENT_ICPX",
    "c2s": "ONEAPI_AGENT_C2S",
    "sycl-ls": "ONEAPI_AGENT_SYCL_LS",
}

_FALLBACK_DIRS = (
    "{oneapi_root}/compiler/latest/bin",
    "{oneapi_root}/dpcpp-ct/latest/bin",
    "/opt/intel/oneapi/compiler/latest/bin",
    "/opt/intel/oneapi/dpcpp-ct/latest/bin",
)


def find_tool(name: str) -> str | None:
    """Return an executable path for ``name`` or None. Never guesses beyond the documented order."""
    override = os.environ.get(_ENV_OVERRIDES.get(name, ""), "")
    if override:
        return override if (os.path.isfile(override) and os.access(override, os.X_OK)) else None
    hit = shutil.which(name)
    if hit:
        return hit
    oneapi_root = os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi")
    for pattern in _FALLBACK_DIRS:
        cand = Path(pattern.format(oneapi_root=oneapi_root)) / name
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    if name == "c2s":  # SYCLomatic also ships the same binary as `dpct`
        return shutil.which("dpct")
    return None


def missing_tool_message(name: str) -> str:
    env = _ENV_OVERRIDES.get(name)
    hint = f" or set {env}=/path/to/{name}" if env else ""
    return f"`{name}` not found on PATH{hint}"


def dpct_include_dir() -> str | None:
    """Directory holding ``dpct/dpct.hpp`` (SYCLomatic helper headers), if discoverable."""
    explicit = os.environ.get("ONEAPI_AGENT_DPCT_INCLUDE")
    if explicit:
        return explicit if (Path(explicit) / "dpct" / "dpct.hpp").is_file() else None
    c2s = find_tool("c2s")
    if c2s:
        inc = Path(c2s).resolve().parent.parent / "include"
        if (inc / "dpct" / "dpct.hpp").is_file():
            return str(inc)
    return None


def cuda_include_dir() -> str | None:
    """CUDA headers needed by c2s to parse ``.cu`` files."""
    candidates = [
        os.environ.get("ONEAPI_AGENT_CUDA_INCLUDE", ""),
        os.path.join(os.environ.get("CUDA_PATH", ""), "include") if os.environ.get("CUDA_PATH") else "",
        "/usr/local/cuda/include",
    ]
    for c in candidates:
        if c and (Path(c) / "cuda_runtime.h").is_file():
            return c
    return None


def toolchain_report() -> dict[str, Any]:
    """What this machine has. Used by ``oneapi-agent doctor`` and in evidence."""
    return {
        "icpx": find_tool("icpx"),
        "c2s": find_tool("c2s"),
        "sycl-ls": find_tool("sycl-ls"),
        "dpct_include": dpct_include_dir(),
        "cuda_include": cuda_include_dir(),
    }
