from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from oneapi_agent.toolchain import CommandResult

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


class FakeRunner:
    """Records calls; answers with a scripted CommandResult (or a callable producing one)."""

    def __init__(self, respond: Callable[[list[str], dict[str, Any]], CommandResult]) -> None:
        self.respond = respond
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kw: Any) -> CommandResult:
        self.calls.append((list(argv), kw))
        return self.respond(list(argv), kw)


def result(argv: list[str], rc: int = 0, stdout: str = "", stderr: str = "",
           duration: float = 0.01) -> CommandResult:
    return CommandResult(argv, rc, stdout, stderr, duration)


@pytest.fixture
def no_toolchain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every oneAPI binary undiscoverable, regardless of the host."""
    # import_module, not "import a.b as x": oneapi_agent.tools re-exports functions named
    # like their submodules (benchmark, migrate), which would shadow the modules.
    tc = importlib.import_module("oneapi_agent.toolchain")
    tb = importlib.import_module("oneapi_agent.tools.benchmark")
    tcomp = importlib.import_module("oneapi_agent.tools.compile")
    tm = importlib.import_module("oneapi_agent.tools.migrate")

    def none(name: str) -> None:
        return None

    for mod in (tc, tb, tcomp, tm):
        monkeypatch.setattr(mod, "find_tool", none)
    monkeypatch.setattr(tc, "dpct_include_dir", lambda: None)
    monkeypatch.setattr(tcomp, "dpct_include_dir", lambda: None)
