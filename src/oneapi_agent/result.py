"""The single return type shared by every tool.

Every tool answers with *evidence*, never a bare claim: a status word, a one-line
summary, a dict of evidence (commands, logs, parsed diagnostics, numbers) and an
explicit list of limitations (what the result does NOT prove).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["ok", "fail", "error", "toolchain_missing", "timeout"]

#: ok                -> the tool ran and the checked property holds
#: fail              -> the tool ran and the checked property does NOT hold
#:                      (compile errors, numeric mismatch, migration errors)
#: error             -> the tool could not do its job (bad input, crash, missing file)
#: toolchain_missing -> a required external binary was not found; evidence names it
#: timeout           -> an external command exceeded its time limit
STATUSES: tuple[str, ...] = ("ok", "fail", "error", "toolchain_missing", "timeout")


@dataclass
class ToolResult:
    tool: str
    status: Status
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"invalid status {self.status!r}; expected one of {STATUSES}")

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        return cls(
            tool=data["tool"],
            status=data["status"],
            summary=data["summary"],
            evidence=dict(data.get("evidence", {})),
            limitations=list(data.get("limitations", [])),
        )
