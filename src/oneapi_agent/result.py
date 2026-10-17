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


SCHEMA_ID = ("https://github.com/checocadena/oneapi-agent-toolkit/blob/main/"
             "docs/tool-result-schema.json")
_FIELD_DOCS = {
    "tool": "Name of the tool that produced this result (e.g. compile, verify).",
    "status": "ok: the checked property holds. fail: the tool ran and it does not hold. "
              "error: the tool could not do its job. toolchain_missing: a required binary "
              "was not found (evidence names it). timeout: an external command timed out.",
    "summary": "One line for humans; never a substitute for evidence.",
    "evidence": "Tool-specific facts: commands that ran, exit codes, parsed diagnostics, "
                "numbers. Keys differ per tool.",
    "limitations": "What this result does NOT prove. Carry these into any report.",
}


def tool_result_json_schema() -> dict[str, Any]:
    """JSON Schema (draft 2020-12) of ``ToolResult.to_dict()``, derived from the dataclass.

    ``docs/tool-result-schema.json`` is this function's output; a test keeps them equal.
    """
    from dataclasses import fields

    types: dict[str, dict[str, Any]] = {
        "tool": {"type": "string"},
        "status": {"type": "string", "enum": list(STATUSES)},
        "summary": {"type": "string"},
        "evidence": {"type": "object"},
        "limitations": {"type": "array", "items": {"type": "string"}},
    }
    names = [f.name for f in fields(ToolResult)]
    if set(names) != set(types):  # a new field must be described here before it ships
        raise RuntimeError(f"ToolResult fields changed: {names}")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "ToolResult",
        "description": "What every oneapi-agent tool returns, over MCP and on the CLI "
                       "(stdout). CLI exit code: 0 = ok, 1 = fail, 2 = anything else.",
        "type": "object",
        "properties": {n: {**types[n], "description": _FIELD_DOCS[n]} for n in names},
        "required": names,
        "additionalProperties": False,
    }
