"""Parse clang-style diagnostics as emitted by icpx (and by c2s, which is clang-based)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

# path:line:col: severity: message [-Wflag]
_DIAG_RE = re.compile(
    r"^(?P<file>[^\s:][^:]*?):(?P<line>\d+):(?:(?P<col>\d+):)?\s*"
    r"(?P<severity>fatal error|error|warning|note|remark):\s*(?P<message>.*?)\s*$"
)
# driver-level, no location: "icpx: error: linker command failed ..."
_DRIVER_RE = re.compile(r"^(?P<tool>[\w.+-]+): (?P<severity>fatal error|error|warning): (?P<message>.*)$")
# "warning: CUDA version is newer ..." (no file)
_BARE_RE = re.compile(r"^(?P<severity>fatal error|error|warning): (?P<message>.*)$")
# GNU ld: "file.cpp:(.text+0x26e): undefined reference to `helper(float*)'"
_LD_UNDEF_RE = re.compile(r"^(?P<file>[^:]+):\(.*?\): (?P<message>undefined reference to .*)$")
_FLAG_RE = re.compile(r"\s*\[(-W[\w\-=+]+)\]$")


@dataclass
class Diagnostic:
    file: str | None
    line: int | None
    column: int | None
    severity: str  # "error" | "warning" | "note" | "remark" | "fatal error"
    message: str
    flag: str | None = None
    snippet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_error(self) -> bool:
        return self.severity in ("error", "fatal error")


def parse_clang_diagnostics(text: str) -> list[Diagnostic]:
    """Parse diagnostics, attaching the source snippet line that clang prints after each one."""
    diags: list[Diagnostic] = []
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        m = _DIAG_RE.match(line)
        if m:
            msg = m.group("message")
            flag = None
            fm = _FLAG_RE.search(msg)
            if fm:
                flag = fm.group(1)
                msg = msg[: fm.start()].rstrip()
            snippet = None
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                sm = re.match(r"^\s*\d+\s*\|\s?(.*)$", nxt)
                if sm:
                    snippet = sm.group(1)
            diags.append(
                Diagnostic(
                    file=m.group("file"),
                    line=int(m.group("line")),
                    column=int(m.group("col")) if m.group("col") else None,
                    severity=m.group("severity"),
                    message=msg,
                    flag=flag,
                    snippet=snippet,
                )
            )
            continue
        m2 = _LD_UNDEF_RE.match(line)
        if m2:
            diags.append(Diagnostic(m2.group("file"), None, None, "error", m2.group("message")))
            continue
        m3 = _DRIVER_RE.match(line)
        if m3 and not line.startswith(("In file included", " ")):
            diags.append(
                Diagnostic(None, None, None, m3.group("severity"),
                           f"{m3.group('tool')}: {m3.group('message')}")
            )
            continue
        m4 = _BARE_RE.match(line)
        if m4:
            msg = m4.group("message")
            flag = None
            fm = _FLAG_RE.search(msg)
            if fm:
                flag = fm.group(1)
                msg = msg[: fm.start()].rstrip()
            diags.append(Diagnostic(None, None, None, m4.group("severity"), msg, flag))
    return diags


def summarize(diags: list[Diagnostic]) -> dict[str, int]:
    counts = {"error": 0, "warning": 0, "note": 0}
    for d in diags:
        key = "error" if d.is_error else d.severity
        counts[key] = counts.get(key, 0) + 1
    return counts
