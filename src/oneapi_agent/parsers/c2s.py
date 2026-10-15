"""Parse SYCLomatic (c2s / dpct) output.

c2s reports migration diagnostics two ways, and we read both:

* on stderr, clang-style: ``file.cu:30:21: warning: DPCT1010:0: <message>``
* inline in the migrated sources, as comment blocks::

      /*
      DPCT1010:0: SYCL uses exceptions to report errors ...
      rewrite this code.
      */

The stderr form carries the *original* CUDA location; the comment form carries
the location in the *migrated* file. Warnings are de-duplicated on (code, id).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .diagnostics import parse_clang_diagnostics

_STDERR_RE = re.compile(
    r"^(?P<file>[^\s:][^:]*?):(?P<line>\d+):(?P<col>\d+):\s*warning:\s*"
    r"(?P<code>DPCT\d{4}):(?P<id>\d+):\s*(?P<message>.*)$"
)
_COMMENT_START_RE = re.compile(r"(?P<code>DPCT\d{4}):(?P<id>\d+):\s*(?P<message>.*)$")


@dataclass
class DpctWarning:
    code: str  # e.g. "DPCT1010"
    id: int  # per-run running number assigned by c2s
    message: str
    file: str | None = None  # original CUDA file (stderr form)
    line: int | None = None
    column: int | None = None
    migrated_file: str | None = None  # migrated file (comment form)
    migrated_line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_c2s_stderr(text: str) -> list[DpctWarning]:
    out: list[DpctWarning] = []
    for raw in text.splitlines():
        m = _STDERR_RE.match(raw.rstrip())
        if m:
            out.append(
                DpctWarning(
                    code=m.group("code"),
                    id=int(m.group("id")),
                    message=m.group("message").strip(),
                    file=m.group("file"),
                    line=int(m.group("line")),
                    column=int(m.group("col")),
                )
            )
    return out


def parse_dpct_comments(source: str, migrated_file: str | None = None) -> list[DpctWarning]:
    """Extract ``DPCTxxxx:n:`` comment blocks (possibly spanning several lines)."""
    out: list[DpctWarning] = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        m = _COMMENT_START_RE.search(lines[i])
        if not m:
            i += 1
            continue
        start_line = i + 1
        parts = [m.group("message").strip()]
        # A block comment continues until "*/"; collect continuation lines.
        if "*/" not in lines[i]:
            j = i + 1
            while j < len(lines) and "*/" not in lines[j] and not _COMMENT_START_RE.search(lines[j]):
                parts.append(lines[j].strip().lstrip("/*").strip())
                j += 1
            i = j
        else:
            parts[0] = parts[0].split("*/")[0].strip()
            i += 1
        message = " ".join(p for p in parts if p)
        out.append(
            DpctWarning(
                code=m.group("code"),
                id=int(m.group("id")),
                message=message,
                migrated_file=migrated_file,
                migrated_line=start_line,
            )
        )
    return out


def merge_warnings(
    from_stderr: list[DpctWarning], from_comments: list[DpctWarning]
) -> list[DpctWarning]:
    """Join both views on (code, id); keep every warning that appears in either."""
    merged: dict[tuple[str, int], DpctWarning] = {}
    for w in from_stderr:
        merged[(w.code, w.id)] = DpctWarning(**w.to_dict())
    for w in from_comments:
        key = (w.code, w.id)
        if key in merged:
            merged[key].migrated_file = w.migrated_file
            merged[key].migrated_line = w.migrated_line
        else:
            merged[key] = DpctWarning(**w.to_dict())
    return sorted(merged.values(), key=lambda w: (w.id, w.code))


def parse_c2s_errors(stderr: str) -> list[dict[str, Any]]:
    """Hard errors c2s hit while parsing CUDA (missing headers, syntax errors).

    c2s often exits 0 even when it could not parse a file, so these must be
    surfaced explicitly.
    """
    return [d.to_dict() for d in parse_clang_diagnostics(stderr) if d.is_error]


def scan_migrated_tree(out_dir: str | Path) -> list[DpctWarning]:
    out: list[DpctWarning] = []
    root = Path(out_dir)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in MIGRATED_SUFFIXES:
            text = p.read_text(errors="replace")
            out.extend(parse_dpct_comments(text, str(p.relative_to(root))))
    return out


MIGRATED_SUFFIXES = {".cpp", ".hpp", ".h", ".cc", ".cxx", ".hh", ".inc", ".cu", ".cuh"}
