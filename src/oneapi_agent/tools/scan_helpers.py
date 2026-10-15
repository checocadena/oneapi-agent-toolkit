"""scan_helpers: inventory of dpct:: helper usage left in migrated code."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..parsers.c2s import MIGRATED_SUFFIXES, parse_dpct_comments
from ..parsers.dpct_scan import SUGGESTIONS, scan_source, strip_comments_and_strings
from ..result import ToolResult

# Patterns that are valid SYCL but are usually artifacts of mechanical migration.
ADVISORIES: list[tuple[str, re.Pattern[str], str]] = [
    (
        "hardcoded_subgroup_32",
        re.compile(r"reqd_sub_group_size\s*\(\s*32\s*\)"),
        "Kernel requires sub-group size 32 (CUDA warp size); not every SYCL device supports it. "
        "Prefer sub-group-size-agnostic code or query info::device::sub_group_sizes.",
    ),
    (
        "nd_item3_for_1d",
        re.compile(r"sycl::range<3>\s*\(\s*1\s*,\s*1\s*,"),
        "1-D launch expressed as a 3-D nd_range with leading 1s (dpct idiom). "
        "An idiomatic port uses sycl::range<1>/nd_range<1>.",
    ),
    (
        "native_math",
        re.compile(r"sycl::native::\w+"),
        "Reduced-precision sycl::native:: math (c2s emits it for some CUDA calls, e.g. expf, "
        "without a DPCT warning). Accuracy is implementation-defined; use the sycl:: full-"
        "precision builtin unless verify shows native precision is acceptable.",
    ),
    (
        "free_function_item_query",
        re.compile(r"this_work_item::get_nd_item"),
        "Kernel fetches its nd_item via this_work_item::get_nd_item (extension). Passing the "
        "item as a lambda parameter is more portable.",
    ),
]


def scan_helpers(path: str, *, max_occurrences: int = 200) -> ToolResult:
    """Scan a file or directory for dpct:: helpers, DPCT_* macros and dpct includes."""
    root = Path(path)
    if not root.exists():
        return ToolResult("scan_helpers", "error", f"path not found: {path}", {"path": path})
    files = (
        [root]
        if root.is_file()
        else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in MIGRATED_SUFFIXES)
    )
    uses = []
    unresolved = []
    advisories: list[dict] = []
    for f in files:
        rel = f.name if root.is_file() else str(f.relative_to(root))
        text = f.read_text(errors="replace")
        uses.extend(scan_source(text, rel))
        unresolved.extend(w.to_dict() for w in parse_dpct_comments(text, rel))
        code_only = strip_comments_and_strings(text).splitlines()
        for name, pat, advice in ADVISORIES:
            for i, line in enumerate(code_only, start=1):
                if pat.search(line):
                    advisories.append({"kind": name, "file": rel, "line": i, "advice": advice})

    by_category = Counter(u.category for u in uses)
    by_symbol = Counter(u.symbol for u in uses)
    by_file = Counter(u.file for u in uses)
    evidence = {
        "path": str(path),
        "files_scanned": len(files),
        "total": len(uses),
        "by_category": dict(by_category.most_common()),
        "by_symbol": dict(by_symbol.most_common()),
        "by_file": dict(by_file.most_common()),
        "occurrences": [u.to_dict() for u in uses[:max_occurrences]],
        "occurrences_truncated": max(0, len(uses) - max_occurrences),
        "suggestions": {c: SUGGESTIONS.get(c, SUGGESTIONS["other"]) for c in by_category},
        "unresolved_dpct_comments": unresolved,
        "advisories": advisories,
    }
    limitations = [
        "Static text scan: macros that expand to dpct:: calls, or helpers reached through "
        "user-defined wrappers, are not detected.",
        "Zero helpers means the code no longer depends on dpct headers; it does not prove "
        "the code is correct or fast. Run compile + verify + benchmark.",
    ]
    if not files:
        return ToolResult("scan_helpers", "error", f"no C/C++ sources under {path}", evidence,
                          limitations)
    if uses:
        top = ", ".join(f"{k}={v}" for k, v in by_category.most_common(4))
        return ToolResult(
            "scan_helpers",
            "fail",
            f"{len(uses)} dpct helper use(s) remain in {len(by_file)} file(s) ({top})",
            evidence,
            limitations,
        )
    extra = f"; {len(unresolved)} DPCT comment(s) and {len(advisories)} advisory note(s) remain" \
        if (unresolved or advisories) else ""
    return ToolResult("scan_helpers", "ok",
                      f"no dpct helpers in {len(files)} file(s){extra}", evidence, limitations)
