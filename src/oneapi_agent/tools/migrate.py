"""migrate: run SYCLomatic (c2s) on CUDA sources and report what it produced and warned about."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from ..parsers.c2s import (
    MIGRATED_SUFFIXES,
    merge_warnings,
    parse_c2s_errors,
    parse_c2s_stderr,
    scan_migrated_tree,
)
from ..result import ToolResult
from ..toolchain import (
    Runner,
    cuda_include_dir,
    find_tool,
    missing_tool_message,
    run_command,
)

CUDA_SUFFIXES = {".cu", ".cuh"}


def migrate(
    cuda_path: str,
    out_dir: str,
    *,
    in_root: str | None = None,
    cuda_include_path: str | None = None,
    extra_args: Sequence[str] = (),
    timeout: float = 900.0,
    runner: Runner | None = None,
    c2s_path: str | None = None,
) -> ToolResult:
    """Migrate a CUDA file or directory with c2s.

    ``cuda_path`` may be a single ``.cu`` file or a directory (all ``.cu`` files in it
    are migrated, with the directory as ``--in-root``). ``extra_args`` are passed through
    as ``--extra-arg=<arg>`` (e.g. ``-I/path/to/headers``).
    """
    src = Path(cuda_path)
    if not src.exists():
        return ToolResult("migrate", "error", f"input not found: {cuda_path}",
                          {"cuda_path": cuda_path})

    if src.is_dir():
        files = sorted(p for p in src.rglob("*.cu"))
        root = Path(in_root) if in_root else src
    else:
        files = [src]
        root = Path(in_root) if in_root else src.parent
    if not files:
        return ToolResult("migrate", "error", f"no .cu files under {cuda_path}",
                          {"cuda_path": cuda_path})

    c2s = c2s_path or find_tool("c2s")
    if runner is None and c2s is None:
        return ToolResult(
            "migrate",
            "toolchain_missing",
            "SYCLomatic (c2s) is not installed; nothing was migrated",
            {"failed_command": f"c2s --in-root={root} --out-root={out_dir} ...",
             "reason": missing_tool_message("c2s")},
            ["No migration was attempted."],
        )
    c2s = c2s or "c2s"

    cuda_inc = cuda_include_path or cuda_include_dir()
    argv = [c2s, f"--in-root={root.resolve()}", f"--out-root={Path(out_dir).resolve()}"]
    if cuda_inc:
        argv.append(f"--cuda-include-path={cuda_inc}")
    for a in extra_args:
        argv.append(f"--extra-arg={a}")
    argv += [str(f.resolve()) for f in files]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    res = (runner or run_command)(argv, timeout=timeout)

    evidence: dict = {"run": res.evidence(), "input_files": [str(f) for f in files],
                      "in_root": str(root), "out_dir": str(out_dir)}
    limitations = [
        "c2s output compiles only with the dpct helper headers on the include path "
        "(<c2s install>/include).",
        "Migration warnings say what c2s could not translate; they do not prove the rest is correct. "
        "Run compile + verify.",
    ]
    if not cuda_inc:
        limitations.append(
            "No CUDA include path was found (set ONEAPI_AGENT_CUDA_INCLUDE); c2s may fail to parse."
        )

    if res.not_found:
        return ToolResult("migrate", "toolchain_missing", f"could not execute {c2s}",
                          {**evidence, "failed_command": res.command}, limitations)
    if res.timed_out:
        return ToolResult("migrate", "timeout", f"c2s exceeded {timeout}s", evidence, limitations)

    stderr_warnings = parse_c2s_stderr(res.stderr)
    out_path = Path(out_dir)
    comment_warnings = scan_migrated_tree(out_path) if out_path.exists() else []
    warnings = merge_warnings(stderr_warnings, comment_warnings)
    errors = parse_c2s_errors(res.stderr)

    migrated = sorted(
        str(p.relative_to(out_path))
        for p in out_path.rglob("*")
        if p.is_file() and p.suffix in MIGRATED_SUFFIXES
    ) if out_path.exists() else []

    evidence.update(
        {
            "migrated_files": migrated,
            "warnings": [w.to_dict() for w in warnings],
            "warning_counts": dict(Counter(w.code for w in warnings)),
            "errors": errors,
        }
    )

    if res.returncode != 0 or errors or not migrated:
        why = []
        if res.returncode != 0:
            why.append(f"c2s exited {res.returncode}")
        if errors:
            why.append(f"{len(errors)} parse error(s), first: {errors[0]['message']}")
        if not migrated:
            why.append("no migrated files produced")
        return ToolResult("migrate", "fail", "migration incomplete: " + "; ".join(why),
                          evidence, limitations)

    return ToolResult(
        "migrate",
        "ok",
        f"migrated {len(files)} CUDA file(s) -> {len(migrated)} file(s); "
        f"{len(warnings)} DPCT warning(s)"
        + (f" ({', '.join(f'{k}x{v}' for k, v in sorted(evidence['warning_counts'].items()))})"
           if warnings else ""),
        evidence,
        limitations,
    )
