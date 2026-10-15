#!/usr/bin/env python3
"""With vs without the toolkit's checks: the same model, the same tasks, graded the same way.

  toolkit   normal loop: migrate, compile, scan_helpers, verify, benchmark, file tools.
            The loop declares success only on compile + verify + zero helpers.
  baseline  `--baseline`: migrate, compile and file tools only. The run ends when the model
            calls finish.
  baseline_shell  baseline plus `run`, a bash shell in the work dir, so the agent can build and
            run its own tests (the reference binary is in the work dir in every arm).

Both arms run with `--evaluate`: after the run, the harness grades the final sources with a
hidden compile + verify + helper scan that the model never sees.

Usage: scripts/ablation.py [--runs 3] [--jobs 4] [--max-iterations 10] [--out docs/ablation]
       [--examples saxpy fuzz fir softmax_hecbench] [--arms toolkit baseline baseline_shell]
       [--model claude-sonnet-5-5]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARM_FLAGS = {"toolkit": [], "baseline": ["--baseline"],
             "baseline_shell": ["--baseline", "--shell"]}
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".cu", ".cuh"}


def save_evidence(run_dir: Path, dst: Path) -> None:
    """Copy trace, report and final sources (no binaries) of one run into the repo."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("trace.jsonl", "report.md"):
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, dst / f)
    for p in (run_dir / "work").rglob("*"):
        if p.is_file() and p.suffix in SOURCE_SUFFIXES and "verify_out" not in p.parts:
            t = dst / "final" / p.relative_to(run_dir / "work")
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)


def run_one(example: str, arm: str, seed: int, max_it: int, runs_root: Path, model: str,
            evidence_root: Path) -> dict:
    task = REPO / "examples" / example / "task.json"
    runs_dir = runs_root / arm / example
    cmd = [sys.executable, "-m", "oneapi_agent.cli", "loop", "--config", str(task),
           "--max-iterations", str(max_it), "--runs-dir", str(runs_dir), "--evaluate"]
    cmd += ARM_FLAGS[arm]
    env = {**os.environ, "ONEAPI_AGENT_MODEL": model}
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
    try:
        out = json.loads(p.stdout[p.stdout.index("{"):])
    except ValueError:
        out = {"stop_reason": "harness_error", "stderr": p.stderr[-2000:]}
    ev = out.get("evaluation") or {}
    row = {"model": model, "example": example, "arm": arm, "seed": seed,
           "stop_reason": out.get("stop_reason"), "iterations": out.get("iterations"),
           "cost_usd": out.get("cost_usd"), "compiled": ev.get("compiled"),
           "verify": ev.get("verify_status"), "verify_summary": ev.get("verify_summary"),
           "helpers": ev.get("helpers"), "note": ev.get("note"),
           "run_dir": out.get("run_dir")}
    if out.get("run_dir"):
        dst = out_dir_for(evidence_root, model=model, arm=arm, example=example, seed=seed)
        save_evidence(Path(out["run_dir"]), dst)
        row["evidence_dir"] = str(dst.relative_to(REPO)) if dst.is_relative_to(REPO) else str(dst)
    row["correct_and_clean"] = bool(row["compiled"] and row["verify"] == "ok"
                                    and row["helpers"] == 0)
    claimed = row["stop_reason"] in ("success", "model_finished")
    row["claimed_done"] = claimed
    row["claimed_but_wrong"] = claimed and not row["correct_and_clean"]
    print(f"[{model} {arm:14}] {example:17} #{seed} stop={row['stop_reason']} verify={row['verify']} "
          f"helpers={row['helpers']} cost=${row['cost_usd'] or 0:.3f}", flush=True)
    return row


def out_dir_for(root: Path, *, model: str, arm: str, example: str, seed: int) -> Path:
    return root / "runs" / model / arm / example / f"run{seed}"


def summarize(rows: list[dict], examples: list[str]) -> str:
    def agg(sel: list[dict]) -> str:
        n = len(sel)
        ok = sum(r["correct_and_clean"] for r in sel)
        wrong = sum(r["verify"] == "fail" for r in sel)
        helpers = sum(1 for r in sel if (r["helpers"] or 0) > 0)
        lied = sum(r["claimed_but_wrong"] for r in sel)
        cost = sum(r["cost_usd"] or 0 for r in sel)
        return f"{ok}/{n} | {wrong}/{n} | {helpers}/{n} | {lied}/{n} | ${cost:.2f}"

    md = ["| Model | Example | Arm | correct + 0 helpers | wrong output | dpct helpers left | "
          "said done but wasn't | cost |", "|---|---|---|---|---|---|---|---|"]
    models = sorted({r.get("model", "?") for r in rows})
    arms = [a for a in ARM_FLAGS if any(r["arm"] == a for r in rows)]
    for m in models:
        for ex in ["**all**", *examples]:
            for arm in arms:
                sel = [r for r in rows if r.get("model", "?") == m and r["arm"] == arm
                       and (ex == "**all**" or r["example"] == ex)]
                if sel:
                    md.append(f"| {m} | {ex} | {arm} | {agg(sel)} |")
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-iterations", type=int, default=10)
    ap.add_argument("--out", default=str(REPO / "docs" / "ablation"))
    ap.add_argument("--examples", nargs="+",
                    default=["saxpy", "fuzz", "fir", "softmax_hecbench"])
    ap.add_argument("--arms", nargs="+", default=["toolkit", "baseline"], choices=list(ARM_FLAGS))
    ap.add_argument("--model", default=os.environ.get("ONEAPI_AGENT_MODEL", "claude-sonnet-5-5"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_root = REPO / "runs" / ("ablation-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S"))
    jobs = [(ex, arm, s) for s in range(1, a.runs + 1) for ex in a.examples for arm in a.arms]
    with ThreadPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(lambda j: run_one(*j, a.max_iterations, runs_root, a.model, out),
                             jobs))
    results = out / "results.json"
    if results.exists():  # accumulate across invocations (e.g. one per model)
        rows = json.loads(results.read_text()) + rows
    results.write_text(json.dumps(rows, indent=2))
    (out / "SUMMARY.md").write_text(summarize(rows, a.examples) + "\n")
    print(summarize(rows, a.examples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
