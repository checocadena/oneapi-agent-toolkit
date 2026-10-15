#!/usr/bin/env python3
"""With vs without the toolkit's checks: the same model, the same tasks, graded the same way.

  toolkit   normal loop: migrate, compile, scan_helpers, verify, benchmark, file tools.
            The loop declares success only on compile + verify + zero helpers.
  baseline  `--baseline`: migrate, compile and file tools only (what a plain coding agent
            has). The run ends when the model calls finish.

Both arms run with `--evaluate`: after the run, the harness grades the final sources with a
hidden compile + verify + helper scan that the model never sees.

Usage: scripts/ablation.py [--runs 3] [--jobs 4] [--max-iterations 10] [--out docs/ablation]
       [--examples saxpy fuzz fir softmax_hecbench]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARMS = ("toolkit", "baseline")


def run_one(example: str, arm: str, seed: int, max_it: int, runs_root: Path) -> dict:
    task = REPO / "examples" / example / "task.json"
    runs_dir = runs_root / arm / example
    cmd = [sys.executable, "-m", "oneapi_agent.cli", "loop", "--config", str(task),
           "--max-iterations", str(max_it), "--runs-dir", str(runs_dir), "--evaluate"]
    if arm == "baseline":
        cmd.append("--baseline")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    try:
        out = json.loads(p.stdout[p.stdout.index("{"):])
    except ValueError:
        out = {"stop_reason": "harness_error", "stderr": p.stderr[-2000:]}
    ev = out.get("evaluation") or {}
    row = {"example": example, "arm": arm, "seed": seed,
           "stop_reason": out.get("stop_reason"), "iterations": out.get("iterations"),
           "cost_usd": out.get("cost_usd"), "compiled": ev.get("compiled"),
           "verify": ev.get("verify_status"), "verify_summary": ev.get("verify_summary"),
           "helpers": ev.get("helpers"), "note": ev.get("note"),
           "run_dir": out.get("run_dir")}
    row["correct_and_clean"] = bool(row["compiled"] and row["verify"] == "ok"
                                    and row["helpers"] == 0)
    claimed = row["stop_reason"] in ("success", "model_finished")
    row["claimed_done"] = claimed
    row["claimed_but_wrong"] = claimed and not row["correct_and_clean"]
    print(f"[{arm:8}] {example:17} #{seed} stop={row['stop_reason']} verify={row['verify']} "
          f"helpers={row['helpers']} cost=${row['cost_usd'] or 0:.3f}", flush=True)
    return row


def summarize(rows: list[dict], examples: list[str]) -> str:
    def agg(sel: list[dict]) -> str:
        n = len(sel)
        ok = sum(r["correct_and_clean"] for r in sel)
        wrong = sum(r["verify"] == "fail" for r in sel)
        helpers = sum(1 for r in sel if (r["helpers"] or 0) > 0)
        lied = sum(r["claimed_but_wrong"] for r in sel)
        cost = sum(r["cost_usd"] or 0 for r in sel)
        return f"{ok}/{n} | {wrong}/{n} | {helpers}/{n} | {lied}/{n} | ${cost:.2f}"

    md = ["| Example | Arm | correct + 0 helpers | wrong output | dpct helpers left | "
          "said done but wasn't | cost |", "|---|---|---|---|---|---|---|"]
    for ex in [*examples, "**all**"]:
        for arm in ARMS:
            sel = [r for r in rows if r["arm"] == arm and (ex == "**all**" or r["example"] == ex)]
            if sel:
                md.append(f"| {ex} | {arm} | {agg(sel)} |")
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-iterations", type=int, default=10)
    ap.add_argument("--out", default=str(REPO / "docs" / "ablation"))
    ap.add_argument("--examples", nargs="+",
                    default=["saxpy", "fuzz", "fir", "softmax_hecbench"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_root = REPO / "runs" / ("ablation-" + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S"))
    jobs = [(ex, arm, s) for s in range(1, a.runs + 1) for ex in a.examples for arm in ARMS]
    with ThreadPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(lambda j: run_one(*j, a.max_iterations, runs_root), jobs))
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    (out / "SUMMARY.md").write_text(summarize(rows, a.examples) + "\n")
    print(summarize(rows, a.examples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
