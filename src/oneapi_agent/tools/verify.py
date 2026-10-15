"""verify: run a reference and a candidate on the same inputs and compare their outputs.

Command templates may use:
  ``{input}``   -> the current input file (or ``{input0}``, ``{input1}``... for all inputs)
  ``{output}``  -> a fresh output path created by verify for each side
If a command has no ``{output}``, its stdout is parsed as numbers instead.

Modes:
  ``array`` - elementwise numeric comparison (abs/rel tolerance, RMS, f32 ULP stats)
  ``audio`` - WAV null test: residual peak / RMS in dBFS, pass if peak <= threshold_dbfs
  ``auto``  - audio if any input or output is a WAV, else array
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..compare import audio_null_test, compare_arrays
from ..numio import load_output, parse_numbers
from ..result import ToolResult
from ..toolchain import CommandResult, Runner, run_command, split_command


def _render(template: str | Sequence[str], inputs: Sequence[str], current: str | None,
            output: str | None) -> list[str]:
    argv = split_command(template)
    subs: dict[str, str] = {}
    if current is not None:
        subs["input"] = current
    for i, p in enumerate(inputs):
        subs[f"input{i}"] = p
    if output is not None:
        subs["output"] = output
    rendered = []
    for a in argv:
        for k, v in subs.items():
            a = a.replace("{" + k + "}", v)
        rendered.append(a)
    return rendered


def _uses(template: str | Sequence[str], key: str) -> bool:
    return ("{" + key + "}") in " ".join(split_command(template))


def verify(
    reference_cmd: str | Sequence[str],
    candidate_cmd: str | Sequence[str],
    inputs: Sequence[str] = (),
    tolerance: float = 1e-5,
    *,
    rtol: float = 0.0,
    max_ulp: int | None = None,
    mode: str = "auto",
    threshold_dbfs: float = -96.0,
    output_format: str = "auto",
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 600.0,
    work_dir: str | None = None,
    runner: Runner | None = None,
) -> ToolResult:
    """Run both commands for each input and compare. Pass only if every input passes."""
    if mode not in ("auto", "array", "audio"):
        return ToolResult("verify", "error", f"unknown mode {mode!r}")
    run = runner or run_command
    inputs = [str(p) for p in inputs]
    base = Path(cwd) if cwd else Path.cwd()
    missing = [p for p in inputs if not (base / p).exists()]
    if missing:
        return ToolResult("verify", "error", f"input(s) not found: {missing}", {"missing": missing})
    for name, cmd in (("reference", reference_cmd), ("candidate", candidate_cmd)):
        if _uses(cmd, "input") and not inputs:
            return ToolResult("verify", "error", f"{name} command uses {{input}} but no inputs given")

    effective_mode = mode
    if mode == "auto" and any(p.lower().endswith(".wav") for p in inputs):
        effective_mode = "audio"
    out_ext = ".wav" if effective_mode == "audio" else ".out"

    wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="oneapi-verify-"))
    wd.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    limitations: list[str] = [
        "Equivalence is shown only for the inputs tested; other inputs/sizes may behave differently.",
    ]
    all_passed = True
    run_cases = inputs if inputs else [None]
    for idx, current in enumerate(run_cases):
        case: dict[str, Any] = {"input": current}
        sides: dict[str, tuple[CommandResult, str | None]] = {}
        for side, tmpl in (("reference", reference_cmd), ("candidate", candidate_cmd)):
            out_path = str((wd / f"case{idx}_{side}{out_ext}").resolve()) if _uses(tmpl, "output") \
                else None
            argv = _render(tmpl, inputs, current, out_path)
            res = run(argv, cwd=cwd, env=env, timeout=timeout)
            case[side] = res.evidence(limit=4000)
            if out_path:
                case[side]["output_file"] = out_path
            sides[side] = (res, out_path)

        # Execution problems first. Reference problems make the check impossible (error);
        # candidate problems are a verification failure.
        for side in ("reference", "candidate"):
            res, out_path = sides[side]
            problem = None
            if res.not_found:
                problem = f"{side} command not found: {res.argv[0]}"
            elif res.timed_out:
                problem = f"{side} timed out after {timeout}s"
            elif res.returncode != 0:
                problem = f"{side} exited with code {res.returncode}"
            elif out_path and not Path(out_path).exists():
                problem = f"{side} did not write its output file"
            if problem:
                case["passed"] = False
                case["reason"] = problem
                cases.append(case)
                status = "timeout" if res.timed_out else ("error" if side == "reference" else "fail")
                if res.not_found and side == "candidate":
                    status = "error"
                return ToolResult("verify", status, problem,  # type: ignore[arg-type]
                                  {"mode": effective_mode, "cases": cases, "work_dir": str(wd)},
                                  limitations)

        loaded = {}
        for side in ("reference", "candidate"):
            res, out_path = sides[side]
            if out_path:
                try:
                    arr, fmt, meta = load_output(out_path, output_format)
                except (ValueError, OSError) as exc:
                    case.update(passed=False, reason=f"could not read {side} output: {exc}")
                    cases.append(case)
                    return ToolResult("verify", "fail" if side == "candidate" else "error",
                                      case["reason"],
                                      {"mode": effective_mode, "cases": cases, "work_dir": str(wd)},
                                      limitations)
            else:
                arr, fmt, meta = parse_numbers(res.stdout), "stdout-text", {}
            loaded[side] = (arr, fmt, meta)
            case[f"{side}_format"] = fmt

        side_mode = effective_mode
        if mode == "auto" and loaded["reference"][1] == "wav":
            side_mode = "audio"
        if side_mode == "audio" and (loaded["reference"][1] != "wav" or loaded["candidate"][1] != "wav"):
            case.update(passed=False, reason="audio mode requires both outputs to be WAV files")
            all_passed = False
            cases.append(case)
            continue
        if side_mode == "audio":
            stats = audio_null_test(
                loaded["reference"][0], loaded["candidate"][0], threshold_dbfs=threshold_dbfs,
                ref_rate=loaded["reference"][2].get("sample_rate"),
                cand_rate=loaded["candidate"][2].get("sample_rate"),
            )
            for side in ("reference", "candidate"):
                if loaded[side][2].get("format_tag") == 1 and loaded[side][2].get("bits") == 16:
                    msg = (f"{side} output is 16-bit PCM: quantization alone limits the null to "
                           "about -90 dBFS; write 32-bit float WAV for a meaningful -96 dBFS test.")
                    if msg not in limitations:
                        limitations.append(msg)
        else:
            stats = compare_arrays(loaded["reference"][0], loaded["candidate"][0], atol=tolerance,
                                   rtol=rtol, max_ulp=max_ulp)
            if loaded["reference"][1] in ("stdout-text", "text"):
                msg = ("Text outputs are compared as every number found in the text, in order; "
                       "printed labels containing digits are compared too.")
                if msg not in limitations:
                    limitations.append(msg)
        case["mode"] = side_mode
        case["stats"] = stats
        case["passed"] = bool(stats["passed"])
        if not stats["passed"]:
            case["reason"] = stats.get("reason", "comparison failed")
        all_passed &= case["passed"]
        cases.append(case)

    n_pass = sum(1 for c in cases if c.get("passed"))
    evidence = {"mode": effective_mode, "cases": cases, "work_dir": str(wd),
                "passed_cases": n_pass, "total_cases": len(cases)}
    if effective_mode == "audio" or any(c.get("mode") == "audio" for c in cases):
        worst = max(
            (c["stats"].get("residual_peak_dbfs") for c in cases
             if c.get("stats", {}).get("residual_peak_dbfs") is not None),
            default=None,
        )
        evidence["worst_residual_peak_dbfs"] = worst
        detail = ("bit-exact" if worst is None else f"worst residual peak {worst:.1f} dBFS") + \
            f", threshold {threshold_dbfs} dBFS"
    else:
        worst_abs = max((c["stats"].get("max_abs_err") or 0.0 for c in cases if "stats" in c),
                        default=0.0)
        evidence["worst_max_abs_err"] = worst_abs
        detail = f"worst max abs err {worst_abs:.3g}, atol {tolerance:g}"
    if all_passed:
        return ToolResult("verify", "ok", f"PASS: {n_pass}/{len(cases)} case(s) match ({detail})",
                          evidence, limitations)
    first_bad = next(c for c in cases if not c.get("passed"))
    return ToolResult("verify", "fail",
                      f"FAIL: {n_pass}/{len(cases)} case(s) match; first failure: "
                      f"{first_bad.get('reason')}", evidence, limitations)
