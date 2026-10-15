"""Standalone, inspectable agent loop for reproducible CUDA -> SYCL ports.

The loop is deliberately explicit:

* **State**: goal, CUDA source, iteration count, the full history of tool calls and
  results, the best passing candidate so far, and token/cost usage.
* **Each iteration**: send state + evidence to the model, execute the tool calls it
  makes, append the results.
* **Stop conditions**:
    - ``success``        verify passed AND scan_helpers found zero helpers, both on the
                         current sources, compiled from those same sources
    - ``max_iterations`` return the best passing snapshot so far (if any)
    - ``stuck``          the same failure signature in 3 consecutive iterations
    - ``budget``         token or cost budget exceeded
    - ``model_stopped``  the model ended without tool calls twice in a row, or refused
* **Integrity**: the model cannot choose verify's reference/candidate commands, cannot
  write outside its work dir, and "success" is computed by the loop, not claimed by
  the model.

Everything is written to ``runs/<timestamp>/trace.jsonl`` and ``report.md``.
``--dry-run`` swaps the API model for a scripted fake so the loop runs without a key.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .parsers.c2s import MIGRATED_SUFFIXES
from .result import ToolResult
from .toolchain import run_command, split_command
from .tools import benchmark, compile_sycl, migrate, scan_helpers, verify

DEFAULT_MODEL = "claude-sonnet-5-5"

# USD per million tokens (input, output). Cache writes bill 1.25x input, reads 0.1x input.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-1": (10.0, 50.0),
}

SYSTEM_PROMPT = """\
You are porting CUDA code to idiomatic SYCL 2020 for Intel oneAPI DPC++ (icpx -fsycl).
Work only through the provided tools. Every tool returns JSON evidence:
{tool, status, summary, evidence, limitations}. Base every statement on evidence.

Procedure: migrate -> compile -> fix errors -> scan_helpers -> refactor to idiomatic SYCL
(remove every dpct:: helper, DPCT_* macro and #include <dpct/...>) -> compile -> verify
-> benchmark -> finish.

Rules:
- Do not write CUDA idioms (warp size 32, error-code plumbing, 3-D launches for 1-D work).
- Fix the first compiler error first. Keep edits small and targeted.
- write_file replaces a whole file: always send the complete file content.
- The loop declares success only when verify passes AND scan_helpers reports zero helpers,
  both on the current sources and after a successful compile of those sources.
- Never widen tolerances (you cannot; they are fixed by the task).
- If the same error repeats, change approach instead of retrying the same edit.
- If a tool reports toolchain_missing, say so and call finish; do not invent results.
"""


# ----------------------------------------------------------------------------- config


@dataclass
class LoopConfig:
    goal: str
    cuda_path: str  # absolute
    reference_cmd: str  # verify reference template ({input}, {output}); run in work dir
    candidate_cmd: str  # verify candidate template; run in work dir
    inputs: list[str] = field(default_factory=list)  # absolute
    reference_build: list[str] | None = None  # argv run once before the loop (in work dir)
    candidate_binary: str = "build/candidate"  # compile output, relative to work dir
    base_flags: list[str] = field(default_factory=lambda: ["-O2"])
    extra_include_dirs: list[str] = field(default_factory=list)  # absolute
    tolerance: float = 1e-5
    rtol: float = 0.0
    mode: str = "auto"
    threshold_dbfs: float = -96.0
    benchmark_runs: int = 5
    max_iterations: int = 12
    max_total_tokens: int = 600_000
    max_cost_usd: float = 5.0
    stuck_threshold: int = 3
    model: str = field(default_factory=lambda: os.environ.get("ONEAPI_AGENT_MODEL", DEFAULT_MODEL))
    name: str = "task"

    @classmethod
    def from_task_file(cls, path: str | Path, **overrides: Any) -> LoopConfig:
        """Load examples/*/task.json; relative paths resolve against the task file's dir."""
        p = Path(path).resolve()
        data = json.loads(p.read_text())
        base = p.parent

        def absolute(x: str) -> str:
            return str((base / x).resolve())

        cfg = cls(
            goal=data["goal"],
            cuda_path=absolute(data["cuda_path"]),
            reference_cmd=data["reference_cmd"],
            candidate_cmd=data["candidate_cmd"],
            inputs=[absolute(i) for i in data.get("inputs", [])],
            reference_build=[
                a.replace("{task_dir}", str(base)) for a in data["reference_build"]
            ] if data.get("reference_build") else None,
            candidate_binary=data.get("candidate_binary", "build/candidate"),
            base_flags=data.get("base_flags", ["-O2"]),
            extra_include_dirs=[absolute(d) for d in data.get("include_dirs", [])],
            tolerance=data.get("tolerance", 1e-5),
            rtol=data.get("rtol", 0.0),
            mode=data.get("mode", "auto"),
            threshold_dbfs=data.get("threshold_dbfs", -96.0),
            benchmark_runs=data.get("benchmark_runs", 5),
            name=data.get("name", p.parent.name),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


# ----------------------------------------------------------------------------- model


@dataclass
class ModelResponse:
    content: list[dict[str, Any]]  # API-shaped content blocks, passed back verbatim
    stop_reason: str
    usage: dict[str, int]
    model: str = ""

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]

    @property
    def text(self) -> str:
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text")


class ModelClient(Protocol):
    name: str

    def create(self, *, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]]) -> ModelResponse: ...


class AnthropicModel:
    """Claude via the Anthropic Python SDK (manual tool-use loop)."""

    def __init__(self, model: str, max_tokens: int = 16000) -> None:
        import anthropic  # optional dependency: pip install .[agent]

        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY (or other SDK credential sources)
        self.name = model
        self.max_tokens = max_tokens
        self.effort = os.environ.get("ONEAPI_AGENT_EFFORT")  # low|medium|high|xhigh|max
        # Server-side refusal fallback (re-runs a declined request on another model).
        self.fallbacks = os.environ.get("ONEAPI_AGENT_FALLBACKS", "1") != "0"

    def create(self, *, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]]) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.name,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
            "cache_control": {"type": "ephemeral"},  # history grows append-only: cache it
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        else:
            resp = self.client.messages.create(**kwargs)
        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens or 0,
            "output_tokens": u.output_tokens or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
        content = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
        return ModelResponse(content, resp.stop_reason or "", usage, resp.model)


class ScriptedModel:
    """Deterministic fake model for --dry-run and tests.

    ``script`` is a list of turns; each turn is a list of ``(tool_name, input)`` pairs,
    or a string (a text-only turn). After the script ends, it replies with text only.
    """

    def __init__(self, script: list[list[tuple[str, dict[str, Any]]] | str],
                 tokens_per_turn: tuple[int, int] = (1500, 200)) -> None:
        self.script = list(script)
        self.turn = 0
        self.name = "scripted-dry-run"
        self.tokens_per_turn = tokens_per_turn

    def create(self, *, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]]) -> ModelResponse:
        usage = {"input_tokens": self.tokens_per_turn[0], "output_tokens": self.tokens_per_turn[1],
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        if self.turn >= len(self.script):
            self.turn += 1
            return ModelResponse([{"type": "text", "text": "(script exhausted)"}], "end_turn",
                                 usage, self.name)
        step = self.script[self.turn]
        self.turn += 1
        if isinstance(step, str):
            return ModelResponse([{"type": "text", "text": step}], "end_turn", usage, self.name)
        content: list[dict[str, Any]] = [{"type": "text", "text": f"[scripted turn {self.turn}]"}]
        for i, (name, inp) in enumerate(step):
            content.append({"type": "tool_use", "id": f"toolu_dry_{self.turn:03d}_{i}",
                            "name": name, "input": inp})
        return ModelResponse(content, "tool_use", usage, self.name)


# ----------------------------------------------------------------------------- tools


def tool_schemas() -> list[dict[str, Any]]:
    def obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
        return {"type": "object", "properties": props, "required": required or [],
                "additionalProperties": False}

    return [
        {"name": "migrate", "description": "Run SYCLomatic (c2s) on the task's CUDA source. "
         "Writes migrated files under <work>/<out_subdir> (default 'migrated'). Returns DPCT "
         "warnings with locations.", "input_schema": obj({"out_subdir": {"type": "string"}})},
        {"name": "compile", "description": "Compile SYCL sources (paths relative to the work dir) "
         "with icpx -fsycl into the task's candidate binary. Returns parsed diagnostics. Set "
         "add_dpct_include=false to prove dpct headers are no longer needed.",
         "input_schema": obj({"sources": {"type": "array", "items": {"type": "string"}},
                              "extra_flags": {"type": "array", "items": {"type": "string"}},
                              "add_dpct_include": {"type": "boolean"}}, ["sources"])},
        {"name": "scan_helpers", "description": "Inventory dpct:: helpers, DPCT_* macros and dpct "
         "includes under a path (relative to work dir, default '.'), with suggestions.",
         "input_schema": obj({"path": {"type": "string"}})},
        {"name": "verify", "description": "Run the task's fixed reference and the compiled "
         "candidate on the task inputs and compare outputs (tolerance fixed by the task).",
         "input_schema": obj({})},
        {"name": "benchmark", "description": "Time the candidate on each SYCL device.",
         "input_schema": obj({"runs": {"type": "integer", "minimum": 1, "maximum": 50}})},
        {"name": "read_file", "description": "Read a text file under the work dir (or the task's "
         "CUDA source).", "input_schema": obj({"path": {"type": "string"}}, ["path"])},
        {"name": "write_file", "description": "Create/replace a whole text file under the work dir.",
         "input_schema": obj({"path": {"type": "string"}, "content": {"type": "string"}},
                             ["path", "content"])},
        {"name": "list_files", "description": "List files under a work-dir path.",
         "input_schema": obj({"path": {"type": "string"}})},
        {"name": "finish", "description": "Declare you are done. The loop checks the evidence "
         "itself; if success conditions are not met, the loop continues or stops.",
         "input_schema": obj({"summary": {"type": "string"}}, ["summary"])},
    ]


ToolFn = Callable[[dict[str, Any]], ToolResult]


class Toolbox:
    """Executes model tool calls against the real tools, confined to the work dir."""

    def __init__(self, cfg: LoopConfig, work: Path) -> None:
        self.cfg = cfg
        self.work = work

    def _inside(self, rel: str, *, allow_task_source: bool = False) -> Path:
        p = (self.work / rel).resolve()
        if p == self.work or self.work in p.parents:
            return p
        if allow_task_source:
            src = Path(self.cfg.cuda_path).resolve()
            q = Path(rel).resolve()
            if q == src or (src.is_dir() and src in q.parents):
                return q
        raise PermissionError(f"path escapes the work dir: {rel}")

    def __call__(self, name: str, args: dict[str, Any]) -> ToolResult:
        try:
            fn = getattr(self, f"t_{name}")
        except AttributeError:
            return ToolResult(name, "error", f"unknown tool {name!r}")
        try:
            return fn(args)
        except PermissionError as exc:
            return ToolResult(name, "error", str(exc))
        except (OSError, ValueError, TypeError) as exc:
            return ToolResult(name, "error", f"{type(exc).__name__}: {exc}")

    def t_migrate(self, a: dict[str, Any]) -> ToolResult:
        out = self._inside(a.get("out_subdir") or "migrated")
        extra = [f"-I{d}" for d in self.cfg.extra_include_dirs]
        return migrate(self.cfg.cuda_path, str(out), extra_args=extra)

    def t_compile(self, a: dict[str, Any]) -> ToolResult:
        srcs = [str(self._inside(s)) for s in a.get("sources", [])]
        out = self._inside(self.cfg.candidate_binary)
        out.parent.mkdir(parents=True, exist_ok=True)
        flags = [*self.cfg.base_flags, *[f"-I{d}" for d in self.cfg.extra_include_dirs],
                 *a.get("extra_flags", [])]
        return compile_sycl(srcs, flags, output=str(out),
                            add_dpct_include=a.get("add_dpct_include", True))

    def t_scan_helpers(self, a: dict[str, Any]) -> ToolResult:
        return scan_helpers(str(self._inside(a.get("path") or ".")))

    def t_verify(self, a: dict[str, Any]) -> ToolResult:
        return verify(self.cfg.reference_cmd, self.cfg.candidate_cmd, self.cfg.inputs,
                      self.cfg.tolerance, rtol=self.cfg.rtol, mode=self.cfg.mode,
                      threshold_dbfs=self.cfg.threshold_dbfs, cwd=str(self.work),
                      work_dir=str(self.work / "verify_out"))

    def t_benchmark(self, a: dict[str, Any]) -> ToolResult:
        argv = split_command(self.cfg.candidate_cmd)
        first = self.cfg.inputs[0] if self.cfg.inputs else ""
        out = str(self.work / "verify_out" / "bench_out")
        argv = [x.replace("{input0}", first).replace("{input}", first).replace("{output}", out)
                for x in argv]
        (self.work / "verify_out").mkdir(exist_ok=True)
        return benchmark(argv, int(a.get("runs") or self.cfg.benchmark_runs), cwd=str(self.work))

    def t_read_file(self, a: dict[str, Any]) -> ToolResult:
        p = self._inside(a["path"], allow_task_source=True)
        text = p.read_text(errors="replace")
        return ToolResult("read_file", "ok", f"{p.name}: {len(text)} chars",
                          {"path": str(p), "content": text})

    def t_write_file(self, a: dict[str, Any]) -> ToolResult:
        p = self._inside(a["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(a["content"])
        return ToolResult("write_file", "ok", f"wrote {p.relative_to(self.work)} "
                          f"({len(a['content'])} chars)", {"path": str(p)})

    def t_list_files(self, a: dict[str, Any]) -> ToolResult:
        root = self._inside(a.get("path") or ".")
        files = sorted(str(p.relative_to(self.work)) for p in root.rglob("*") if p.is_file())
        return ToolResult("list_files", "ok", f"{len(files)} file(s)", {"files": files[:500]})


class MockToolbox(Toolbox):
    """Toolbox whose heavy tools return canned results (dry-run without a toolchain).

    File tools stay real (inside the work dir) so source hashing behaves normally.
    ``queues`` maps tool name -> list of ToolResults consumed in order (last one repeats).
    """

    def __init__(self, cfg: LoopConfig, work: Path, queues: dict[str, list[ToolResult]]) -> None:
        super().__init__(cfg, work)
        self.queues = {k: list(v) for k, v in queues.items()}

    def _next(self, name: str) -> ToolResult:
        q = self.queues.get(name)
        if not q:
            return ToolResult(name, "error", f"mock: no canned result for {name}")
        r = q.pop(0) if len(q) > 1 else q[0]
        limitations = [*r.limitations, "MOCK RESULT (dry-run --mock-tools): not a real tool run."]
        return ToolResult(r.tool, r.status, r.summary, dict(r.evidence), limitations)

    def t_migrate(self, a: dict[str, Any]) -> ToolResult:
        return self._next("migrate")

    def t_compile(self, a: dict[str, Any]) -> ToolResult:
        return self._next("compile")

    def t_scan_helpers(self, a: dict[str, Any]) -> ToolResult:
        return self._next("scan_helpers")

    def t_verify(self, a: dict[str, Any]) -> ToolResult:
        return self._next("verify")

    def t_benchmark(self, a: dict[str, Any]) -> ToolResult:
        return self._next("benchmark")


# ----------------------------------------------------------------------------- state


@dataclass
class ToolEvent:
    iteration: int
    name: str
    input: dict[str, Any]
    result: ToolResult
    source_hash: str


@dataclass
class LoopState:
    goal: str
    cuda_source: str
    iteration: int = 0
    history: list[ToolEvent] = field(default_factory=list)
    best: dict[str, Any] | None = None  # {iteration, source_hash, helpers, path, verify_summary}
    usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})
    cost_usd: float | None = 0.0
    iteration_signatures: list[str | None] = field(default_factory=list)
    last_compile_ok_hash: str | None = None
    last_verify_ok_hash: str | None = None
    last_scan_zero_hash: str | None = None
    stop_reason: str | None = None
    stop_detail: str = ""


def source_hash(work: Path) -> str:
    """Hash of every source file in the work dir (binaries and outputs excluded)."""
    h = hashlib.sha256()
    for p in sorted(work.rglob("*")):
        if p.is_file() and p.suffix in MIGRATED_SUFFIXES and "verify_out" not in p.parts \
                and "best" not in p.parts:
            h.update(str(p.relative_to(work)).encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()[:16]


def failure_signature(r: ToolResult) -> str | None:
    """Stable fingerprint of a failure, used for stuck detection."""
    if r.status == "ok" or r.tool in ("read_file", "list_files", "write_file", "finish"):
        return None
    if r.tool == "scan_helpers" and r.status == "fail":
        return None  # helpers remaining is a work list, not an error
    if r.tool == "compile":
        diags = [d for d in r.evidence.get("diagnostics", []) if d.get("severity") in
                 ("error", "fatal error")]
        if diags:
            d = diags[0]
            return f"compile:{Path(d.get('file') or '?').name}:{d.get('line')}:{d.get('message')}"
    if r.tool == "verify":
        cases = r.evidence.get("cases", [])
        bad = next((c for c in cases if not c.get("passed")), None)
        if bad:
            return f"verify:{bad.get('reason')}"
    return f"{r.tool}:{r.status}:{r.summary[:160]}"


def estimate_cost(model: str, usage: dict[str, int]) -> float | None:
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return None
    pin, pout = price
    return (usage["input_tokens"] * pin
            + usage["cache_creation_input_tokens"] * pin * 1.25
            + usage["cache_read_input_tokens"] * pin * 0.10
            + usage["output_tokens"] * pout) / 1e6


# ----------------------------------------------------------------------------- loop


class AgentLoop:
    def __init__(self, cfg: LoopConfig, model: ModelClient, *, runs_dir: str | Path = "runs",
                 toolbox_factory: Callable[[LoopConfig, Path], Toolbox] | None = None,
                 run_name: str | None = None) -> None:
        self.cfg = cfg
        self.model = model
        stamp = run_name or _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_dir = Path(runs_dir).resolve() / stamp
        self.work = self.run_dir / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.toolbox = (toolbox_factory or Toolbox)(cfg, self.work)
        self.trace_path = self.run_dir / "trace.jsonl"
        src = Path(cfg.cuda_path)
        cuda_text = src.read_text(errors="replace") if src.is_file() else \
            "\n".join(f"// {p}\n{p.read_text(errors='replace')}" for p in sorted(src.rglob("*.cu")))
        self.state = LoopState(goal=cfg.goal, cuda_source=cuda_text)
        self.messages: list[dict[str, Any]] = []
        self._text_only_turns = 0

    # -- tracing
    def _trace(self, kind: str, **data: Any) -> None:
        rec = {"ts": _dt.datetime.now(_dt.UTC).isoformat(), "type": kind, **data}
        with self.trace_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # -- prompts
    def _initial_message(self) -> str:
        c = self.cfg
        return (
            f"GOAL: {c.goal}\n\n"
            f"CUDA source ({c.cuda_path}):\n```cuda\n{self.state.cuda_source}\n```\n\n"
            f"Work dir: all paths are relative to it. Compile output goes to "
            f"{c.candidate_binary}.\nVerify compares `{c.candidate_cmd}` against the reference "
            f"`{c.reference_cmd}` on {len(c.inputs)} input(s); mode={c.mode}, "
            f"atol={c.tolerance}, rtol={c.rtol}, audio threshold={c.threshold_dbfs} dBFS.\n"
            f"Budget: {c.max_iterations} iterations, {c.max_total_tokens} tokens, "
            f"${c.max_cost_usd:.2f}.\nStart with migrate."
        )

    def _state_note(self) -> str:
        s, c = self.state, self.cfg
        cur = source_hash(self.work)
        best = (f"iteration {s.best['iteration']} ({s.best['helpers']} helpers)" if s.best
                else "none yet")
        return (
            f"[loop state] iteration {s.iteration}/{c.max_iterations}; sources {cur}; "
            f"compiled-from-current={'yes' if s.last_compile_ok_hash == cur else 'no'}; "
            f"verify-passed-on-current={'yes' if s.last_verify_ok_hash == cur else 'no'}; "
            f"zero-helpers-on-current={'yes' if s.last_scan_zero_hash == cur else 'no'}; "
            f"best passing: {best}; tokens used {s.usage['input_tokens'] + s.usage['output_tokens']}"
        )

    # -- bookkeeping after each tool call
    def _record(self, name: str, args: dict[str, Any], result: ToolResult) -> None:
        h = source_hash(self.work)
        s = self.state
        s.history.append(ToolEvent(s.iteration, name, args, result, h))
        if name == "compile":
            s.last_compile_ok_hash = h if result.ok else None
        elif name == "verify" and result.ok and s.last_compile_ok_hash == h:
            s.last_verify_ok_hash = h
            self._maybe_snapshot(h, result)
        elif name == "verify":
            if not result.ok:
                s.last_verify_ok_hash = None
        elif name == "scan_helpers":
            whole_tree = args.get("path") in (None, "", ".")
            if result.ok and whole_tree:
                s.last_scan_zero_hash = h
                if s.best and s.best["source_hash"] == h:
                    s.best["helpers"] = 0
        self._trace("tool_result", iteration=s.iteration, name=name, input=args,
                    source_hash=h, result=result.to_dict())

    def _maybe_snapshot(self, h: str, verify_result: ToolResult) -> None:
        """Keep the best passing candidate: passing verify, fewest remaining helpers."""
        scan = scan_helpers(str(self.work))
        helpers = int(scan.evidence.get("total", 0)) if scan.status in ("ok", "fail") else 10**9
        best = self.state.best
        if best is None or helpers <= best["helpers"]:
            dest = self.run_dir / "best"
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            for p in self.work.rglob("*"):
                if p.is_file() and p.suffix in MIGRATED_SUFFIXES and "verify_out" not in p.parts:
                    tgt = dest / p.relative_to(self.work)
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, tgt)
            self.state.best = {"iteration": self.state.iteration, "source_hash": h,
                               "helpers": helpers, "path": str(dest),
                               "verify_summary": verify_result.summary}
            self._trace("best_snapshot", **self.state.best)

    def success(self) -> bool:
        s = self.state
        cur = source_hash(self.work)
        return (s.last_compile_ok_hash == cur and s.last_verify_ok_hash == cur
                and s.last_scan_zero_hash == cur)

    def _over_budget(self) -> str | None:
        s, c = self.state, self.cfg
        tokens = (s.usage["input_tokens"] + s.usage["output_tokens"]
                  + s.usage["cache_creation_input_tokens"] + s.usage["cache_read_input_tokens"])
        if tokens > c.max_total_tokens:
            return f"token budget exceeded: {tokens} > {c.max_total_tokens}"
        if s.cost_usd is not None and s.cost_usd > c.max_cost_usd:
            return f"cost budget exceeded: ${s.cost_usd:.4f} > ${c.max_cost_usd:.2f}"
        return None

    def _stop(self, reason: str, detail: str = "") -> None:
        self.state.stop_reason = reason
        self.state.stop_detail = detail

    # -- main
    def prepare(self) -> bool:
        """Build the reference once (harness-side, not model-controlled)."""
        if not self.cfg.reference_build:
            return True
        (self.work / "build").mkdir(exist_ok=True)
        res = run_command(self.cfg.reference_build, cwd=str(self.work), timeout=600)
        self._trace("reference_build", run=res.evidence())
        if res.not_found or res.returncode != 0:
            self._stop("error", f"reference build failed: {res.command} -> "
                                f"{res.returncode} {res.stderr[-500:]}")
            return False
        return True

    def run(self) -> dict[str, Any]:
        c = self.cfg
        self._trace("start", config={k: v for k, v in c.__dict__.items()}, model=self.model.name,
                    run_dir=str(self.run_dir))
        if not self.prepare():
            return self.finish()
        self.messages = [{"role": "user", "content": self._initial_message()}]
        tools = tool_schemas()

        while self.state.stop_reason is None:
            if self.state.iteration >= c.max_iterations:
                self._stop("max_iterations",
                           f"reached {c.max_iterations} iterations without meeting success "
                           "conditions")
                break
            self.state.iteration += 1
            it = self.state.iteration
            resp = self.model.create(system=SYSTEM_PROMPT, messages=self.messages, tools=tools)
            for k in self.state.usage:
                self.state.usage[k] += int(resp.usage.get(k, 0))
            # Scripted dry runs cost nothing; unknown models -> None (token budget still applies).
            self.state.cost_usd = 0.0 if isinstance(self.model, ScriptedModel) \
                else estimate_cost(self.model.name, self.state.usage)
            self._trace("model_response", iteration=it, stop_reason=resp.stop_reason,
                        content=resp.content, usage=resp.usage, model=resp.model)
            self.messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "refusal":
                self._stop("model_stopped", "model refused (stop_reason=refusal)")
                break

            uses = resp.tool_uses
            if not uses:
                self._text_only_turns += 1
                self.state.iteration_signatures.append(None)
                if self.success():
                    self._stop("success", "success conditions met")
                    break
                if self._text_only_turns >= 2:
                    self._stop("model_stopped", "model ended twice without tool calls")
                    break
                self.messages.append({"role": "user", "content": [
                    {"type": "text", "text": "Success conditions are not met yet. Continue "
                     "with tool calls, or call finish if you are blocked.\n" + self._state_note()}
                ]})
                continue
            self._text_only_turns = 0

            results_blocks: list[dict[str, Any]] = []
            iter_sig: str | None = None
            finished = None
            for u in uses:
                name, args = u["name"], u.get("input") or {}
                self._trace("tool_call", iteration=it, name=name, input=args)
                if name == "finish":
                    finished = args.get("summary", "")
                    ok = self.success()
                    result = ToolResult("finish", "ok" if ok else "fail",
                                        "accepted: success conditions met" if ok else
                                        "rejected: verify+zero-helpers not both satisfied on "
                                        "current compiled sources", {"model_summary": finished})
                    self._trace("tool_result", iteration=it, name=name, input=args,
                                source_hash=source_hash(self.work), result=result.to_dict())
                else:
                    result = self.toolbox(name, args)
                    self._record(name, args, result)
                    sig = failure_signature(result)
                    if sig:
                        iter_sig = sig
                    elif result.ok and name in ("compile", "verify"):
                        iter_sig = None  # a later success in the same iteration clears it
                payload = result.to_json(indent=None)
                if len(payload) > 30_000:
                    payload = payload[:30_000] + '..."[truncated by loop]"'
                results_blocks.append({"type": "tool_result", "tool_use_id": u["id"],
                                       "content": payload,
                                       **({"is_error": True} if result.status == "error" else {})})
            self.state.iteration_signatures.append(iter_sig)
            results_blocks.append({"type": "text", "text": self._state_note()})
            self.messages.append({"role": "user", "content": results_blocks})

            if self.success():
                self._stop("success", "verify passed and zero dpct helpers on the current "
                                      "compiled sources")
                break
            sigs = self.state.iteration_signatures[-c.stuck_threshold:]
            if len(sigs) == c.stuck_threshold and sigs[0] is not None and len(set(sigs)) == 1:
                self._stop("stuck", f"same failure {c.stuck_threshold} iterations in a row: "
                                    f"{sigs[0]}")
                break
            budget = self._over_budget()
            if budget:
                self._stop("budget", budget)
                break
            if finished is not None and any(e.result.status == "toolchain_missing"
                                            for e in self.state.history[-5:]):
                self._stop("blocked", "model finished after toolchain_missing: " + finished)
                break
        return self.finish()

    def _final_benchmark(self) -> None:
        """On success, benchmark the verified binary (harness-side) if not done on this hash."""
        cur = source_hash(self.work)
        last = _last(self.state.history, "benchmark")
        if last is not None and last.source_hash == cur:
            return
        self._trace("tool_call", iteration=self.state.iteration, name="benchmark",
                    input={"runs": self.cfg.benchmark_runs}, by="harness")
        result = self.toolbox("benchmark", {"runs": self.cfg.benchmark_runs})
        self._record("benchmark", {"runs": self.cfg.benchmark_runs, "by": "harness"}, result)

    def finish(self) -> dict[str, Any]:
        s = self.state
        if s.stop_reason == "success":
            self._final_benchmark()
        outcome = {
            "stop_reason": s.stop_reason,
            "stop_detail": s.stop_detail,
            "iterations": s.iteration,
            "usage": s.usage,
            "cost_usd": s.cost_usd,
            "best": s.best,
            "run_dir": str(self.run_dir),
            "trace": str(self.trace_path),
            "report": str(self.run_dir / "report.md"),
            "model": self.model.name,
        }
        self._trace("stop", **outcome)
        (self.run_dir / "report.md").write_text(render_report(self, outcome))
        return outcome


# ----------------------------------------------------------------------------- report


def _last(history: list[ToolEvent], tool: str) -> ToolEvent | None:
    return next((e for e in reversed(history) if e.name == tool), None)


def render_report(loop: AgentLoop, outcome: dict[str, Any]) -> str:
    s, c = loop.state, loop.cfg
    lines = [
        f"# Port report: {c.name}",
        "",
        f"- **Result:** `{outcome['stop_reason']}` — {outcome['stop_detail']}",
        f"- **Model:** `{outcome['model']}`" + (" (scripted dry run — no LLM was called)"
                                                 if isinstance(loop.model, ScriptedModel) else ""),
        f"- **Iterations:** {s.iteration} / {c.max_iterations}",
        f"- **Tokens:** in {s.usage['input_tokens']}, out {s.usage['output_tokens']}, "
        f"cache write {s.usage['cache_creation_input_tokens']}, "
        f"cache read {s.usage['cache_read_input_tokens']}"
        + (f"; est. cost ${s.cost_usd:.4f}" if s.cost_usd else ""),
        "- **Best passing candidate:** "
        + (f"iteration {s.best['iteration']}, {s.best['helpers']} helper(s), `{s.best['path']}`"
           if s.best else "none"),
        f"- **Trace:** `{loop.trace_path}`",
        "",
        "## Iterations",
        "",
        "| it | tool | status | summary |",
        "|----|------|--------|---------|",
    ]
    for e in s.history:
        summ = e.result.summary.replace("|", "\\|").replace("\n", " ")
        if len(summ) > 140:
            summ = summ[:137] + "..."
        lines.append(f"| {e.iteration} | {e.name} | {e.result.status} | {summ} |")

    lines += ["", "## What failed and what fixed it", ""]
    fixes = []
    for i, e in enumerate(s.history):
        if e.name in ("compile", "verify") and e.result.status in ("fail", "error"):
            later_ok = next((f for f in s.history[i + 1:] if f.name == e.name and f.result.ok),
                            None)
            edits = [f for f in s.history[i + 1:(s.history.index(later_ok) if later_ok else None)]
                     if f.name == "write_file"]
            sig = failure_signature(e.result) or e.result.summary
            if later_ok:
                edited = sorted({Path(f.input.get("path", "?")).name for f in edits})
                fixes.append(f"- it {e.iteration}: `{sig}` → fixed by iteration "
                             f"{later_ok.iteration} after editing "
                             f"{', '.join(edited) or '(no file edits)'}")
            else:
                fixes.append(f"- it {e.iteration}: `{sig}` → **not fixed**")
    lines += fixes or ["- (no compile/verify failures)"]

    lines += ["", "## Final evidence", ""]
    for tool in ("migrate", "compile", "scan_helpers", "verify", "benchmark"):
        ev = _last(s.history, tool)
        if ev is None:
            lines.append(f"- **{tool}:** not run")
            continue
        lines.append(f"- **{tool}** (it {ev.iteration}, sources `{ev.source_hash}`): "
                     f"`{ev.result.status}` — {ev.result.summary}")
        for lim in ev.result.limitations:
            if "MOCK" in lim:
                lines.append(f"  - ⚠️ {lim}")
    cur = source_hash(loop.work)
    lines += ["", f"Current sources hash: `{cur}`. Success requires compile, verify and "
              "zero-helper scan all on this hash.", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- dry run


def dry_run_script(scenario: str, cfg: LoopConfig, idiomatic_source: str | None = None
                   ) -> list[list[tuple[str, dict[str, Any]]] | str]:
    """Scripted model turns for --dry-run.

    success: migrate -> compile -> scan -> write idiomatic file -> compile -> scan ->
             verify -> benchmark -> finish
    stuck:   repeats the same failing compile
    max_iterations: keeps alternating edits that never satisfy success
    """
    if scenario == "success":
        sol = idiomatic_source or "// placeholder idiomatic source\nint main() { return 0; }\n"
        # A deliberately broken first refactor (missing <sycl/sycl.hpp>) exercises the
        # compile-fail -> fix path before the correct version is written.
        broken = "\n".join(line for line in sol.splitlines()
                           if "#include <sycl/sycl.hpp>" not in line) + "\n"
        target = "migrated/" + Path(cfg.cuda_path).stem + ".dp.cpp"
        return [
            [("migrate", {})],
            [("list_files", {"path": "migrated"})],
            [("compile", {"sources": [target]})],
            [("scan_helpers", {"path": "migrated"})],
            [("write_file", {"path": target, "content": broken}),
             ("compile", {"sources": [target], "add_dpct_include": False})],
            [("write_file", {"path": target, "content": sol}),
             ("compile", {"sources": [target], "add_dpct_include": False}),
             ("scan_helpers", {})],
            [("verify", {})],
            [("finish", {"summary": "scripted dry run complete"})],
        ]
    if scenario == "stuck":
        bad = "#include <sycl/sycl.hpp>\nint main() { return undeclared_var; }\n"
        return [[("write_file", {"path": "port.cpp", "content": bad}),
                 ("compile", {"sources": ["port.cpp"]})]] * 6
    if scenario == "max_iterations":
        turns: list[list[tuple[str, dict[str, Any]]] | str] = []
        for i in range(50):
            turns.append([("write_file", {"path": "port.cpp", "content": f"// attempt {i}\n"}),
                          ("compile", {"sources": ["port.cpp"]})])
        return turns
    raise ValueError(f"unknown scenario {scenario!r}")


def mock_queues(scenario: str) -> dict[str, list[ToolResult]]:
    """Canned tool results that make each dry-run scenario behave as its name says."""
    ok_compile = ToolResult("compile", "ok", "compiled 1 source(s) (mock)", {"diagnostics": []})
    if scenario == "success":
        return {
            "migrate": [ToolResult("migrate", "ok", "migrated 1 file (mock); 1 DPCT warning",
                                   {"warnings": [{"code": "DPCT1010", "id": 0}]})],
            "compile": [ok_compile,
                        ToolResult("compile", "fail", "compile failed with 1 error(s) (mock)",
                                   {"diagnostics": [{"file": "x.dp.cpp", "line": 1,
                                                     "severity": "error",
                                                     "message": "use of undeclared identifier "
                                                                "'sycl'"}]}),
                        ok_compile],
            "scan_helpers": [
                ToolResult("scan_helpers", "fail", "9 dpct helper use(s) remain (mock)",
                           {"total": 9}),
                ToolResult("scan_helpers", "ok", "no dpct helpers (mock)", {"total": 0}),
            ],
            "verify": [ToolResult("verify", "ok", "PASS: 1/1 case(s) match (mock)",
                                  {"cases": [{"passed": True}]})],
            "benchmark": [ToolResult("benchmark", "ok", "mock timings — not measured",
                                     {"results": []})],
        }
    if scenario == "stuck":
        err = ToolResult("compile", "fail", "compile failed with 1 error(s) at port.cpp:2",
                         {"diagnostics": [{"file": "port.cpp", "line": 2, "column": 21,
                                           "severity": "error",
                                           "message": "use of undeclared identifier "
                                                      "'undeclared_var'"}]})
        return {"compile": [err]}
    if scenario == "max_iterations":
        diags = [ToolResult("compile", "fail", f"compile failed (mock variant {i})",
                            {"diagnostics": [{"file": "port.cpp", "line": i + 1,
                                              "severity": "error", "message": f"error {i}"}]})
                 for i in range(3)]
        return {"compile": diags * 20}
    raise ValueError(f"unknown scenario {scenario!r}")


def default_task_file() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "saxpy" / "task.json"


def cli_entry(args: Any) -> dict[str, Any]:
    task = Path(args.config) if args.config else default_task_file()
    cfg = LoopConfig.from_task_file(task, max_iterations=args.max_iterations)
    if args.dry_run:
        sol_path = task.parent / "sycl" / (Path(cfg.cuda_path).stem + ".cpp")
        sol = sol_path.read_text() if sol_path.is_file() else None
        model: ModelClient = ScriptedModel(dry_run_script(args.scenario, cfg, sol))
        factory = None
        if args.mock_tools:
            queues = mock_queues(args.scenario)

            def factory(c: LoopConfig, w: Path) -> Toolbox:
                return MockToolbox(c, w, queues)

            cfg.reference_build = None  # nothing real is run in fully mocked mode
    else:
        if args.mock_tools:
            raise SystemExit("--mock-tools requires --dry-run")
        model = AnthropicModel(cfg.model)
        factory = None
    loop = AgentLoop(cfg, model, runs_dir=args.runs_dir, toolbox_factory=factory)
    return loop.run()
