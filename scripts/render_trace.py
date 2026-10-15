#!/usr/bin/env python3
"""Render an agent-loop trace (trace.jsonl from oneapi_agent.loop) as a self-contained replay page.

One static HTML file: inline CSS and JS, no network at view time. A presenter steps through the
run one tool call at a time (buttons, arrow keys, PageUp/PageDown for clickers); each step shows
the model's text for that iteration, the tool input, the evidence the tool returned, and for
write_file a unified diff against the previous version of the same file.

Everything taken from the trace is model or tool output, so all of it is HTML-escaped; the page
script only shows and hides elements and never inserts trace text.

Usage:
  scripts/render_trace.py TRACE.jsonl -o out.html [--title TITLE]
  scripts/render_trace.py --site docs/replay/runs.json
      (renders every run listed in the manifest next to it, plus index.html)
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------- loading


def load_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{n}: not JSON: {exc}") from exc
        if isinstance(rec, dict):
            records.append(rec)
    return records


# ----------------------------------------------------------------------------- paths

_REPO_ROOT = re.compile(
    r"/home/[^/\s\"'<>]+/oneapi-agent-toolkit(?:/\.claude/worktrees/[^/\s\"'<>]+)?(/)?")


class PathStripper:
    """Turn absolute paths into the relative ones the agent saw (work dir) or repo-relative."""

    def __init__(self, run_dir: str | None) -> None:
        self.run_dir = run_dir.rstrip("/") if run_dir else None

    def __call__(self, s: str) -> str:
        if self.run_dir:
            s = s.replace(self.run_dir + "/work/", "").replace(self.run_dir + "/work", ".")
        s = _REPO_ROOT.sub(lambda m: "" if m.group(1) else ".", s)
        return s.replace("/root/", "")


# ----------------------------------------------------------------------------- model


@dataclass
class Step:
    index: int
    iteration: int
    tool: str
    input: dict[str, Any]
    by: str | None = None
    result: dict[str, Any] | None = None
    source_hash: str | None = None
    notes: list[str] = field(default_factory=list)
    diff: list[str] | None = None
    diff_base: str | None = None
    kind: str = "tool"  # tool | text | evaluation

    @property
    def status(self) -> str:
        if self.kind == "text":
            return "text"
        if self.result is None:
            return "missing"
        return str(self.result.get("status", "?"))

    @property
    def summary(self) -> str:
        if self.result is None:
            return "(no tool_result recorded)"
        return str(self.result.get("summary", ""))


@dataclass
class Run:
    start: dict[str, Any]
    stop: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    reference_build: dict[str, Any] | None
    reasoning: dict[int, list[str]]
    thinking: dict[int, list[str]]
    steps: list[Step]


def _norm(path: str, strip: PathStripper) -> str:
    p = strip(path)
    return p[2:] if p.startswith("./") else p


def build_run(records: list[dict[str, Any]], strip: PathStripper) -> Run:
    start: dict[str, Any] = {}
    stop = evaluation = reference_build = None
    reasoning: dict[int, list[str]] = {}
    thinking: dict[int, list[str]] = {}
    steps: list[Step] = []
    pending: list[Step] = []
    files: dict[str, str] = {}  # latest known content per (relative) path

    for rec in records:
        kind = rec.get("type")
        it = int(rec.get("iteration") or 0)
        if kind == "start":
            start = rec
        elif kind == "stop":
            stop = rec
        elif kind == "evaluation":
            evaluation = rec
        elif kind == "reference_build":
            reference_build = rec
        elif kind == "model_response":
            content = rec.get("content") or []
            texts = [str(b.get("text", "")) for b in content
                     if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
            thinks = [str(b.get("thinking", "")) for b in content
                      if isinstance(b, dict) and b.get("type") == "thinking" and b.get("thinking")]
            reasoning.setdefault(it, []).extend(texts)
            thinking.setdefault(it, []).extend(thinks)
            if not any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content):
                steps.append(Step(len(steps), it, "(text only)", {}, kind="text"))
        elif kind == "tool_call":
            st = Step(len(steps), it, str(rec.get("name", "?")), dict(rec.get("input") or {}),
                      by=rec.get("by"))
            steps.append(st)
            pending.append(st)
        elif kind == "tool_result":
            name = rec.get("name")
            match = next((s for s in pending if s.tool == name and s.iteration == it), None)
            if match is None:  # result without a call (older traces): make a step for it
                match = Step(len(steps), it, str(name), dict(rec.get("input") or {}))
                steps.append(match)
            else:
                pending.remove(match)
            match.result = rec.get("result") or {}
            match.source_hash = rec.get("source_hash")
            _track_files(match, files, strip)
        elif kind == "best_snapshot":
            target = pending[-1] if pending else (steps[-1] if steps else None)
            if target is not None:
                target.notes.append(
                    f"Loop saved a best-passing snapshot (iteration {rec.get('iteration')}, "
                    f"{rec.get('helpers')} helper(s)): {rec.get('verify_summary', '')}")

    if evaluation is not None:
        ok = evaluation.get("verify_status") == "ok" and evaluation.get("helpers") == 0
        steps.append(Step(len(steps), int((stop or {}).get("iterations") or 0), "hidden check",
                          {}, result={"status": "ok" if ok else "fail",
                                      "summary": _evaluation_line(evaluation),
                                      "evidence": {k: v for k, v in evaluation.items()
                                                   if k not in ("ts", "type")}},
                          kind="evaluation"))
    return Run(start, stop, evaluation, reference_build, reasoning, thinking, steps)


def _track_files(step: Step, files: dict[str, str], strip: PathStripper) -> None:
    res = step.result or {}
    if step.tool == "read_file" and res.get("status") == "ok":
        ev = res.get("evidence") or {}
        path = ev.get("path") or step.input.get("path")
        if isinstance(path, str) and isinstance(ev.get("content"), str):
            files.setdefault(_norm(path, strip), ev["content"])
    elif step.tool == "write_file" and isinstance(step.input.get("content"), str):
        path = _norm(str(step.input.get("path", "?")), strip)
        new = step.input["content"]
        old = files.get(path)
        step.diff_base = "previous version in this trace" if old is not None else \
            "empty (no earlier version of this file appears in the trace)"
        step.diff = list(difflib.unified_diff(
            (old or "").splitlines(), new.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=3))
        if res.get("status", "ok") == "ok":
            files[path] = new


def _evaluation_line(ev: dict[str, Any]) -> str:
    if not ev.get("compiled"):
        return "did not compile: " + str(ev.get("note", ""))
    status = "PASS" if ev.get("verify_status") == "ok" else "FAIL"
    return f"verify {status} ({ev.get('verify_summary')}); dpct helpers: {ev.get('helpers')}"


# ----------------------------------------------------------------------------- html helpers


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def pretty(obj: Any, strip: PathStripper) -> str:
    return esc(strip(json.dumps(obj, indent=2, ensure_ascii=False, default=str)))


_STATUS_CLASS = {"ok": "ok", "fail": "fail", "error": "err", "toolchain_missing": "err",
                 "missing": "err", "text": "muted"}


def badge(status: str, label: str | None = None) -> str:
    cls = _STATUS_CLASS.get(status, "muted")
    return f'<span class="badge {cls}">{esc(label or status)}</span>'


def render_diff(lines: list[str]) -> str:
    out = []
    for line in lines:
        if line.startswith(("+++", "---")):
            cls = "dh"
        elif line.startswith("@@"):
            cls = "dk"
        elif line.startswith("+"):
            cls = "da"
        elif line.startswith("-"):
            cls = "dd"
        else:
            cls = "dc"
        out.append(f'<span class="{cls}">{esc(line) or " "}</span>')
    return '<pre class="diff">' + "".join(out) + "</pre>"


def _money(v: Any) -> str:
    return f"${v:.4f}" if isinstance(v, int | float) else "n/a"


def _tokens(usage: dict[str, Any] | None) -> str:
    if not usage:
        return "n/a"
    return f"{sum(int(v or 0) for v in usage.values()):,}"


def verdict(run: Run) -> tuple[str, str]:
    """(css class, text) for the loop's own outcome."""
    stop = run.stop or {}
    reason = stop.get("stop_reason")
    if reason == "success":
        return "ok", "Loop-verified success"
    if reason == "model_finished":
        return "warn", "Agent declared itself done"
    if reason is None:
        return "muted", "Run did not record a stop"
    return "fail", f"Stopped without success: {reason}"


# ----------------------------------------------------------------------------- page

CSS = """
:root{--bg:#fbfbf8;--fg:#1b1d22;--muted:#5d6370;--card:#ffffff;--line:#d9dbe0;--accent:#2457c5;
--ok:#137a3a;--okbg:#e2f4e8;--fail:#b3261e;--failbg:#fbe4e2;--err:#9a5b00;--errbg:#fdf0d9;
--warn:#8a4b00;--warnbg:#fff1d6;--sel:#e6edfb;--code:#f3f4f6;--add:#dcf5e3;--del:#fde2e0;--hunk:#5b4bb7}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121418;--fg:#e8eaee;
--muted:#9aa1ad;--card:#1b1e24;--line:#2e333c;--accent:#7aa5ff;--ok:#5fd38a;--okbg:#173322;
--fail:#ff8a80;--failbg:#3a1a19;--err:#ffc261;--errbg:#3a2a10;--warn:#ffc261;--warnbg:#3a2a10;
--sel:#23314d;--code:#15171c;--add:#173322;--del:#3a1a19;--hunk:#b3a6ff}}
:root[data-theme=dark]{--bg:#121418;--fg:#e8eaee;--muted:#9aa1ad;--card:#1b1e24;--line:#2e333c;
--accent:#7aa5ff;--ok:#5fd38a;--okbg:#173322;--fail:#ff8a80;--failbg:#3a1a19;--err:#ffc261;
--errbg:#3a2a10;--warn:#ffc261;--warnbg:#3a2a10;--sel:#23314d;--code:#15171c;--add:#173322;
--del:#3a1a19;--hunk:#b3a6ff}
*{box-sizing:border-box}
html{font-size:20px}
body{margin:0;background:var(--bg);color:var(--fg);font:1rem/1.45 system-ui,-apple-system,
"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:.7rem .9rem;
overflow:auto;font-size:.78rem;line-height:1.4;margin:.4rem 0;max-height:60vh;white-space:pre-wrap;
overflow-wrap:anywhere}
header{padding:.7rem 1.5rem;border-bottom:1px solid var(--line);background:var(--card)}
header h1{font-size:1.45rem;margin:0 0 .15rem}
.goal{color:var(--muted);margin:0 0 .4rem;max-width:110ch;font-size:.9rem}
.banners{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(22rem,100%),1fr));gap:.6rem;
margin-top:.5rem}
.facts{display:flex;flex-wrap:wrap;gap:.4rem 1.6rem;margin:0;font-size:.9rem}
.facts div{display:flex;gap:.4rem}.facts dt{color:var(--muted)}.facts dd{margin:0;font-weight:600}
.banner{margin:0;padding:.45rem .9rem;border-radius:8px;font-weight:600;border:1px solid}
.banner.ok{background:var(--okbg);color:var(--ok);border-color:var(--ok)}
.banner.fail{background:var(--failbg);color:var(--fail);border-color:var(--fail)}
.banner.warn{background:var(--warnbg);color:var(--warn);border-color:var(--warn)}
.banner.muted{color:var(--muted);border-color:var(--line)}
.banner small{display:block;font-weight:500;font-size:.8rem;opacity:.9;text-transform:uppercase;
letter-spacing:.06em}
main{display:grid;grid-template-columns:minmax(17rem,26rem) 1fr;gap:0;min-height:0}
nav{border-right:1px solid var(--line);padding:.6rem .5rem;position:sticky;top:0;align-self:start;
max-height:100vh;overflow:auto}
nav ol{list-style:none;margin:0;padding:0}
nav li a{display:grid;grid-template-columns:2.2rem 1fr auto;gap:.15rem .5rem;padding:.4rem .5rem;
border-radius:8px;text-decoration:none;color:var(--fg);font-size:.82rem;border:2px solid transparent}
nav li a:hover{background:var(--sel)}
nav li a.active{background:var(--sel);border-color:var(--accent)}
nav .n{color:var(--muted);font-variant-numeric:tabular-nums}
nav .t{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
nav .s{grid-column:2/4;color:var(--muted);font-size:.74rem;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}
.badge{display:inline-block;padding:.05rem .5rem;border-radius:999px;font-size:.72rem;
font-weight:700;text-transform:uppercase;letter-spacing:.04em;border:1px solid;align-self:center}
.badge.ok{color:var(--ok);background:var(--okbg)}.badge.fail{color:var(--fail);background:var(--failbg)}
.badge.err{color:var(--err);background:var(--errbg)}.badge.muted{color:var(--muted)}
.stage{padding:1rem 1.5rem 3rem;min-width:0}
.controls{display:flex;align-items:center;gap:.8rem;margin-bottom:.8rem;position:sticky;top:0;
background:var(--bg);padding:.4rem 0;z-index:2}
.controls button{font:inherit;font-size:.9rem;padding:.35rem 1rem;border-radius:8px;
border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.controls button:hover{border-color:var(--accent)}
.controls .pos{font-variant-numeric:tabular-nums;color:var(--muted)}
.bar{flex:1;height:6px;background:var(--line);border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent);width:0}
.hint{font-size:.72rem;color:var(--muted)}
.step h2{font-size:1.35rem;margin:.2rem 0 .3rem;display:flex;gap:.6rem;align-items:center;
flex-wrap:wrap}
.step h2 .meta{color:var(--muted);font-weight:500;font-size:1rem}
.summary{font-size:1.1rem;margin:.2rem 0 1rem;font-weight:500}
.summary.ok{color:var(--ok)}.summary.fail{color:var(--fail)}.summary.err{color:var(--err)}
.step h3{font-size:.85rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
margin:1.2rem 0 .3rem}
.reason{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);
border-radius:8px;padding:.6rem .9rem;white-space:pre-wrap;max-width:90ch}
.reason.none{color:var(--muted);font-style:italic;border-left-color:var(--line)}
.claim{background:var(--warnbg);border-left-color:var(--warn)}
.note,.limit{padding:.45rem .8rem;border-radius:8px;margin:.3rem 0;max-width:90ch}
.note{background:var(--okbg);color:var(--ok)}
.limit{background:var(--warnbg);color:var(--warn)}
details{margin:.3rem 0}
summary{cursor:pointer;color:var(--accent);font-size:.9rem}
.diff .da{background:var(--add);display:block}.diff .dd{background:var(--del);display:block}
.diff .dk{color:var(--hunk);display:block}.diff .dh{color:var(--muted);font-weight:700;display:block}
.diff .dc{display:block}
.diff{white-space:pre;overflow-wrap:normal}
.hidden-check{border:3px solid var(--fail);border-radius:12px;padding:.8rem 1rem;background:var(--failbg)}
.hidden-check.ok{border-color:var(--ok);background:var(--okbg)}
.js .step:not(.current){display:none}
.step{scroll-margin-top:4rem}
.nojs-only{display:block}.js .nojs-only{display:none}
footer{color:var(--muted);font-size:.75rem;padding:1rem 1.5rem;border-top:1px solid var(--line)}
@media (max-width:900px){html{font-size:17px}main{grid-template-columns:1fr}
nav{position:static;max-height:40vh;border-right:0;border-bottom:1px solid var(--line)}
.hint{display:none}.controls{gap:.5rem}.controls button{padding:.3rem .6rem;white-space:nowrap}
.controls .pos{white-space:nowrap}
.stage,header{padding-left:16px;padding-right:16px}}
"""

JS = """
(function(){
  document.documentElement.classList.add('js');
  var steps=[].slice.call(document.querySelectorAll('.step'));
  var links=[].slice.call(document.querySelectorAll('nav a[data-i]'));
  var pos=document.getElementById('pos'), bar=document.getElementById('bar');
  var cur=-1;
  function show(i, push){
    if(!steps.length) return;
    i=Math.max(0,Math.min(steps.length-1,i));
    if(cur>=0){steps[cur].classList.remove('current');links[cur].classList.remove('active');}
    cur=i; steps[i].classList.add('current'); links[i].classList.add('active');
    pos.textContent='Step '+(i+1)+' of '+steps.length;
    bar.style.width=((i+1)/steps.length*100)+'%';
    links[i].scrollIntoView({block:'nearest'});
    if(push!==false && history.replaceState) history.replaceState(null,'','#step-'+(i+1));
  }
  function fromHash(){
    var m=/^#step-(\\d+)$/.exec(location.hash); return m?parseInt(m[1],10)-1:0;
  }
  document.getElementById('prev').onclick=function(){show(cur-1)};
  document.getElementById('next').onclick=function(){show(cur+1)};
  links.forEach(function(a){a.addEventListener('click',function(e){
    e.preventDefault(); show(parseInt(a.getAttribute('data-i'),10)); window.scrollTo(0,0);});});
  document.addEventListener('keydown',function(e){
    if(e.altKey||e.ctrlKey||e.metaKey) return;
    var k=e.key;
    if(k==='ArrowRight'||k==='ArrowDown'||k==='PageDown'||k==='j'){show(cur+1);e.preventDefault();}
    else if(k==='ArrowLeft'||k==='ArrowUp'||k==='PageUp'||k==='k'){show(cur-1);e.preventDefault();}
    else if(k==='Home'){show(0);e.preventDefault();}
    else if(k==='End'){show(steps.length-1);e.preventDefault();}
    else if(k==='t'){var r=document.documentElement, d=r.getAttribute('data-theme');
      var dark=d?d==='dark':matchMedia('(prefers-color-scheme: dark)').matches;
      r.setAttribute('data-theme',dark?'light':'dark');}
  });
  window.addEventListener('hashchange',function(){show(fromHash(),false);window.scrollTo(0,0);});
  show(fromHash(),false);
})();
"""


def _page(title: str, body: str, js: str = "") -> str:
    script = f"<script>{js}</script>" if js else ""
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{esc(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n{body}\n"
        f"{script}\n</body>\n</html>\n"
    )


def render_header(run: Run, title: str, strip: PathStripper) -> str:
    cfg = run.start.get("config") or {}
    stop = run.stop or {}
    goal = cfg.get("goal", "")
    arm = "baseline (no toolkit checks)" if (stop.get("baseline") or cfg.get("baseline")) \
        else "toolkit"
    facts = [
        ("Task", cfg.get("name", "?")),
        ("Arm", arm),
        ("Model", stop.get("model") or run.start.get("model", "?")),
        ("Stop", stop.get("stop_reason", "?")),
        ("Iterations", stop.get("iterations", "?")),
        ("Tool calls", sum(1 for s in run.steps if s.kind == "tool")),
        ("Tokens", _tokens(stop.get("usage"))),
        ("Cost", _money(stop.get("cost_usd"))),
    ]
    dl = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(strip(str(v)))}</dd></div>" for k, v in facts)
    cls, text = verdict(run)
    detail = stop.get("stop_detail")
    out = [f"<header><h1>{esc(title)}</h1>",
           f'<p class="goal">{esc(strip(goal))}</p>',
           f'<dl class="facts">{dl}</dl>', '<div class="banners">',
           f'<div class="banner {cls}"><small>Loop outcome</small>{esc(text)}'
           + (f" — {esc(strip(str(detail)))}" if detail else "") + "</div>"]
    if run.evaluation is not None:
        ev = run.evaluation
        ok = ev.get("verify_status") == "ok" and ev.get("helpers") == 0
        out.append(f'<div class="banner {"ok" if ok else "fail"}"><small>Hidden check the agent '
                   f"never saw (harness re-verifies the final sources)</small>"
                   f"{esc(strip(_evaluation_line(ev)))}</div>")
    out.append("</div></header>")
    return "\n".join(out)


def render_step(step: Step, run: Run, total: int, strip: PathStripper) -> str:
    status = step.status
    scls = _STATUS_CLASS.get(status, "muted")
    tool = step.tool + (" (by harness)" if step.by == "harness" else "")
    # id is not "step-N" on purpose: the #step-N hash is handled by the script, and a matching
    # id would make the browser scroll the header out of view on load.
    parts = [f'<section class="step" id="s-{step.index + 1}">',
             f"<h2>{esc(tool)} {badge(status)}<span class=\"meta\">step {step.index + 1}/{total}"
             + (f" · iteration {step.iteration}" if step.iteration else "") + "</span></h2>"]

    if step.kind == "evaluation":
        ev = run.evaluation or {}
        ok = status == "ok"
        parts.append(f'<div class="hidden-check {"ok" if ok else ""}"><h3>Hidden check the agent '
                     "never saw</h3>"
                     f'<p class="summary {scls}">{esc(strip(step.summary))}</p>'
                     "<p>After the run the harness recompiled the final sources and ran verify and a "
                     "whole-tree helper scan. None of this was shown to the model.</p></div>")
        parts.append(f"<details open><summary>evaluation record</summary>"
                     f"<pre>{pretty({k: v for k, v in ev.items() if k != 'ts'}, strip)}</pre>"
                     "</details></section>")
        return "\n".join(parts)

    if step.kind == "tool":
        parts.append(f'<p class="summary {scls}">{esc(strip(step.summary))}</p>')

    parts.append(f"<h3>Model text · iteration {step.iteration}</h3>")
    texts = run.reasoning.get(step.iteration) or []
    if texts:
        parts.append(f'<div class="reason">{esc(strip(chr(10).join(texts)))}</div>')
    else:
        parts.append('<div class="reason none">No text this turn; the model went straight to '
                     "tool calls.</div>")
    for t in run.thinking.get(step.iteration) or []:
        parts.append(f"<details><summary>model thinking</summary><div class=\"reason\">"
                     f"{esc(strip(t))}</div></details>")

    if step.kind == "text":
        parts.append("</section>")
        return "\n".join(parts)

    inp = dict(step.input)
    if step.tool == "finish" and isinstance(inp.get("summary"), str):
        parts.append("<h3>What the agent claims</h3>")
        parts.append(f'<div class="reason claim">{esc(strip(inp.pop("summary")))}</div>')
    if step.tool == "write_file" and isinstance(inp.get("content"), str):
        content = inp.pop("content")
        inp["content"] = f"<{len(content.splitlines())} lines; diff below>"
    if inp:
        parts.append("<h3>Tool input</h3>")
        parts.append(f"<pre>{pretty(inp, strip)}</pre>")
    elif step.tool != "finish":
        parts.append("<h3>Tool input</h3><pre>{}</pre>")

    if step.diff is not None:
        added = sum(1 for d in step.diff if d.startswith("+") and not d.startswith("+++"))
        removed = sum(1 for d in step.diff if d.startswith("-") and not d.startswith("---"))
        parts.append(f"<h3>Diff vs {esc(step.diff_base)} · +{added} −{removed}</h3>")
        parts.append(render_diff([strip(d) for d in step.diff]) if step.diff
                     else '<div class="reason none">No change.</div>')
        parts.append(f"<details><summary>full file as written</summary>"
                     f"<pre>{esc(strip(step.input.get('content', '')))}</pre></details>")

    for note in step.notes:
        parts.append(f'<div class="note">{esc(strip(note))}</div>')
    if step.tool == "finish" and (run.stop or {}).get("baseline"):
        parts.append('<div class="limit">Baseline arm: the loop accepts finish as-is. Nothing in '
                     "this run executed the binary or compared its output with the reference.</div>")
    res = step.result or {}
    lims = res.get("limitations") or []
    if lims:
        parts.append("<h3>Limitations the tool reported</h3>")
        parts.extend(f'<div class="limit">{esc(strip(str(x)))}</div>' for x in lims)
    if step.result is not None:
        ev_text = pretty(res.get("evidence", {}), strip)
        is_open = " open" if len(ev_text) < 1600 and step.tool not in ("read_file",
                                                                      "write_file") else ""
        parts.append(f"<h3>Evidence</h3><details{is_open}><summary>evidence JSON "
                     f"({len(ev_text):,} chars)"
                     + (f" · source hash {esc(step.source_hash)}" if step.source_hash else "")
                     + f"</summary><pre>{ev_text}</pre></details>")
    parts.append("</section>")
    return "\n".join(parts)


def render_html(records: list[dict[str, Any]], title: str | None = None) -> str:
    start = next((r for r in records if r.get("type") == "start"), {})
    strip = PathStripper(start.get("run_dir"))
    run = build_run(records, strip)
    cfg = run.start.get("config") or {}
    title = title or f"Replay: {cfg.get('name', 'agent run')}"
    total = len(run.steps)
    nav = []
    for s in run.steps:
        tool = s.tool + ("*" if s.by == "harness" else "")
        it = f"{s.iteration}" if s.iteration else ""
        line = s.summary if s.kind != "text" else "model text, no tool call"
        nav.append(f'<li><a href="#step-{s.index + 1}" data-i="{s.index}">'
                   f'<span class="n">{esc(it)}</span><span class="t">{esc(tool)}</span>'
                   f'{badge(s.status)}<span class="s">{esc(strip(line))}</span></a></li>')
    body = [render_header(run, title, strip), "<main>",
            '<nav aria-label="steps"><ol>' + "".join(nav) + "</ol></nav>",
            '<div class="stage"><div class="controls"><button id="prev" type="button">◀ Prev'
            '</button><button id="next" type="button">Next ▶</button><span class="pos" id="pos">'
            '</span><span class="bar"><i id="bar"></i></span><span class="hint">← → keys · '
            "t: theme</span></div>"]
    if run.reference_build:
        rb = (run.reference_build.get("run") or {})
        body.append(f'<details class="nojs-only"><summary>reference build</summary><pre>'
                    f"{pretty(rb, strip)}</pre></details>")
    body.extend(render_step(s, run, total, strip) for s in run.steps)
    if not run.steps:
        body.append('<section class="step"><p>No steps in this trace.</p></section>')
    body.append("</div></main>")
    body.append(f"<footer>Generated by scripts/render_trace.py from trace.jsonl · "
                f"{len(records)} records · model output is shown verbatim (escaped) · "
                f"* = call made by the harness, not the model</footer>")
    return _page(title, "\n".join(body), JS)


# ----------------------------------------------------------------------------- site


def _run_facts(records: list[dict[str, Any]]) -> dict[str, Any]:
    start = next((r for r in records if r.get("type") == "start"), {})
    stop = next((r for r in records if r.get("type") == "stop"), {})
    ev = next((r for r in records if r.get("type") == "evaluation"), None)
    tools = [r.get("name") for r in records if r.get("type") == "tool_call" and r.get("by") != "harness"]
    finish = next((r for r in records if r.get("type") == "tool_call" and r.get("name") == "finish"), None)
    return {"task": (start.get("config") or {}).get("name", "?"),
            "arm": "baseline" if stop.get("baseline") else "toolkit",
            "stop": stop.get("stop_reason", "?"), "iterations": stop.get("iterations", "?"),
            "cost": stop.get("cost_usd"), "tool_calls": len(tools),
            "verify_calls": tools.count("verify"), "evaluation": ev,
            "claim": ((finish or {}).get("input") or {}).get("summary")}


def render_index(entries: list[dict[str, Any]], title: str, intro: str) -> str:
    cards = []
    for e in entries:
        f = e["facts"]
        ev = f["evaluation"]
        if ev is not None:
            ok = ev.get("verify_status") == "ok" and ev.get("helpers") == 0
            check = f'{badge("ok" if ok else "fail")} hidden check: {esc(_evaluation_line(ev))}'
        else:
            check = '<span class="badge muted">no hidden check recorded</span>'
        stop_cls = {"success": "ok", "model_finished": "error"}.get(f["stop"], "fail")
        cards.append(
            f'<li><a class="card-title" href="{esc(e["page"])}">{esc(e["title"])}</a>'
            f'<p class="goal">{esc(e.get("note", ""))}</p>'
            f'<p>{badge(stop_cls, f["stop"])} {esc(f["arm"])} · {esc(f["iterations"])} iterations · '
            f'{esc(f["tool_calls"])} tool calls · {esc(f["verify_calls"])} verify · cost '
            f"{esc(_money(f['cost']))}</p><p>{check}</p></li>")
    compare = ""
    pair = [e for e in entries if e.get("compare")]
    if len(pair) == 2:
        rows = [("Arm", lambda f: f["arm"]), ("Iterations", lambda f: f["iterations"]),
                ("Tool calls", lambda f: f["tool_calls"]), ("verify calls", lambda f: f["verify_calls"]),
                ("Loop stop", lambda f: f["stop"]), ("Cost", lambda f: _money(f["cost"])),
                ("Hidden check", lambda f: _evaluation_line(f["evaluation"])
                 if f["evaluation"] else "n/a")]
        head = "".join(f'<th><a href="{esc(e["page"])}">{esc(e["title"])}</a></th>' for e in pair)
        body = "".join(f"<tr><th>{esc(k)}</th>" + "".join(f"<td>{esc(fn(e['facts']))}</td>" for e in pair)
                       + "</tr>" for k, fn in rows)
        none = "(no finish call: the loop stopped on its own success check)"
        claims = "".join(f'<td><div class="reason claim">{esc(e["facts"]["claim"] or none)}'
                         "</div></td>" for e in pair)
        compare = (f'<h2>Side by side: same model, same task</h2><div class="tablewrap"><table>'
                   f"<tr><th></th>{head}</tr>{body}<tr><th>Agent's final claim</th>{claims}</tr>"
                   "</table></div>")
    css = ("ul.runs{list-style:none;padding:0;display:grid;gap:1rem;max-width:70rem}"
           "ul.runs li{background:var(--card);border:1px solid var(--line);border-radius:12px;"
           "padding:.8rem 1.1rem}.card-title{font-size:1.2rem;font-weight:700}"
           ".tablewrap{overflow-x:auto;max-width:100%}"
           "table{border-collapse:collapse;max-width:80rem;font-size:.9rem}"
           "td,th{border:1px solid var(--line);padding:.45rem .7rem;text-align:left;vertical-align:top}"
           "td{min-width:16rem}")
    body = (f"<header><h1>{esc(title)}</h1><p class=\"goal\">{esc(intro)}</p></header>"
            f'<div class="stage"><style>{css}</style>{compare}<h2>Runs</h2>'
            f'<ul class="runs">{"".join(cards)}</ul></div>'
            "<footer>Generated by scripts/render_trace.py --site. Each page is self-contained and "
            "works offline.</footer>")
    return _page(title, body)


def build_site(manifest_path: Path) -> list[Path]:
    """Manifest: {"title", "intro", "runs": [{"slug", "trace", "title", "note", "compare"?}]};
    trace paths are relative to the manifest. Writes <slug>.html and index.html next to it."""
    m = json.loads(manifest_path.read_text())
    out_dir = manifest_path.parent
    written: list[Path] = []
    entries = []
    for r in m["runs"]:
        records = load_trace(out_dir / r["trace"])
        page = out_dir / f"{r['slug']}.html"
        page.write_text(render_html(records, r.get("title")))
        written.append(page)
        entries.append({**r, "page": page.name, "facts": _run_facts(records)})
    index = out_dir / "index.html"
    index.write_text(render_index(entries, m.get("title", "Agent run replays"), m.get("intro", "")))
    written.append(index)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("trace", nargs="?", type=Path, help="trace.jsonl written by the agent loop")
    ap.add_argument("-o", "--output", type=Path, help="output HTML file")
    ap.add_argument("--title", help="page title (default: from the trace)")
    ap.add_argument("--site", type=Path, help="render every run in this manifest + index.html")
    a = ap.parse_args(argv)
    if a.site:
        for p in build_site(a.site):
            print(p)
        return 0
    if not a.trace or not a.output:
        ap.error("TRACE and -o are required (or use --site MANIFEST)")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(render_html(load_trace(a.trace), a.title))
    print(a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
