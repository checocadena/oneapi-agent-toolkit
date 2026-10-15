"""``oneapi-agent ci setup``: a GitHub Actions workflow that runs ``remote-test`` on labelled PRs.

Writes ``.github/workflows/gpu-test.yml`` and then offers to copy the provider keys found on
this machine into GitHub Actions secrets with ``gh secret set NAME``. The key goes to ``gh``
on stdin only: never on a command line, in output or in the result.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .keys import PROVIDER_KEY_ENV, PROVIDERS, find_key
from .result import ToolResult

DEFAULT_INSTALL = "oneapi-agent-toolkit @ git+https://github.com/checocadena/oneapi-agent-toolkit@main"
WORKFLOW_PATH = ".github/workflows/gpu-test.yml"

# ${{ }} expressions are GitHub's; {placeholders} are filled by ``render_workflow``. Inputs and
# event fields reach the shell only through env vars, never pasted into the script.
TEMPLATE = """\
# Written by `oneapi-agent ci setup`. Rents a GPU, runs the SYCL port's gpu-test there,
# deletes the machine, and uploads the results.
# Runs on pull requests labelled `gpu-test` (from this repository only, never from forks)
# and on manual dispatch. Spending is capped by --max-usd and --max-minutes.
name: gpu-test
on:
  pull_request:
    types: [labeled, synchronize, reopened]
  workflow_dispatch:
    inputs:
      task:
        description: task.json to test
        default: "{task}"
      provider:
        description: runpod, vast, vultr or digitalocean
        default: "{provider}"
      vendor:
        description: nvidia or amd
        default: "{vendor}"
permissions:
  contents: read
concurrency:
  group: gpu-test-${{{{ github.event.pull_request.number || github.ref }}}}
  cancel-in-progress: false  # a cancelled run could not delete its machine
jobs:
  gpu-test:
    # Fork PRs get no secrets anyway; this guard makes sure they never even start.
    if: >-
      github.event_name == 'workflow_dispatch' ||
      (github.event.pull_request.head.repo.full_name == github.repository &&
       contains(github.event.pull_request.labels.*.name, 'gpu-test') &&
       (github.event.action != 'labeled' || github.event.label.name == 'gpu-test'))
    runs-on: ubuntu-latest
    timeout-minutes: {timeout}
    env:
      TASK: ${{{{ inputs.task || '{task}' }}}}
      PROVIDER: ${{{{ inputs.provider || '{provider}' }}}}
      VENDOR: ${{{{ inputs.vendor || '{vendor}' }}}}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "{install}"
      - name: remote-test
        env:  # only this step sees the provider keys
{secret_env}
        run: >-
          oneapi-agent remote-test "$TASK" --provider "$PROVIDER" --vendor "$VENDOR"
          --max-usd {max_usd} --max-minutes {max_minutes} --into gpu-run
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: gpu-test-results
          path: gpu-run/
          if-no-files-found: warn
"""


def render_workflow(task: str, provider: str = "runpod", vendor: str = "nvidia",
                    max_usd: float = 2.0, max_minutes: float = 45,
                    install: str = DEFAULT_INSTALL) -> str:
    for name, value in (("task", task), ("install", install)):
        if any(c in value for c in "\"'\n$`\\"):
            raise ValueError(f"{name} must not contain quotes, $, backticks or newlines")
    secret_env = "\n".join(f"          {env}: ${{{{ secrets.{env} }}}}"
                           for env in PROVIDER_KEY_ENV.values())
    return TEMPLATE.format(task=task, provider=provider, vendor=vendor, max_usd=max_usd,
                           max_minutes=max_minutes, timeout=int(max_minutes) + 20,
                           secret_env=secret_env, install=install)


def _ask(question: str) -> bool:
    """y/N on the terminal; prompts go to stderr so stdout stays the JSON result."""
    if not sys.stdin.isatty():
        return False
    print(question + " [y/N] ", end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().lower() in ("y", "yes")


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def _store_secrets(repo: Path, yes: bool, ask: Callable[[str], bool],
                   run: Runner) -> dict[str, Any]:
    found = {p: f for p in PROVIDERS if (f := find_key(p))}
    if not found:
        return {"skipped": "no provider key found on this machine"}
    if not shutil.which("gh"):
        return {"skipped": "gh (GitHub CLI) is not installed"}
    auth = run(["gh", "auth", "status"], cwd=repo, capture_output=True, text=True)
    if auth.returncode != 0:
        return {"skipped": "gh is not logged in (run `gh auth login`)"}
    out: dict[str, Any] = {}
    for provider, (key, source) in found.items():
        name = PROVIDER_KEY_ENV[provider]
        if not yes and not ask(f"Store the {provider} key (from {source}) as GitHub secret {name}?"):
            out[name] = "skipped"
            continue
        r = run(["gh", "secret", "set", name], cwd=repo, input=key, capture_output=True, text=True)
        if r.returncode == 0:
            out[name] = f"set (from {source})"
        else:  # gh's message names the repo and the error; scrub the key in case it echoes it
            out[name] = "failed: " + (r.stderr or r.stdout or "").strip().replace(key, "***")[:300]
    return out


def ci_setup(repo: str = ".", task: str = "task.json", provider: str = "runpod",
             vendor: str = "nvidia", max_usd: float = 2.0, max_minutes: float = 45,
             install: str = DEFAULT_INSTALL, force: bool = False, secrets: bool = True,
             yes: bool = False, ask: Callable[[str], bool] = _ask,
             run: Runner | None = None) -> ToolResult:
    root = Path(repo)
    run = run or subprocess.run
    if provider not in PROVIDERS:
        return ToolResult("ci_setup", "error", f"unknown provider {provider!r}; choose from {PROVIDERS}")
    try:
        text = render_workflow(task, provider, vendor, max_usd, max_minutes, install)
    except ValueError as e:
        return ToolResult("ci_setup", "error", str(e))
    path = root / WORKFLOW_PATH
    if path.exists() and not force:
        return ToolResult("ci_setup", "error", f"{path} exists; pass --force to overwrite",
                          {"workflow": str(path)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    ev: dict[str, Any] = {"workflow": str(path), "task": task, "provider": provider,
                          "vendor": vendor, "max_usd": max_usd, "max_minutes": max_minutes,
                          "install": install}
    ev["secrets"] = _store_secrets(root, yes, ask, run) if secrets else {"skipped": "--no-secrets"}
    lim = ["Add the `gpu-test` label to a pull request (or run the workflow by hand) to start "
           "a run; each run rents a GPU and costs money, up to --max-usd.",
           "Pull requests from forks never run it.",
           f"The workflow installs `{install}`; pin a commit or tag for reproducible runs."]
    missing = [PROVIDER_KEY_ENV[provider]]
    if not str(ev["secrets"].get(missing[0], "")).startswith("set"):
        lim.append(f"The default provider needs the {missing[0]} repository secret: "
                   f"`gh secret set {missing[0]}` (it reads the value from stdin).")
    return ToolResult("ci_setup", "ok", f"wrote {path}", ev, lim)
