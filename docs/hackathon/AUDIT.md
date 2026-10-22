# Repository audit (2026-10-22, before the hackathon)

Audited at `main` = `fc0d5fe` (v0.3.0), then extended on branch `meta-hackathon-prep`.

## What exists and works

| Area | State | Evidence |
|---|---|---|
| Five evidence-returning tools (`migrate`, `compile`, `scan_helpers`, `verify`, `benchmark`) | working | `tests/test_tools.py`; CI `toolchain` job runs them with the real icpx + SYCLomatic |
| MCP server + `oneapi-agent setup` for Claude Code, Codex, Cursor, Gemini CLI | working | `docs/agent-registration.md` |
| Standalone agent loop (success = verify ok AND zero helpers on the same source) | working | `tests/test_loop.py`; real runs in `docs/real-runs/` |
| Hidden-grader comparison (toolkit / no checks / shell only) | done with Claude | 40/40 vs 23/40 vs 14/40 (`docs/ablation/round2/`); 30/30 vs 18/30 vs 17/30 (`docs/agent-ports/`) |
| GPU verification: `gpu-test`, `bundle`, `remote-test` (4 clouds, machine always deleted) | NVIDIA and AMD verified | `docs/gpu-testing.md`; 10/10 agent ports pass on NVIDIA and an MI350X |
| Replay viewer for traces | working | `docs/replay/` (GitHub Pages) |
| CI | green on `main` | `test` (3.11, 3.13: ruff, mypy, pytest) and `toolchain` (real toolchain + claims gate) |
| Tests | 199 pass, 4 skip without the toolchain | `pytest -q` |

## What was missing for the hackathon (fixed on this branch)

| Gap | Fix |
|---|---|
| The loop could only call Claude (`AnthropicModel` hard-wired in `cli_entry`) | Provider layer in `src/oneapi_agent/models.py`: `anthropic` and `meta` (OpenAI-compatible), chosen by `--provider`, task file, env or model-name prefix |
| No way to exercise a non-Claude path without a key | `--dry-run --provider meta` runs the real Meta client against a scripted server (`ScriptedTransport`); 10 new tests |
| No `.env.example` | Added; lists every setting the toolkit reads |
| No one-command demo | `scripts/demo.sh` (offline, both providers, results marked MOCK) |
| README status row said "174 passed" (stale); roadmap said the loop was Claude-only | Updated |
| Third-party components and licenses not listed in one place | Table added to the README |
| An empty `ONEAPI_AGENT_MODEL` (e.g. from a sourced `.env`) became an empty model id | Falls back to the default |

## Still missing or unfinished (not fixed here)

- **No Meta result of any kind.** The Meta client has never reached the real API; base URL,
  key name and model ids need the day-one check (`docs/meta-model-api.md`).
- **No built-in price for Muse.** Deliberate: prices come from `.env` until confirmed on
  Meta's pricing page; the loop refuses to run an unpriced model (D11).
- **Multi-file CUDA projects are untested.** Every measured port is a single-file program;
  `migrate` on a project (`c2s -p` with `compile_commands.json`) is on the roadmap. This is the
  main gap for porting a real repository such as llm.c.
- **Intel GPUs untested** (the Level Zero path exists).
- **Benchmarks include JIT and runtime start-up**; AOT builds are on the roadmap.
- **Not on PyPI** (`publish.yml` waits for a PyPI trusted publisher).
- **PR #17** (links to the two intel/llvm issues) is open, not merged.
- Outside code contributions wait for a CLA (dual licensing).
