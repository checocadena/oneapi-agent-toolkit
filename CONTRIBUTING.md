# Contributing

Thanks for your interest. Issues, bug reports, results from your own runs, and ideas are
welcome right now: open an issue.

## Code contributions: not accepted yet

oneapi-agent-toolkit is released under the [AGPL-3.0](LICENSE) and is also offered under
commercial licenses (see [License](README.md#license)). Offering both is only possible
while the author holds the rights to all of the code. Until a contributor license
agreement (CLA) is in place, pull requests with code from outside contributors cannot be
merged. If you have a fix, please describe it in an issue instead.

When a CLA is adopted, this file will link to it and explain how to sign it.

## Ground rules for anything that lands here

- **Evidence, not claims.** A result goes in the docs only with the committed output
  behind it (traces, tool results, final sources). See [`AGENTS.md`](AGENTS.md).
- **Run the checks** before proposing a change: `pytest -q`, `ruff check .`, `mypy src`.
- **Don't edit past results.** Fix the code and rerun; record what changed in
  [`DECISIONS.md`](DECISIONS.md) and [`PROGRESS.md`](PROGRESS.md).
