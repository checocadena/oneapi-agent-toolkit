#!/usr/bin/env bash
# Two-minute demo: the agent loop end to end, with no toolchain, no API key and no network.
#
#   1. Claude path (scripted model) and 2. Meta path (the real Meta client talking to a
#   scripted server), both with mock tool results, which every report marks MOCK.
#   3. The replay of a real run without the toolkit's checks (link only).
#
# Usage: scripts/demo.sh [runs_dir]   (default: a fresh temp dir)
set -euo pipefail
RUNS="${1:-$(mktemp -d)}"
OA="${OA:-oneapi-agent}"

summary() {  # print the outcome fields that matter from the loop's JSON
  python3 -c '
import json, sys
o = json.load(sys.stdin)
print("  stop_reason:", o["stop_reason"], "  iterations:", o.get("iterations"))
print("  report:     ", o["report"])'
}

for provider in anthropic meta; do
  echo "== $provider: oneapi-agent loop --dry-run --mock-tools --provider $provider"
  "$OA" loop --dry-run --mock-tools --provider "$provider" --runs-dir "$RUNS/$provider" | summary
  echo
done

echo "== What goes wrong without the checks (a real Claude run, step by step):"
echo "   https://checocadena.github.io/oneapi-agent-toolkit/docs/replay/baseline-softmax-fail.html"
echo
echo "Real tools need the oneAPI toolchain (README Quickstart); real models need an API key"
echo "(.env.example). Traces and reports from this demo: $RUNS"
