# Running the agent loop on Meta's models

The loop reaches models through a provider layer ([`src/oneapi_agent/models.py`](../src/oneapi_agent/models.py)).
`anthropic` (Claude) is the default. `meta` sends the same conversation to the Meta Model API's
OpenAI-compatible chat completions endpoint, translating the loop's tool calls both ways.

**Status (2026-10-22):** the Meta path is tested offline only, against a scripted server
that answers with real chat-completion JSON. **No request has reached Meta's API yet**, and
no Muse result exists. The base URL, key name and model ids below come from Meta's developer
docs as read before API access opened; the checklist at the end confirms them.

## Try it now (no key, no network)

```bash
oneapi-agent loop --dry-run --mock-tools --provider meta
```

This runs the real Meta client (request building, tool schemas, tool-call and tool-result
translation, usage accounting) over a scripted transport, with mock tool results. The trace
in `runs/<ts>/trace.jsonl` shows `"provider": "meta"` and the model id.

## Real runs

```bash
export ONEAPI_AGENT_META_API_KEY=...        # or META_API_KEY; never commit it
export ONEAPI_AGENT_PRICE_IN=... ONEAPI_AGENT_PRICE_OUT=...   # USD per 1M tokens, from Meta's pricing page
oneapi-agent loop --provider meta --model muse-spark-1.3 --config examples/saxpy/task.json
```

| Setting | Default | Notes |
|---|---|---|
| `--provider` / `ONEAPI_AGENT_PROVIDER` / task `"provider"` | from the model name | `muse-*` selects meta, `claude-*` anthropic |
| `--model` / `ONEAPI_AGENT_MODEL` | `muse-spark-1.3` for meta | |
| `ONEAPI_AGENT_META_API_KEY` or `META_API_KEY` | none | sent as `Authorization: Bearer`; never logged |
| `ONEAPI_AGENT_META_BASE_URL` | `https://api.meta.ai/v1` | the client POSTs to `<base>/chat/completions` |
| `ONEAPI_AGENT_PRICE_IN` / `_OUT` | none | required: Muse has no built-in price, and the loop refuses to run a model it cannot cost-cap ([D11](../DECISIONS.md)) |

Requests retry on HTTP 429 and 5xx (4 attempts, exponential back-off). Every run keeps the
per-run caps: `max_cost_usd` (default $5), `max_total_tokens` and `max_iterations`.

## How the translation works

| Loop (Anthropic-shaped) | Meta (OpenAI-compatible) |
|---|---|
| `system` string | first message, `role: system` |
| tool `{name, description, input_schema}` | `{type: function, function: {name, description, parameters}}` |
| assistant `tool_use` blocks | assistant `tool_calls` (arguments as a JSON string) |
| user `tool_result` blocks | one `role: tool` message each, in order |
| `stop_reason` | `finish_reason`: `tool_calls` → `tool_use`, `stop` → `end_turn`, `length` → `max_tokens`, `content_filter` → `refusal` |
| usage | `prompt_tokens` (minus cached) → input, `cached_tokens` → cache read, `completion_tokens` → output |

Tool-call arguments that are not valid JSON are passed to the tool as
`{"_unparsed_arguments": ...}`, so the tool rejects them with evidence and the model can
retry, instead of the run crashing.

## Day-one checklist (when API access opens)

1. Confirm the base URL, the auth header and the key's environment name in Meta's docs. If
   they differ, change `PROVIDERS["meta"]` in `models.py` (one line each).
2. List the available model ids and pick the default (`DEFAULT_META_MODEL` in `loop.py`).
3. Read the pricing page; put the prices in `.env` (`ONEAPI_AGENT_PRICE_IN` / `_OUT`).
4. One smoke call with a tiny budget: `--max-iterations 2` on `examples/saxpy`.
5. Check that tool calling works (the trace shows `tool_call` events with parsed inputs).
6. Record the outcome, including any failure, in `PROGRESS.md` before running anything larger.
