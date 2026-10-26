# Running the agent loop on Meta's models

The loop reaches models through a provider layer ([`src/oneapi_agent/models.py`](../src/oneapi_agent/models.py)).
`anthropic` (Claude) is the default. `meta` sends the same conversation to the Meta Model API's
OpenAI-compatible chat completions endpoint, translating the loop's tool calls both ways.

**Status (2026-10-26):** the Meta path is tested offline only, against a scripted server
that answers with real chat-completion JSON. **No request has reached Meta's API yet**, and
no Muse result exists. Base URL, auth, model ids, prices and request constraints below were
checked against Meta's docs at https://dev.meta.ai/docs on 2026-10-26.

## Try it now (no key, no network)

```bash
oneapi-agent loop --dry-run --mock-tools --provider meta
```

This runs the real Meta client (request building, tool schemas, tool-call and tool-result
translation, usage accounting) over a scripted transport, with mock tool results. The trace
in `runs/<ts>/trace.jsonl` shows `"provider": "meta"` and the model id.

## Real runs

```bash
export MODEL_API_KEY=...      # Meta's documented name; never commit it
oneapi-agent loop --provider meta --model muse-spark-1.3 --config examples/saxpy/task.json
```

| Setting | Default | Notes |
|---|---|---|
| `--provider` / `ONEAPI_AGENT_PROVIDER` / task `"provider"` | from the model name | `muse-*` selects meta, `claude-*` anthropic |
| `--model` / `ONEAPI_AGENT_MODEL` | `muse-spark-1.3` for meta | |
| `ONEAPI_AGENT_META_API_KEY`, `MODEL_API_KEY` or `META_API_KEY` | none | sent as `Authorization: Bearer`; never logged |
| `ONEAPI_AGENT_AUTH_FROM_NETWORK=1` | off | the key is added outside this process (e.g. a proxy that injects the header for `api.meta.ai`): no key needed, none sent. Not used for Claude |
| `ONEAPI_AGENT_META_BASE_URL` | `https://api.meta.ai/v1` | the client POSTs to `<base>/chat/completions` |
| `ONEAPI_AGENT_EFFORT` | unset (model decides) | sent as `reasoning_effort`: minimal, low, medium, high, xhigh, max (`max` on 1.3 and Wonder only) |
| prices | built in | Muse Spark 1.1–1.3 $1.25 in / $0.15 cached / $4.25 out per 1M tokens; Muse Wonder $4.00 / $0.20 / $20.00; override with `ONEAPI_AGENT_PRICE_IN` / `_OUT` / `_CACHED_IN` |

Requests use `max_completion_tokens` (it counts reasoning tokens), never set `tool_choice`
(Meta accepts only the default `auto`), and handle several tool calls per turn.

### Muse Spark Open (open weights) on Baseten

`--provider baseten` (or any model named `meta-models/...`) sends the same requests to
`https://inference.baseten.co/v1` with `BASETEN_API_KEY`; default model
`meta-models/muse-spark-open` (Meta's deploy page; the hackathon guide names
`meta-models/muse-spark-open-1026`, which is not on Meta's or Baseten's pages yet). Same
built-in price as Muse Spark.

Requests retry on HTTP 429 and 5xx (4 attempts), waiting for `Retry-After` when the server
sends it, else exponential back-off with jitter. Every run keeps the
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

1. ✅ Base URL, auth header, key name, model ids and prices checked in Meta's docs
   (2026-10-26).
2. `GET /v1/models` with the real key, to confirm the ids the account can use.
3. One smoke call with a tiny budget: `--max-iterations 2` on `examples/saxpy`.
4. Check that tool calling works (the trace shows `tool_call` events with parsed inputs).
5. Record the outcome, including any failure, in `PROGRESS.md` before running anything larger.
