"""Model providers for the agent loop.

The loop talks to every model through one interface, ``ModelClient.create``, with
Anthropic-shaped messages and content blocks (``text``, ``tool_use``, ``tool_result``).
A provider adapts that interface to a vendor API:

* ``anthropic``: Claude through the Anthropic Python SDK (``AnthropicModel``).
* ``meta``: Meta's Muse models through the Meta Model API's OpenAI-compatible chat
  completions endpoint (``OpenAICompatModel``). Requests go through a small transport
  function, so tests and ``--dry-run`` swap in a scripted transport and exercise the
  whole Meta path (request building, tool-call translation, usage) without a network.
* ``ScriptedModel``: the deterministic fake behind ``--dry-run`` for any provider.

Adding a provider means one class with a ``create`` method plus an entry in
``PROVIDERS``; nothing in the loop changes.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


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


def _usage(input_tokens: int = 0, output_tokens: int = 0, cache_write: int = 0,
           cache_read: int = 0) -> dict[str, int]:
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_write, "cache_read_input_tokens": cache_read}


# ----------------------------------------------------------------------------- providers


@dataclass(frozen=True)
class Provider:
    name: str
    key_env: tuple[str, ...]  # first one set wins
    base_url_env: str | None = None
    default_base_url: str | None = None
    model_prefixes: tuple[str, ...] = field(default=())  # for inferring the provider


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider("anthropic", ("ONEAPI_AGENT_API_KEY", "ANTHROPIC_API_KEY",
                                        "ANTHROPIC_AUTH_TOKEN"),
                          "ONEAPI_AGENT_BASE_URL", "https://api.anthropic.com", ("claude-",)),
    # Meta Model API (https://dev.meta.ai/docs/authentication, read 2026-10-26): Bearer
    # auth; the docs name the key MODEL_API_KEY.
    "meta": Provider("meta", ("ONEAPI_AGENT_META_API_KEY", "MODEL_API_KEY", "META_API_KEY"),
                     "ONEAPI_AGENT_META_BASE_URL", "https://api.meta.ai/v1", ("muse-",)),
    # Muse Spark Open (open weights) on Baseten, per dev.meta.ai/docs/muse-spark-open/deploy.
    "baseten": Provider("baseten", ("ONEAPI_AGENT_BASETEN_API_KEY", "BASETEN_API_KEY"),
                        "ONEAPI_AGENT_BASETEN_BASE_URL", "https://inference.baseten.co/v1",
                        ("meta-models/",)),
}
PROVIDER_ENV = "ONEAPI_AGENT_PROVIDER"
# Set to 1 when the key is added to requests outside this process (e.g. a proxy that injects
# the Authorization header for the provider's host): no key is needed or sent.
NETWORK_AUTH_ENV = "ONEAPI_AGENT_AUTH_FROM_NETWORK"


def resolve_provider(model: str, explicit: str | None = None,
                     environ: Mapping[str, str] | None = None) -> str:
    """Pick the provider: an explicit choice, then ``ONEAPI_AGENT_PROVIDER``, then the model
    name's prefix (``claude-`` -> anthropic, ``muse-`` -> meta), else anthropic."""
    env = os.environ if environ is None else environ
    choice = explicit or env.get(PROVIDER_ENV)
    if choice:
        if choice not in PROVIDERS:
            raise ValueError(f"unknown provider {choice!r}; choose one of {sorted(PROVIDERS)}")
        return choice
    for p in PROVIDERS.values():
        if model.startswith(p.model_prefixes):
            return p.name
    return "anthropic"


def provider_api_key(provider: str, environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    return next((env[k] for k in PROVIDERS[provider].key_env if env.get(k)), None)


def auth_from_network(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(NETWORK_AUTH_ENV) == "1"


# ----------------------------------------------------------------------------- anthropic


DEFAULT_API_BASE_URL = "https://api.anthropic.com"


def make_anthropic_client(anthropic_module: Any) -> Any:
    """Build the SDK client for the loop.

    ``ONEAPI_AGENT_API_KEY`` takes precedence over the SDK's own credential lookup. Hosts
    that reserve ``ANTHROPIC_API_KEY`` for their own auth (Claude Code sessions blank it)
    can use this name instead. With it, the base URL is pinned to ``ONEAPI_AGENT_BASE_URL``
    (default: the public API) so an inherited ``ANTHROPIC_BASE_URL`` meant for the host
    never receives the loop's key.
    """
    key = os.environ.get("ONEAPI_AGENT_API_KEY")
    if key:
        base = os.environ.get("ONEAPI_AGENT_BASE_URL", DEFAULT_API_BASE_URL)
        return anthropic_module.Anthropic(api_key=key, base_url=base)
    return anthropic_module.Anthropic()  # ANTHROPIC_API_KEY or other SDK credential sources


class AnthropicModel:
    """Claude via the Anthropic Python SDK (manual tool-use loop)."""

    def __init__(self, model: str, max_tokens: int = 16000) -> None:
        import anthropic  # optional dependency: pip install .[agent]

        self.client = make_anthropic_client(anthropic)
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
        usage = _usage(u.input_tokens or 0, u.output_tokens or 0,
                       getattr(u, "cache_creation_input_tokens", 0) or 0,
                       getattr(u, "cache_read_input_tokens", 0) or 0)
        content = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
        return ModelResponse(content, resp.stop_reason or "", usage, resp.model)


# ----------------------------------------------------------------------------- openai-compatible

# (url, headers, JSON body) -> parsed JSON response
Transport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    """Seconds to wait: the server's ``Retry-After`` if given, else exponential with jitter."""
    if retry_after:
        try:
            return min(float(retry_after), 120.0)
        except ValueError:
            pass
    return 2 ** attempt + random.uniform(0, 1)


def http_transport(timeout_s: float = 300.0, attempts: int = 4) -> Transport:
    """POST JSON with retries on rate limits and server errors. Never logs headers (key)."""

    def post(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode()
        for attempt in range(attempts):
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={**headers, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in RETRY_STATUS and attempt + 1 < attempts:
                    time.sleep(_backoff(attempt, e.headers.get("Retry-After") if e.headers
                                        else None))
                    continue
                raise RuntimeError(f"{url}: HTTP {e.code}: {detail}") from None
            except urllib.error.URLError as e:
                if attempt + 1 < attempts:
                    time.sleep(_backoff(attempt))
                    continue
                raise RuntimeError(f"{url}: {e.reason}") from None
        raise AssertionError("unreachable")

    return post


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t["name"],
                                              "description": t.get("description", ""),
                                              "parameters": t["input_schema"]}}
            for t in tools]


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic-shaped history -> OpenAI chat messages.

    Assistant ``tool_use`` blocks become ``tool_calls``; user ``tool_result`` blocks become
    ``role: tool`` messages (in order, before any text in the same user turn).
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"],
                                   "arguments": json.dumps(b.get("input") or {})}}
                     for b in content if b.get("type") == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        texts = []
        for b in content:
            if b.get("type") == "tool_result":
                c = b.get("content", "")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": c})
            elif b.get("type") == "text":
                texts.append(b["text"])
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


FINISH_TO_STOP = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens",
                  "content_filter": "refusal"}


def from_openai_response(resp: dict[str, Any]) -> ModelResponse:
    """OpenAI chat completion -> Anthropic-shaped ``ModelResponse``."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            # Keep the turn: the loop's tools reject the bad input with evidence.
            args = {"_unparsed_arguments": raw}
        content.append({"type": "tool_use", "id": call.get("id") or f"call_{len(content)}",
                        "name": fn.get("name", ""), "input": args if isinstance(args, dict)
                        else {"_arguments": args}})
    finish = choice.get("finish_reason") or ""
    stop = FINISH_TO_STOP.get(finish, finish)
    if any(b["type"] == "tool_use" for b in content):
        stop = "tool_use"  # some servers report "stop" alongside tool calls
    u = resp.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    usage = _usage(max((u.get("prompt_tokens") or 0) - cached, 0),
                   u.get("completion_tokens") or 0, 0, cached)
    return ModelResponse(content, stop, usage, resp.get("model", ""))


class OpenAICompatModel:
    """Any OpenAI-compatible chat completions API with tool calling (Meta, Baseten).

    Follows the Meta Model API's documented constraints: ``max_completion_tokens`` (it counts
    reasoning tokens), no ``tool_choice`` other than the default ``auto``, and an optional
    ``reasoning_effort`` from ``ONEAPI_AGENT_EFFORT`` (minimal|low|medium|high|xhigh|max).
    """

    def __init__(self, model: str, *, provider: str = "meta", api_key: str | None = None,
                 base_url: str | None = None, transport: Transport | None = None,
                 max_tokens: int = 16000) -> None:
        p = PROVIDERS[provider]
        self.name = model
        self.provider = provider
        self.api_key = api_key if api_key is not None else provider_api_key(provider)
        self.effort = os.environ.get("ONEAPI_AGENT_EFFORT")
        self.base_url = (base_url or (os.environ.get(p.base_url_env) if p.base_url_env else None)
                         or p.default_base_url or "").rstrip("/")
        self.transport = transport or http_transport()
        self.max_tokens = max_tokens
        self.is_mock = False  # True when the transport is scripted (costs nothing)

    def create(self, *, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]]) -> ModelResponse:
        body: dict[str, Any] = {"model": self.name, "max_completion_tokens": self.max_tokens,
                                "messages": to_openai_messages(system, messages)}
        if self.effort:
            body["reasoning_effort"] = self.effort
        if tools:
            body["tools"] = to_openai_tools(tools)
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = from_openai_response(self.transport(f"{self.base_url}/chat/completions",
                                                   headers, body))
        resp.model = resp.model or self.name
        return resp


# ----------------------------------------------------------------------------- fakes


class ScriptedModel:
    """Deterministic fake model for --dry-run and tests.

    ``script`` is a list of turns; each turn is a list of ``(tool_name, input)`` pairs,
    or a string (a text-only turn). After the script ends, it replies with text only.
    """

    is_mock = True  # costs nothing; the loop does not price it

    def __init__(self, script: list[list[tuple[str, dict[str, Any]]] | str],
                 tokens_per_turn: tuple[int, int] = (1500, 200)) -> None:
        self.script = list(script)
        self.turn = 0
        self.name = "scripted-dry-run"
        self.tokens_per_turn = tokens_per_turn

    def create(self, *, system: str, messages: list[dict[str, Any]],
               tools: list[dict[str, Any]]) -> ModelResponse:
        usage = _usage(*self.tokens_per_turn)
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


class ScriptedTransport:
    """Fake OpenAI-compatible server that plays a dry-run script.

    It answers with real chat-completion JSON, so ``OpenAICompatModel`` runs its full
    request/response translation offline. Every request is kept in ``requests`` for tests.
    """

    def __init__(self, script: list[list[tuple[str, dict[str, Any]]] | str],
                 tokens_per_turn: tuple[int, int] = (1500, 200)) -> None:
        self.script = list(script)
        self.tokens_per_turn = tokens_per_turn
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((url, body))
        turn = len(self.requests)
        step = self.script[turn - 1] if turn <= len(self.script) else "(script exhausted)"
        if isinstance(step, str):
            msg: dict[str, Any] = {"role": "assistant", "content": step}
            finish = "stop"
        else:
            msg = {"role": "assistant", "content": f"[scripted turn {turn}]",
                   "tool_calls": [{"id": f"call_dry_{turn:03d}_{i}", "type": "function",
                                   "function": {"name": n, "arguments": json.dumps(a)}}
                                  for i, (n, a) in enumerate(step)]}
            finish = "tool_calls"
        pin, pout = self.tokens_per_turn
        return {"id": f"chatcmpl-dry-{turn}", "object": "chat.completion",
                "model": body.get("model", ""),
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": pin, "completion_tokens": pout,
                          "total_tokens": pin + pout}}


def make_model(provider: str, model: str, *, dry_run_script: list[Any] | None = None
               ) -> ModelClient:
    """The client for ``provider``. With ``dry_run_script`` nothing leaves the machine:
    anthropic gets ``ScriptedModel``; OpenAI-compatible providers (meta, baseten) get their
    real client over ``ScriptedTransport``."""
    if dry_run_script is not None:
        if provider != "anthropic":
            # The name says "scripted" so no report or trace can pass for a real Muse run.
            m = OpenAICompatModel(f"scripted-dry-run ({provider} client, {model})",
                                  provider=provider, api_key="",
                                  transport=ScriptedTransport(dry_run_script))
            m.is_mock = True
            return m
        return ScriptedModel(dry_run_script)
    if provider == "anthropic":
        return AnthropicModel(model)
    return OpenAICompatModel(model, provider=provider)
