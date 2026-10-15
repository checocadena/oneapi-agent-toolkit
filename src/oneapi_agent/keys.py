"""GPU provider API keys: the OS password store (``login``/``logout``) and ``providers``.

A key is looked up in this order (``find_key``): an explicit argument, the provider's
environment variable(s), then the OS password store through the optional ``keyring`` package
(macOS Keychain, Windows Credential Manager, Linux Secret Service), under the service name
``oneapi-agent`` with the provider name as the user name.

A key is only ever sent to its own provider's API. It is never written to disk in plain text
by this module, never logged or printed, and never part of a ToolResult.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .result import Status, ToolResult

SERVICE = "oneapi-agent"
PROVIDERS = ("runpod", "vast", "vultr", "digitalocean")
PROVIDER_KEY_ENV = {"runpod": "RUNPOD_API_KEY", "vast": "VAST_API_KEY", "vultr": "VULTR_API_KEY",
                    "digitalocean": "OCEAN_API_KEY"}
# Also accepted: the name doctl uses.
PROVIDER_KEY_ALIASES = {"digitalocean": ("DIGITALOCEAN_TOKEN",)}
# Where each provider's API key is created.
CONSOLE_URL = {"runpod": "https://console.runpod.io/user/credentials",
               "vast": "https://cloud.vast.ai/manage-keys/",
               "vultr": "https://my.vultr.com/settings/#settingsapi",
               "digitalocean": "https://cloud.digitalocean.com/account/api/tokens"}
AGENT_NOTE = ("Ask the user to create the key and run the login command in their own terminal. "
              "Do not ask for the key, and do not handle, store or pass it: no tool accepts one.")
INSTALL_HINT = "pip install 'oneapi-agent-toolkit[keyring]'"
USER_AGENT = "oneapi-agent-toolkit (+https://github.com/checocadena/oneapi-agent-toolkit)"


def _keyring() -> Any:
    """The ``keyring`` module, or None when it is not installed."""
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def stored_key(provider: str) -> str | None:
    """The key saved by ``login``, or None (no keyring package, no backend, nothing saved)."""
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(SERVICE, provider) or None
    except Exception:  # no usable backend (headless Linux, locked store): fall through
        return None


def find_key(provider: str, api_key: str | None = None) -> tuple[str, str] | None:
    """``(key, source)`` for a provider, or None. ``source`` names where the key came from
    (``argument``, the environment variable, or ``keyring``), never the key."""
    if api_key:
        return api_key, "argument"
    for env in (PROVIDER_KEY_ENV.get(provider, ""), *PROVIDER_KEY_ALIASES.get(provider, ())):
        if env and os.environ.get(env):
            return os.environ[env], env
    key = stored_key(provider)
    return (key, "keyring") if key else None


def missing_key(provider: str) -> dict[str, Any]:
    """What a user (or an agent relaying to one) needs to bring a key; contains no key."""
    envs = [PROVIDER_KEY_ENV[provider], *PROVIDER_KEY_ALIASES.get(provider, ())]
    return {"provider": provider, "console_url": CONSOLE_URL[provider],
            "login_command": f"oneapi-agent login {provider}", "env": envs,
            "agent_note": AGENT_NOTE}


def missing_key_result(tool: str, provider: str) -> ToolResult:
    """The ``toolchain_missing`` result for a provider with no key anywhere."""
    env = PROVIDER_KEY_ENV[provider]
    return ToolResult(tool, "toolchain_missing",
                      f"no {provider} API key: {env} is not set and none is saved",
                      missing_key(provider),
                      [f"Create an API key at {CONSOLE_URL[provider]}, then run "
                       f"`oneapi-agent login {provider}` in a terminal (or export {env}).",
                       AGENT_NOTE])


def _no_keyring(tool: str) -> ToolResult:
    return ToolResult(tool, "toolchain_missing", "the keyring package is not installed",
                      {"install": INSTALL_HINT},
                      [f"Install it with: {INSTALL_HINT}",
                       "Or keep using the environment variables: "
                       + ", ".join(PROVIDER_KEY_ENV.values()) + " (or DIGITALOCEAN_TOKEN)."])


def login(provider: str, key: str) -> ToolResult:
    """Save ``key`` in the OS password store for ``provider``."""
    if provider not in PROVIDERS:
        return ToolResult("login", "error", f"unknown provider {provider!r}; choose from {PROVIDERS}")
    kr = _keyring()
    if kr is None:
        return _no_keyring("login")
    key = key.strip()
    if not key:
        return ToolResult("login", "error", "no key given; nothing saved")
    try:
        kr.set_password(SERVICE, provider, key)
    except Exception as e:
        return ToolResult("login", "error", f"the password store refused the key ({type(e).__name__})",
                          {"provider": provider})
    lim = []
    env = PROVIDER_KEY_ENV[provider]
    if os.environ.get(env):
        lim.append(f"{env} is set in this environment and takes precedence over the saved key.")
    return ToolResult("login", "ok", f"{provider} key saved in the OS password store",
                      {"provider": provider, "service": SERVICE},
                      lim + ["Not checked against the provider; run `oneapi-agent providers`."])


def logout(provider: str) -> ToolResult:
    """Remove the saved key for ``provider`` from the OS password store."""
    if provider not in PROVIDERS:
        return ToolResult("logout", "error", f"unknown provider {provider!r}; choose from {PROVIDERS}")
    kr = _keyring()
    if kr is None:
        return _no_keyring("logout")
    try:
        if not kr.get_password(SERVICE, provider):
            return ToolResult("logout", "ok", f"no saved {provider} key", {"provider": provider})
        kr.delete_password(SERVICE, provider)
    except Exception as e:
        return ToolResult("logout", "error", f"the password store failed ({type(e).__name__})",
                          {"provider": provider})
    return ToolResult("logout", "ok", f"{provider} key removed from the OS password store",
                      {"provider": provider, "service": SERVICE})


# ------------------------------------------------------------------- providers

Request = Callable[[str, str, str, Any], Any]  # (method, url, key, body) -> parsed JSON


def _http(method: str, url: str, key: str, body: Any = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Accept": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(r, timeout=30) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def _num(v: Any) -> float | None:
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def account_facts(provider: str, key: str, req: Request = _http) -> dict[str, Any]:
    """One read-only API call that proves the key works, and the account facts it returns."""
    if provider == "runpod":
        # REST v2 has no account endpoint; the GraphQL ``myself`` query only reads.
        data = req("POST", "https://api.runpod.io/graphql", key,
                   {"query": "query { myself { clientBalance currentSpendPerHr spendLimit } }"})
        if data.get("errors"):
            raise PermissionError("Runpod rejected the key")
        me = (data.get("data") or {}).get("myself") or {}
        return {"balance_usd": _num(me.get("clientBalance")),
                "spend_per_hour_usd": _num(me.get("currentSpendPerHr")),
                "spend_limit_usd": _num(me.get("spendLimit"))}
    if provider == "vast":
        me = req("GET", "https://console.vast.ai/api/v0/users/current/", key, None) or {}
        return {"credit_usd": _num(me.get("credit")), "balance_usd": _num(me.get("balance"))}
    if provider == "vultr":
        acc = (req("GET", "https://api.vultr.com/v2/account", key, None) or {}).get("account", {})
        # Vultr reports prepaid credit as a negative balance.
        return {"balance_usd": _num(acc.get("balance")),
                "pending_charges_usd": _num(acc.get("pending_charges"))}
    if provider == "digitalocean":
        acc = (req("GET", "https://api.digitalocean.com/v2/account", key, None) or {}).get("account", {})
        facts = {"status": acc.get("status"), "droplet_limit": acc.get("droplet_limit")}
        facts.update({k: v for k, v in acc.items() if "gpu" in k.lower()})
        return facts
    raise ValueError(f"unknown provider {provider!r}")


def providers(req: Request = _http) -> ToolResult:
    """For each provider: is a key found, where from, and does the provider accept it."""
    rows: dict[str, dict[str, Any]] = {}
    for name in PROVIDERS:
        found = find_key(name)
        if found is None:
            rows[name] = {"key": "missing", **{k: v for k, v in missing_key(name).items()
                                               if k not in ("provider", "agent_note")}}
            continue
        key, source = found
        row: dict[str, Any] = {"key": "found", "source": source}
        try:
            row["account"] = {k: v for k, v in account_facts(name, key, req).items() if v is not None}
            row["valid"] = True
        except urllib.error.HTTPError as e:
            row["valid"] = None if e.code >= 500 or e.code == 429 else False
            row["error"] = f"HTTP {e.code}"
        except PermissionError as e:
            row["valid"], row["error"] = False, str(e)
        except (urllib.error.URLError, OSError, ValueError) as e:
            row["valid"], row["error"] = None, f"could not check ({type(e).__name__})"
        rows[name] = row
    have = [n for n, r in rows.items() if r["key"] == "found"]
    bad = [n for n in have if rows[n].get("valid") is False]
    unchecked = [n for n in have if rows[n].get("valid") is None]

    def part(n: str) -> str:
        r = rows[n]
        if r["key"] == "missing":
            return f"{n}: no key"
        state = {True: "valid", False: "INVALID", None: "unchecked"}[r["valid"]]
        return f"{n}: {state} ({r['source']})"

    summary = "; ".join(part(n) for n in PROVIDERS)
    status: Status
    if not have:
        status = "toolchain_missing"
    elif bad:
        status = "fail"
    elif unchecked:
        status = "error"
    else:
        status = "ok"
    lim = [AGENT_NOTE] if len(have) < len(PROVIDERS) else []
    return ToolResult("providers", status, summary, {"providers": rows},
                      lim + ["Each check is one read-only API call; nothing is created or deleted.",
                       "A valid key does not mean the provider has GPUs in stock."])
