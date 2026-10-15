"""Provider API keys: lookup order, login/logout through a fake keyring, ``providers``.

No real keyring and no network: ``keyring`` is a fake module in ``sys.modules`` and every
provider call goes through a fake request function.
"""

from __future__ import annotations

import io
import sys
import types
import urllib.error

import pytest

from oneapi_agent import cli, keys, remote

SECRET = "sk-very-secret-test-key-123"
ENV_NAMES = ("RUNPOD_API_KEY", "VAST_API_KEY", "VULTR_API_KEY", "OCEAN_API_KEY", "DIGITALOCEAN_TOKEN")


class FakeKeyring(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("keyring")
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service, user):
        return self.store.get((service, user))

    def set_password(self, service, user, password):
        self.store[(service, user)] = password

    def delete_password(self, service, user):
        del self.store[(service, user)]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for n in ENV_NAMES:
        monkeypatch.delenv(n, raising=False)


@pytest.fixture
def kr(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)
    return fake


@pytest.fixture
def no_keyring(monkeypatch):
    monkeypatch.setitem(sys.modules, "keyring", None)  # import raises ImportError


def run_cli(argv, capsys, stdin_text=None, monkeypatch=None):
    if stdin_text is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out + out.err


# ---------------------------------------------------------------- lookup order

def test_lookup_order_argument_then_env_then_keyring(kr, monkeypatch):
    assert keys.find_key("runpod") is None
    kr.store[("oneapi-agent", "runpod")] = "from-keyring"
    assert keys.find_key("runpod") == ("from-keyring", "keyring")
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    assert keys.find_key("runpod") == ("from-env", "RUNPOD_API_KEY")
    assert keys.find_key("runpod", "explicit") == ("explicit", "argument")


def test_digitalocean_alias_env_and_keyring_per_provider(kr, monkeypatch):
    kr.store[("oneapi-agent", "vast")] = "vast-key"
    assert keys.find_key("digitalocean") is None
    monkeypatch.setenv("DIGITALOCEAN_TOKEN", "do-token")
    assert keys.find_key("digitalocean") == ("do-token", "DIGITALOCEAN_TOKEN")
    monkeypatch.setenv("OCEAN_API_KEY", "ocean")
    assert keys.find_key("digitalocean") == ("ocean", "OCEAN_API_KEY")
    assert keys.find_key("vast") == ("vast-key", "keyring")


def test_lookup_without_keyring_or_with_broken_backend(no_keyring, monkeypatch):
    assert keys.find_key("vultr") is None
    monkeypatch.setenv("VULTR_API_KEY", "v")
    assert keys.find_key("vultr") == ("v", "VULTR_API_KEY")


def test_broken_backend_is_ignored(kr, monkeypatch):
    def boom(*a):
        raise RuntimeError("no backend")

    monkeypatch.setattr(kr, "get_password", boom)
    assert keys.find_key("runpod") is None


def test_make_provider_and_remote_test_use_the_keyring(kr, tmp_path):
    kr.store[("oneapi-agent", "vultr")] = "kr-vultr"
    prov = remote.make_provider("vultr")
    assert prov.api.key == "kr-vultr"
    kr.store.clear()
    res = remote.remote_test(str(tmp_path / "task.json"), provider="vultr")
    assert res.status == "toolchain_missing"
    assert "oneapi-agent login vultr" in " ".join(res.limitations)
    assert "VULTR_API_KEY" in res.summary


def test_make_provider_digitalocean_alias(no_keyring, monkeypatch):
    monkeypatch.setenv("DIGITALOCEAN_TOKEN", "tok")
    assert remote.make_provider("digitalocean").api.key == "tok"


# ---------------------------------------------------------------- login / logout

def test_login_from_stdin_stores_in_keyring_and_never_prints_key(kr, capsys, monkeypatch):
    code, out = run_cli(["login", "runpod"], capsys, SECRET + "\n", monkeypatch)
    assert code == 0
    assert kr.store[("oneapi-agent", "runpod")] == SECRET
    assert SECRET not in out


def test_login_from_hidden_prompt_on_a_tty(kr, capsys, monkeypatch):
    stdin = io.StringIO("")
    stdin.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(sys, "stdin", stdin)
    prompts = []
    monkeypatch.setattr(cli.getpass, "getpass", lambda p: prompts.append(p) or SECRET)
    code, out = run_cli(["login", "vast"], capsys)
    assert code == 0 and prompts and "hidden" in prompts[0]
    assert kr.store[("oneapi-agent", "vast")] == SECRET
    assert SECRET not in out


def test_login_empty_key_saves_nothing(kr, capsys, monkeypatch):
    code, out = run_cli(["login", "vultr"], capsys, "\n", monkeypatch)
    assert code == 2 and not kr.store


def test_logout_removes_key_and_is_idempotent(kr, capsys):
    kr.store[("oneapi-agent", "digitalocean")] = SECRET
    code, out = run_cli(["logout", "digitalocean"], capsys)
    assert code == 0 and not kr.store and "removed" in out
    code, out = run_cli(["logout", "digitalocean"], capsys)
    assert code == 0 and "no saved" in out


def test_login_without_keyring_explains_install_and_reads_no_key(no_keyring, capsys, monkeypatch):
    stdin = io.StringIO(SECRET + "\n")
    monkeypatch.setattr(sys, "stdin", stdin)
    code, out = run_cli(["login", "runpod"], capsys)
    assert code == 2
    assert "oneapi-agent-toolkit[keyring]" in out and "RUNPOD_API_KEY" in out
    assert stdin.read() == SECRET + "\n"  # the key was never consumed
    code, out = run_cli(["logout", "runpod"], capsys)
    assert code == 2 and "keyring" in out


# ---------------------------------------------------------------- providers

def http_error(code):
    return urllib.error.HTTPError("u", code, "Unauthorized", {}, io.BytesIO(b"{}"))


def fake_req(valid_keys):
    """Answers like each provider's account endpoint; 401 for keys not in ``valid_keys``."""
    calls = []

    def req(method, url, key, body):
        calls.append((method, url, body))
        assert method == "GET" or "graphql" in url  # read-only calls only
        if key not in valid_keys:
            if "graphql" in url:
                return {"errors": [{"message": "Unauthorized"}]}
            raise http_error(401)
        if "runpod" in url:
            return {"data": {"myself": {"clientBalance": 12.345, "currentSpendPerHr": 0,
                                        "spendLimit": 80}}}
        if "vast" in url:
            return {"credit": 4.5, "balance": 0, "email": "someone@example.com"}
        if "vultr" in url:
            return {"account": {"balance": -10, "pending_charges": 1.25, "email": "x@y"}}
        return {"account": {"droplet_limit": 25, "status": "active", "gpu_droplet_limit": 1}}

    req.calls = calls
    return req


def test_providers_valid_keys_show_source_and_account_facts(kr, monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    kr.store[("oneapi-agent", "digitalocean")] = SECRET + "-do"
    req = fake_req({SECRET, SECRET + "-do"})
    res = keys.providers(req)
    rows = res.evidence["providers"]
    assert res.status == "ok"
    assert rows["runpod"]["source"] == "RUNPOD_API_KEY" and rows["runpod"]["valid"] is True
    assert rows["runpod"]["account"]["balance_usd"] == 12.35
    assert rows["digitalocean"]["source"] == "keyring"
    assert rows["digitalocean"]["account"] == {"status": "active", "droplet_limit": 25,
                                               "gpu_droplet_limit": 1}
    assert rows["vast"]["key"] == "missing" and "oneapi-agent login vast" in rows["vast"]["how"]
    assert len(req.calls) == 2
    assert SECRET not in res.to_json() and SECRET not in res.summary


def test_providers_invalid_key_fails_without_leaking_it(kr, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", SECRET)
    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    res = keys.providers(fake_req(set()))
    rows = res.evidence["providers"]
    assert res.status == "fail"
    assert rows["vast"]["valid"] is False and rows["vast"]["error"] == "HTTP 401"
    assert rows["runpod"]["valid"] is False
    assert "vast: INVALID (VAST_API_KEY)" in res.summary
    assert SECRET not in res.to_json()


def test_providers_unreachable_is_unchecked(kr, monkeypatch):
    monkeypatch.setenv("VULTR_API_KEY", SECRET)

    def down(method, url, key, body):
        raise urllib.error.URLError("no route")

    res = keys.providers(down)
    assert res.status == "error"
    assert res.evidence["providers"]["vultr"]["valid"] is None


def test_providers_no_key_anywhere(no_keyring, capsys, monkeypatch):
    def never(*a):
        raise AssertionError("no key, so no API call")

    monkeypatch.setattr(keys, "_http", never)
    monkeypatch.setattr(keys.providers, "__defaults__", (never,))
    code, out = run_cli(["--summary", "providers"], capsys)
    assert code == 2
    assert out.strip() == ("[providers] toolchain_missing: runpod: no key; vast: no key; "
                           "vultr: no key; digitalocean: no key")


def test_providers_cli_json_never_contains_key(kr, capsys, monkeypatch):
    kr.store[("oneapi-agent", "vultr")] = SECRET
    monkeypatch.setattr(keys.providers, "__defaults__", (fake_req({SECRET}),))
    code, out = run_cli(["providers"], capsys)
    assert code == 0
    assert '"balance_usd": -10.0' in out and "x@y" not in out
    assert SECRET not in out
