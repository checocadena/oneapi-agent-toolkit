"""``oneapi-agent ci setup``: the workflow text and ``gh secret set`` (gh is always mocked)."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from oneapi_agent import ci, cli

SECRET = "sk-very-secret-ci-key-456"


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for n in ("RUNPOD_API_KEY", "VAST_API_KEY", "VULTR_API_KEY", "OCEAN_API_KEY", "DIGITALOCEAN_TOKEN"):
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setitem(sys.modules, "keyring", None)


class FakeGh:
    def __init__(self, logged_in=True, fail=False):
        self.logged_in, self.fail, self.calls = logged_in, fail, []

    def __call__(self, argv, **kw):
        assert argv[0] == "gh"
        self.calls.append((argv, kw))
        if argv[1:3] == ["auth", "status"]:
            return subprocess.CompletedProcess(argv, 0 if self.logged_in else 1, "", "")
        if self.fail:
            return subprocess.CompletedProcess(argv, 1, "", f"HTTP 404 echo {kw.get('input')}")
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def gh_installed(monkeypatch):
    monkeypatch.setattr(ci.shutil, "which", lambda name: "/usr/bin/gh" if name == "gh" else None)


def test_workflow_guards_forks_label_caps_and_uploads():
    text = ci.render_workflow("examples/fir/task.json", "vast", "nvidia", 1.5, 30)
    assert "github.event.pull_request.head.repo.full_name == github.repository" in text
    assert "contains(github.event.pull_request.labels.*.name, 'gpu-test')" in text
    assert "workflow_dispatch" in text and "pull_request_target" not in text
    assert "--max-usd 1.5 --max-minutes 30" in text
    assert "actions/upload-artifact" in text and "path: gpu-run/" in text
    assert "VAST_API_KEY: ${{ secrets.VAST_API_KEY }}" in text
    # dispatch inputs reach the shell through env vars only
    assert '"$TASK"' in text and "${{ inputs.task }}\"" not in text
    install_step = text.index("pip install")
    assert text.index("secrets.") > install_step  # pip install never sees the keys


def test_workflow_rejects_shell_metacharacters():
    with pytest.raises(ValueError):
        ci.render_workflow('t.json"; curl evil')


def test_setup_writes_workflow_and_refuses_overwrite(tmp_path):
    res = ci.ci_setup(str(tmp_path), task="t.json", secrets=False)
    wf = tmp_path / ".github/workflows/gpu-test.yml"
    assert res.ok and wf.read_text().startswith("# Written by `oneapi-agent ci setup`")
    assert ci.ci_setup(str(tmp_path), secrets=False).status == "error"
    assert ci.ci_setup(str(tmp_path), secrets=False, force=True).ok


def test_secrets_go_to_gh_on_stdin_after_asking(tmp_path, monkeypatch, gh_installed):
    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    monkeypatch.setenv("VULTR_API_KEY", SECRET + "-v")
    gh, asked = FakeGh(), []

    def ask(q):
        asked.append(q)
        return "runpod" in q

    res = ci.ci_setup(str(tmp_path), ask=ask, run=gh)
    assert res.evidence["secrets"] == {"RUNPOD_API_KEY": "set (from RUNPOD_API_KEY)",
                                       "VULTR_API_KEY": "skipped"}
    assert len(asked) == 2
    sets = [(a, kw) for a, kw in gh.calls if a[1] == "secret"]
    assert sets == [(["gh", "secret", "set", "RUNPOD_API_KEY"],
                     {"cwd": tmp_path, "input": SECRET, "capture_output": True, "text": True})]
    assert all(SECRET not in " ".join(a) for a, _ in gh.calls)  # never on a command line
    assert SECRET not in res.to_json()


def test_yes_stores_keyring_key_without_asking(tmp_path, monkeypatch, gh_installed):
    kr = types.ModuleType("keyring")
    kr.get_password = lambda s, u: SECRET if u == "digitalocean" else None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", kr)
    gh = FakeGh()

    def never(q):
        raise AssertionError("--yes must not ask")

    res = ci.ci_setup(str(tmp_path), yes=True, ask=never, run=gh)
    assert res.evidence["secrets"] == {"OCEAN_API_KEY": "set (from keyring)"}
    assert gh.calls[-1][1]["input"] == SECRET


def test_gh_failure_is_reported_without_the_key(tmp_path, monkeypatch, gh_installed):
    monkeypatch.setenv("VAST_API_KEY", SECRET)
    res = ci.ci_setup(str(tmp_path), yes=True, run=FakeGh(fail=True))
    assert res.evidence["secrets"]["VAST_API_KEY"].startswith("failed: HTTP 404")
    assert SECRET not in res.to_json()


def test_skips_when_gh_missing_or_logged_out_or_no_keys(tmp_path, monkeypatch, gh_installed):
    assert ci.ci_setup(str(tmp_path), run=FakeGh()).evidence["secrets"] == {
        "skipped": "no provider key found on this machine"}
    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    gh = FakeGh(logged_in=False)
    res = ci.ci_setup(str(tmp_path), force=True, yes=True, run=gh)
    assert "not logged in" in res.evidence["secrets"]["skipped"]
    assert [a for a, _ in gh.calls] == [["gh", "auth", "status"]]
    monkeypatch.setattr(ci.shutil, "which", lambda name: None)
    res = ci.ci_setup(str(tmp_path), force=True, yes=True, run=FakeGh())
    assert "not installed" in res.evidence["secrets"]["skipped"]
    assert any("gh secret set RUNPOD_API_KEY" in m for m in res.limitations)


def test_cli_ci_setup_without_a_tty_never_stores(tmp_path, monkeypatch, capsys, gh_installed):
    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    gh = FakeGh()
    monkeypatch.setattr(ci.subprocess, "run", gh)
    code = cli.main(["ci", "setup", "--repo", str(tmp_path), "--task", "t.json"])
    out = capsys.readouterr()
    assert code == 0 and SECRET not in out.out + out.err
    assert not any(a[1] == "secret" for a, _ in gh.calls)  # stdin is not a TTY: no answer = no
