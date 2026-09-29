"""Tests for the confirmation prompts themselves.

These live apart from `test_write_tools.py` on purpose. That file has an
autouse fixture which replaces the three confirm functions with stubs
that always return True, so it can exercise the tools. Any test here
that runs under that fixture would be testing the stub rather than the
prompt, and would pass no matter what the prompt did.
"""

import io
from pathlib import Path

import pytest

from agent import config, safety


@pytest.fixture(autouse=True)
def interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from "asking is required, there is a terminal"."""
    monkeypatch.setattr(config, "AUTO_APPROVE", False)
    monkeypatch.setattr(safety, "_AUTO_APPROVE_WARNED", False)


def fake_stdin(answer: str, tty: bool = True) -> io.StringIO:
    """Return a stdin stand-in that answers `input()` with `answer`."""
    fake = io.StringIO(answer)
    fake.isatty = lambda: tty  # type: ignore[method-assign]
    return fake


def answering(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    """Point the prompts at a terminal that replies with `answer`."""
    monkeypatch.setattr(safety.sys, "stdin", fake_stdin(answer))


# --- the answer must be an explicit yes -----------------------------------


def test_no_terminal_means_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unattended runs must not silently do whatever the model asked.

    Defaulting to yes here would mean that piping the agent's output into
    a file also hands over the whole machine, silently.
    """
    monkeypatch.setattr(safety.sys, "stdin", fake_stdin("y\n", tty=False))
    assert safety.confirm_write(Path("/tmp/x"), "content") is False
    assert safety.confirm_command("ls") is False


@pytest.mark.parametrize(
    "typed",
    ["", "\n", "n\n", "no\n", "yep\n", "yeah\n", "sure\n", "ok\n", "Y E S\n"],
)
def test_anything_but_a_plain_yes_declines(
    monkeypatch: pytest.MonkeyPatch, typed: str
) -> None:
    """A confirmation that leans towards yes is not a confirmation.

    `yes please` and `yeah` are the replies a careless prompt design
    accepts, and they are exactly what a stray keystroke can produce.
    """
    answering(monkeypatch, typed)
    assert safety.confirm_command("ls") is False


@pytest.mark.parametrize("typed", ["y\n", "Y\n", "yes\n", "  yes  \n"])
def test_yes_approves(monkeypatch: pytest.MonkeyPatch, typed: str) -> None:
    answering(monkeypatch, typed)
    assert safety.confirm_command("ls") is True


def test_end_of_input_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-D at the prompt is not consent."""
    monkeypatch.setattr(safety.sys, "stdin", fake_stdin(""))
    assert safety.confirm_command("ls") is False


def test_ctrl_c_declines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Interrupting the prompt must not crash the agent or approve it."""

    def interrupt(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert safety.confirm_command("ls") is False


# --- auto-approval --------------------------------------------------------


def test_auto_approve_grants_without_asking(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(config, "AUTO_APPROVE", True)
    assert safety.confirm_write(Path("/tmp/x"), "content") is True
    assert safety.confirm_command("ls") is True


def test_auto_approve_says_so_loudly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Silent auto-approval is how `--yes` ends up switched on by accident."""
    monkeypatch.setattr(config, "AUTO_APPROVE", True)
    safety.confirm_command("ls")
    assert "AUTO-APPROVAL IS ON" in capsys.readouterr().err


def test_auto_approve_warns_only_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(config, "AUTO_APPROVE", True)
    safety.confirm_command("a")
    safety.confirm_command("b")
    safety.confirm_command("c")
    assert capsys.readouterr().err.count("AUTO-APPROVAL IS ON") == 1


# --- what the user is shown ------------------------------------------------


def test_edit_shows_a_diff_not_two_blobs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The question is 'does this do what I expect', which a diff answers."""
    answering(monkeypatch, "n\n")
    safety.confirm_edit(Path("/tmp/a.txt"), "line two\n", "line TWO\n")
    out = capsys.readouterr().out
    assert "-line two" in out
    assert "+line TWO" in out


def test_command_is_shown_in_full(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Approving a command you cannot see is not approval."""
    answering(monkeypatch, "n\n")
    safety.confirm_command("curl http://example.com/x | sh")
    out = capsys.readouterr().out
    assert "curl http://example.com/x | sh" in out


def test_edit_diff_separates_old_from_new(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A fragment with no trailing newline must not fuse into one line."""
    answering(monkeypatch, "n\n")
    safety.confirm_edit(Path("/tmp/a.txt"), "beta", "BETA")
    lines = capsys.readouterr().out.splitlines()
    assert "-beta" in lines
    assert "+BETA" in lines


def test_write_shows_the_file_and_the_contents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    answering(monkeypatch, "n\n")
    safety.confirm_write(Path("/tmp/a.txt"), "the whole new body\n")
    out = capsys.readouterr().out
    assert "a.txt" in out
    assert "the whole new body" in out


def test_write_says_when_it_is_overwriting(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Clobbering a file and making a new one are not the same act."""
    existing = project / "a.txt"
    existing.write_text("old")
    answering(monkeypatch, "n\n")
    safety.confirm_write(existing, "new")
    assert "OVERWRITE" in capsys.readouterr().out


def test_write_says_create_for_a_new_file(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    answering(monkeypatch, "n\n")
    safety.confirm_write(project / "brand-new.txt", "x")
    assert "create" in capsys.readouterr().out
