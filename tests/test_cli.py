"""Tests for what the CLI says when something goes wrong.

Failures arrive as exceptions raised inside `llm.py`, and a stack trace
at the terminal is a bug rather than a report: the user cannot act on
one. These check that each ordinary failure is answered with the command
that fixes it, and that the answer names the right problem.

They also pin the ordering inside `explain_failure`, which matters
because the cases overlap. A 404 from a local server means "not
downloaded", while a 404 from a remote one means "wrong name", and the
first matching branch decides which the user is told.
"""

import pytest

from agent import config, main as cli


def test_a_missing_model_is_answered_with_setup() -> None:
    """Ollama reports a model it has not pulled as a plain 404.

    That reads exactly like a typo in the name, so the answer has to be
    the command that puts the model on the machine.
    """
    error = RuntimeError("HTTP 404 from provider: model 'x' not found")
    text = cli.explain_failure(error)
    assert "marth --setup" in text
    assert "not on this machine" in text


def test_a_refused_connection_is_answered_with_serve() -> None:
    """The other half of a first run: Ollama installed but not started."""
    error = RuntimeError("Could not reach a model server: Connection refused")
    text = cli.explain_failure(error)
    assert "ollama serve" in text
    assert config.OPENAI_BASE_URL in text


def test_an_unrecognised_failure_still_reports_something() -> None:
    """The fallback cannot know the fix, but it must not swallow the
    message either -- that would turn a diagnosable failure into a
    mystery."""
    text = cli.explain_failure(ValueError("something nobody predicted"))
    assert "something nobody predicted" in text


def test_list_models_without_a_server_reports_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--list-models` is usually typed before `--setup` has ever run, so
    nothing listening is the ordinary case here rather than an edge one.

    An unhandled exception from this path used to reach the terminal as a
    full traceback, which is the one thing a first-time user should never
    be shown.
    """
    # Port 9 is the discard port: nothing listens on it.
    monkeypatch.setattr(config, "OPENAI_BASE_URL", "http://127.0.0.1:9/v1")

    assert cli.main(["--list-models"]) == 1

    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "ollama serve" in err


class _BrokenLLM:
    """An `LLM` that fails the way a server does when the model is gone."""

    model_name = "missing-model"

    def send(
        self, history, tool_schemas=None
    ):  # type: ignore[no-untyped-def]
        """Raise rather than reply, as `OpenAICompatLLM` does on a 404."""
        raise RuntimeError("HTTP 404 from provider: model 'x' not found")

    def list_model_names(self) -> list[str]:  # type: ignore[no-untyped-def]
        """Never reached; present to satisfy the protocol."""
        return []


def test_a_server_failure_during_a_run_is_not_called_a_config_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every failure the provider raises is a `RuntimeError`, so a branch
    that special-cased `RuntimeError` labelled all of them the same way.
    A missing model is not a configuration error, and being told it was
    sends the reader looking in the wrong place.
    """
    text = cli.run_and_report(
        "do the thing", _BrokenLLM(), live=False, record=False
    )
    assert "Configuration error" not in text
    assert "marth --setup" in text
    # The summary is not printed for a run that never happened.
    assert "steps," not in capsys.readouterr().out
