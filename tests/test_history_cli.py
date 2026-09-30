"""Tests for the two commands that read the history back.

`--history` and `--run` are the only way to find out what a run did after
the fact, and both of them were shipped with a bug that no other test
could see: `--run`'s help said its argument was "the first column of
`--history`", and the first column was the timestamp. The feature was
unusable and nothing was red.

So these tests are about the two agreeing with each other, and about what
happens when a run id is ambiguous. A prefix match that silently returns
the *newest* of several runs is worse than one that refuses: the user
asked about one run and would read another's output as though it were
this one's.

They drive `main.show_history` and `main.show_run` rather than the
renderers underneath, because the disagreement above was between the two
commands, not inside either one.
"""

from pathlib import Path

import pytest

from agent import config, history, main
from agent.runs import Run, ToolCall


def store(root: Path, task: str, started: float, outcome: str = "finished") -> Path:
    """Write one finished run under `root` and return its file."""
    run = Run(task=task, model="fake-model", provider="test", started=started)
    run.calls.append(
        ToolCall(
            step=1,
            name="read_file",
            args={"path": "a.py"},
            status="ok",
            output="x = 1",
            seconds=0.1,
        )
    )
    run.outcome = outcome
    run.answer = "Read it."
    run.steps_used = 1
    run.finished = started + 1.0
    log = history.RunLog(run, directory=history.runs_dir(root))
    log.open()
    log.write_call(run.calls[0])
    log.close(run.answer, run.outcome)
    return log.path


@pytest.fixture
def two_runs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Two runs an hour apart, so their ids share a date but not a time."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    first = store(tmp_path, "the first task", 1_700_000_000.0)
    second = store(tmp_path, "the second task", 1_700_003_600.0)
    return {"first": first, "second": second}


# --- --history -------------------------------------------------------------


def test_history_lists_every_run(two_runs, capsys: pytest.CaptureFixture) -> None:
    assert main.show_history() == 0
    out = capsys.readouterr().out
    assert "the first task" in out
    assert "the second task" in out


def test_history_shows_the_id_run_needs(two_runs, capsys: pytest.CaptureFixture) -> None:
    """The bug: the listing had no id, so there was nothing to pass to `--run`."""
    main.show_history()
    out = capsys.readouterr().out
    assert two_runs["first"].stem in out
    assert two_runs["second"].stem in out


def test_history_lists_the_newest_run_first(two_runs, capsys) -> None:
    """Nobody scrolls a history file to find the run they just made."""
    main.show_history()
    out = capsys.readouterr().out
    assert out.index("the second task") < out.index("the first task")


def test_history_shows_the_outcome_not_just_the_task(
    two_runs, capsys
) -> None:
    """A run that gave up and a run that finished must be distinguishable."""
    main.show_history()
    assert "finished" in capsys.readouterr().out


def test_history_says_so_when_there_is_nothing(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A project that has never run the agent has no history, and that is fine."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    assert main.show_history() == 0
    assert "No runs recorded" in capsys.readouterr().out


def test_one_unreadable_file_does_not_hide_the_others(
    two_runs, capsys: pytest.CaptureFixture
) -> None:
    """200 runs, one of them damaged: the other 199 are still worth reading."""
    (two_runs["first"].parent / "garbage.jsonl").write_text("not json at all\n")
    assert main.show_history() == 0
    captured = capsys.readouterr()
    assert "the second task" in captured.out
    assert "skipping" in captured.err


# --- --run -----------------------------------------------------------------


def test_run_prints_the_run_whose_id_was_given(
    two_runs, capsys: pytest.CaptureFixture
) -> None:
    assert main.show_run(two_runs["first"].stem) == 0
    out = capsys.readouterr().out
    assert "the first task" in out
    assert "the second task" not in out, "showed the wrong run"


def test_run_shows_the_full_output_not_the_one_line_summary(
    two_runs, capsys
) -> None:
    """The one-line summary is for the live log. The point of `--run` is the rest."""
    main.show_run(two_runs["first"].stem)
    out = capsys.readouterr().out
    assert "read_file" in out
    assert "x = 1" in out


def test_run_accepts_a_prefix(two_runs, capsys) -> None:
    """The timestamp part of an id is guessable, so a prefix is allowed."""
    stem = two_runs["second"].stem
    assert main.show_run(stem[:-5]) == 0
    assert "the second task" in capsys.readouterr().out


def test_run_refuses_an_ambiguous_prefix(two_runs, capsys) -> None:
    """Two runs an hour apart share a date, so a short prefix is not unique.

    Showing the newest one silently would mean the user reads one run's
    tool calls believing they are another's. Better to say which runs
    matched and stop.
    """
    shared = two_runs["first"].stem[:8]  # the date, common to both
    assert main.show_run(shared) == 1
    out = capsys.readouterr().out
    assert "matches 2 runs" in out
    assert two_runs["first"].stem in out
    assert two_runs["second"].stem in out
    assert "x = 1" not in out, "showed a run it had already said was ambiguous"


def test_run_says_so_when_nothing_matches(
    two_runs, capsys: pytest.CaptureFixture
) -> None:
    assert main.show_run("no-such-run") == 1
    assert "No run matching" in capsys.readouterr().out


def test_run_does_not_leave_a_damaged_file_hanging(
    two_runs, capsys
) -> None:
    """A file that is not a run file is reported, not shown as an empty run."""
    bad = two_runs["first"].parent / "20991231-235959-dead.jsonl"
    bad.write_text("{}\n")
    assert main.show_run(bad.stem) == 1
    assert "dead" in capsys.readouterr().err


def test_a_run_with_no_footer_is_shown_as_running(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A file with no footer is a run that never got to say how it ended."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    run = Run(task="interrupted", model="m", provider="test", started=1_700_000_000.0)
    log = history.RunLog(run, directory=history.runs_dir(tmp_path))
    log.open()
    # Header only. A process killed mid-run leaves exactly this, and the
    # header deliberately does not carry an outcome because that is not
    # known until the end.
    history.list_runs(tmp_path)

    assert main.show_history() == 0
    assert "running" in capsys.readouterr().out
