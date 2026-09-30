"""Tests for run history.

Two kinds of thing are checked here. The first is that a run survives a
round trip through the disk, because a log nobody can read back is
worthless. The second is that a broken run is still readable: the
history exists precisely to explain a crash, so a run that died
mid-write has to load as far as it got.

Nothing here touches the network or a real clock that matters. `tmp_path`
keeps the files out of the developer's own `.marth-ai` directory.
"""

import json
from pathlib import Path

import pytest

from agent import config, history
from agent.runs import Run, ToolCall


def make_run(task: str = "do the thing", model: str = "fake-model") -> Run:
    """Return a finished run record with one call, ready to store."""
    run = Run(task=task, model=model, provider="test", started=1_700_000_000.0)
    run.calls.append(
        ToolCall(
            step=1,
            name="write_file",
            args={"path": "a.py", "content": "x = 1"},
            status="ok",
            output="Wrote a.py",
            seconds=0.5,
        )
    )
    run.outcome = "finished"
    run.answer = "Wrote the file."
    run.finished = 1_700_000_005.0
    return run


def store(run: Run, root, **kwargs) -> history.RunLog:
    """Write a run to disk under `root` and return the log."""
    log = history.RunLog(run, directory=history.runs_dir(root))
    log.open()
    for call in run.calls:
        log.write_call(call)
    log.close(run.answer, run.outcome, **kwargs)
    return log


# --- round trip ------------------------------------------------------------


def test_a_run_survives_a_round_trip(tmp_path) -> None:
    """The whole point of storing it."""
    original = make_run()
    store(original, tmp_path)
    files = history.list_runs(tmp_path)
    assert len(files) == 1
    loaded = history.load_run(files[0])
    assert loaded.task == original.task
    assert loaded.model == original.model
    assert loaded.answer == original.answer
    assert loaded.outcome == original.outcome


def test_every_tool_call_comes_back(tmp_path) -> None:
    """Losing calls would defeat the purpose of a history viewer."""
    run = make_run()
    for n in range(2, 6):
        run.calls.append(
            ToolCall(step=n, name="read_file", args={"path": f"{n}.py"})
        )
    store(run, tmp_path)
    loaded = history.load_run(history.list_runs(tmp_path)[0])
    assert [call.name for call in loaded.calls] == [
        "write_file",
        "read_file",
        "read_file",
        "read_file",
        "read_file",
    ]
    assert loaded.calls[0].args == {"path": "a.py", "content": "x = 1"}


def test_a_refused_call_is_recorded_as_refused(tmp_path) -> None:
    """Reviewing a run is most useful when something was refused."""
    run = make_run()
    run.calls.append(
        ToolCall(
            step=2,
            name="write_file",
            args={"path": ".env", "content": "KEY=1"},
            status="declined",
            output="Declined by the user.",
            approved=False,
        )
    )
    store(run, tmp_path)
    loaded = history.load_run(history.list_runs(tmp_path)[0])
    assert loaded.calls[1].status == "declined"
    assert loaded.calls[1].approved is False


def test_timings_survive(tmp_path) -> None:
    """A viewer sorts by duration, so it cannot be recomputed later."""
    store(make_run(), tmp_path)
    loaded = history.load_run(history.list_runs(tmp_path)[0])
    assert loaded.calls[0].seconds == 0.5
    assert loaded.finished == 1_700_000_005.0


def test_the_step_count_survives(tmp_path) -> None:
    """It is not derivable from the calls.

    A step can make several calls, and the last step of a run often
    makes none at all, because the model replied with prose instead.
    """
    run = make_run()
    run.steps_used = 7  # one call recorded, seven steps taken
    store(run, tmp_path)
    loaded = history.load_run(history.list_runs(tmp_path)[0])
    assert loaded.steps_used == 7


def test_the_file_is_json_lines(tmp_path) -> None:
    """One line per event, so a torn write costs one line, not the file."""
    store(make_run(), tmp_path)
    lines = history.list_runs(tmp_path)[0].read_text().strip().split("\n")
    assert len(lines) == 3  # header, one call, footer
    assert all(json.loads(line) for line in lines)


# --- crashes ---------------------------------------------------------------


def test_a_run_with_no_footer_loads_as_still_running(tmp_path) -> None:
    """The state a crashed run is left in. It must be readable, not raise."""
    run = make_run()
    log = history.RunLog(run, directory=history.runs_dir(tmp_path))
    log.open()
    log.write_call(run.calls[0])
    loaded = history.load_run(history.list_runs(tmp_path)[0])
    assert loaded.outcome == "running"
    assert len(loaded.calls) == 1


def test_a_torn_final_line_does_not_hide_the_run(tmp_path) -> None:
    """A crash mid-write leaves half a line. Everything before it counts."""
    run = make_run()
    log = history.RunLog(run, directory=history.runs_dir(tmp_path))
    log.open()
    log.write_call(run.calls[0])
    path = history.list_runs(tmp_path)[0]
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "call", "name": "run_com')  # cut off mid-write
    loaded = history.load_run(path)
    assert len(loaded.calls) == 1
    assert loaded.task == run.task


def test_damage_in_the_middle_is_reported_not_swallowed(tmp_path) -> None:
    """Quietly ignoring corruption would hide a real problem.

    The bad line has to have a good line after it, or it is just the
    torn-last-line case above and is tolerated on purpose.
    """
    run = make_run()
    log = history.RunLog(run, directory=history.runs_dir(tmp_path))
    log.open()
    path = history.list_runs(tmp_path)[0]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json at all\n")
        handle.write(
            json.dumps({"type": "footer", "outcome": "finished", "answer": "x"})
            + "\n"
        )
    with pytest.raises(ValueError, match="line 2"):
        history.load_run(path)


def test_a_file_with_no_header_is_not_a_run(tmp_path) -> None:
    run = make_run()
    log = history.RunLog(run, directory=history.runs_dir(tmp_path))
    log.open()
    path = history.list_runs(tmp_path)[0]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "call", "name": "x"}) + "\n")
    with pytest.raises(ValueError, match="no header"):
        history.load_run(path)


def test_an_error_is_recorded_verbatim(tmp_path) -> None:
    """The point of the log is to explain a failure, so keep the text."""
    run = make_run()
    run.outcome = "failed"
    store(run, tmp_path, error="ConnectionError: connection reset")
    loaded = history.load_run(history.list_runs(tmp_path)[0])
    assert loaded.outcome == "failed"
    assert "connection reset" in loaded.error


# --- filenames and trimming ------------------------------------------------


def test_run_files_sort_newest_first(tmp_path) -> None:
    """The viewer relies on name order; names start with the timestamp."""
    for stamp in (1_700_000_000.0, 1_700_000_600.0, 1_700_000_300.0):
        run = make_run()
        run.started = stamp
        store(run, tmp_path)
    loaded = [
        history.load_run(path).started for path in history.list_runs(tmp_path)
    ]
    assert loaded == sorted(loaded, reverse=True)


def test_two_runs_in_the_same_second_do_not_merge(tmp_path) -> None:
    """A collision would produce one file holding two unrelated runs."""
    for _ in range(2):
        run = make_run()
        run.started = 1_700_000_000.0  # identical timestamp on purpose
        store(run, tmp_path)
    files = history.list_runs(tmp_path)
    assert len(files) == 2, "the second run overwrote the first"
    for path in files:
        assert len(history.load_run(path).calls) == 1


def test_old_runs_are_trimmed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """History that grows forever is a disk problem, not a feature."""
    monkeypatch.setattr(config, "MAX_HISTORY_RUNS", 3)
    for n in range(6):
        run = make_run()
        run.started = 1_700_000_000.0 + n * 60
        store(run, tmp_path)
    files = history.list_runs(tmp_path)
    assert len(files) == 3
    # The newest survive; the oldest are gone.
    assert history.load_run(files[0]).started == 1_700_000_000.0 + 5 * 60


def test_trimming_keeps_the_newest_not_the_first_found(tmp_path, monkeypatch) -> None:
    """Deleting by list order would quietly drop the wrong end."""
    monkeypatch.setattr(config, "MAX_HISTORY_RUNS", 1)
    for n in range(3):
        run = make_run()
        run.started = 1_700_000_000.0 + n * 60
        store(run, tmp_path)
    remaining = history.list_runs(tmp_path)
    assert len(remaining) == 1
    assert history.load_run(remaining[0]).started == 1_700_000_000.0 + 2 * 60


# --- being a good guest ----------------------------------------------------


def test_nothing_is_written_when_history_is_off(tmp_path, monkeypatch) -> None:
    """A one-off run on someone else's repo should leave no trace."""
    monkeypatch.setattr(config, "HISTORY_ENABLED", False)
    store(make_run(), tmp_path)
    assert history.list_runs(tmp_path) == []
    assert not (tmp_path / config.HISTORY_DIRNAME).exists()


def test_no_directory_is_created_when_nothing_ran(tmp_path) -> None:
    """Reading history on a clean checkout must not make a directory."""
    assert history.list_runs(tmp_path) == []
    assert not (tmp_path / config.HISTORY_DIRNAME).exists()


def test_the_history_directory_is_gitignored() -> None:
    """Otherwise every run shows up as an untracked file in `git status`.

    Checks the repository's own .gitignore rather than a fixture's, since
    a tmp_path copy would only prove the test agrees with itself.
    """
    repo_ignore = Path(__file__).resolve().parent.parent / ".gitignore"
    entries = {
        line.strip().rstrip("/")
        for line in repo_ignore.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert config.HISTORY_DIRNAME in entries, (
        f"{config.HISTORY_DIRNAME} is not in .gitignore; run files would "
        "show up as untracked"
    )


def test_a_write_failure_does_not_stop_the_run(tmp_path) -> None:
    """Losing the log must never cost the user their work.

    A read-only workspace is the realistic version of this: the log
    cannot be written, and the agent still has to do the task. The
    blocker is a directory sitting where the file needs to go, since
    creating a deep path with `mkdir(parents=True)` would succeed.
    """
    run = make_run()
    log = history.RunLog(run, directory=tmp_path)
    (tmp_path / log.path.name).mkdir()  # occupy the filename
    log.open()  # must not raise
    log.write_call(run.calls[0])
    log.close("answer", "finished")  # must not raise


def test_clearing_removes_everything_and_reports_how_many(tmp_path) -> None:
    for n in range(3):
        run = make_run()
        run.started = 1_700_000_000.0 + n * 60
        store(run, tmp_path)
    assert history.clear(tmp_path) == 3
    assert history.list_runs(tmp_path) == []


def test_clearing_an_empty_history_is_not_an_error(tmp_path) -> None:
    assert history.clear(tmp_path) == 0


def test_run_files_are_the_only_ones_listed(tmp_path) -> None:
    """A stray file in the directory must not become a run."""
    store(make_run(), tmp_path)
    (history.runs_dir(tmp_path) / "notes.txt").write_text("hello")
    assert len(history.list_runs(tmp_path)) == 1
