"""Tests for the plain terminal output.

The summary is what a user reads after a run, so the things it must not
do are more important than the things it does: it must not list a
refused write as a change, must not call a shell command a file, and
must not say "finished" without saying what happened.

Rendering goes to a StringIO rather than stdout so a test can read back
exactly what the user would have seen.
"""

import io

from agent import report
from agent.runs import Run, ToolCall


def make_run(**kwargs) -> Run:
    """Return a finished run, ready to be summarised."""
    run = Run(task="do the thing", model="fake", provider="test", started=1000.0)
    run.finished = 1007.0
    run.steps_used = 3
    run.outcome = "finished"
    run.answer = "Done."
    for key, value in kwargs.items():
        setattr(run, key, value)
    return run


def call(name: str, status: str = "ok", **args) -> ToolCall:
    """Return one tool call for a run."""
    return ToolCall(step=1, name=name, args=args, status=status, seconds=0.1)


def render(run: Run) -> str:
    """Return what `summary` would print, without printing it."""
    buffer = io.StringIO()
    report.summary(run, out=buffer)
    return buffer.getvalue()


# --- the shape of the summary ---------------------------------------------


def test_a_finished_run_says_so() -> None:
    text = render(make_run())
    assert "finished" in text
    assert "3 steps" in text
    assert "7s" in text


def test_the_task_is_not_needed_the_summary_speaks_for_itself() -> None:
    """Kept deliberately terse: the answer is printed above this."""
    text = render(make_run())
    assert "do the thing" not in text


# --- what counts as a change ----------------------------------------------


def test_a_written_file_is_listed_as_changed() -> None:
    run = make_run(calls=[call("write_file", path="a.py")])
    assert "a.py" in render(run)


def test_a_refused_write_is_not_listed_as_changed() -> None:
    """It did not happen. Listing it would misreport the run."""
    run = make_run(calls=[call("write_file", status="declined", path="a.py")])
    text = render(run)
    assert "Changed:" not in text
    assert "not done" in text


def test_a_blocked_write_is_not_listed_as_changed() -> None:
    run = make_run(calls=[call("run_command", status="blocked", command="rm -rf /")])
    assert "Changed:" not in render(run)


def test_a_command_is_not_reported_as_a_changed_file() -> None:
    """A regression: `python main.py` under "Changed:" reads as a file."""
    run = make_run(calls=[call("run_command", command="python main.py")])
    text = render(run)
    assert "Commands run:" in text
    assert "Changed:" not in text


def test_a_command_that_failed_is_still_listed_as_run() -> None:
    """It ran, and a summary that hid it would hide the only bad news.

    Regression from recording a non-zero exit as `error`: filtering this
    list on `ok` alone dropped the failing command from the report
    entirely, which is the opposite of what marking it red was for.
    """
    run = make_run(calls=[call("run_command", status="error", command="pytest -q")])
    text = render(run)
    assert "Commands run:" in text
    assert "pytest -q" in text
    assert "1 error" in text
    assert "1 errors" not in text


def test_a_declined_command_is_not_listed_as_run() -> None:
    """Nothing ran. The refusal list is the honest record of it."""
    run = make_run(
        calls=[call("run_command", status="declined", command="pytest -q")]
    )
    text = render(run)
    assert "Commands run:" not in text
    assert "not done" in text


def test_the_same_file_twice_is_listed_once() -> None:
    """Edited five times is still one file changed."""
    calls = [call("edit_file", path="a.py") for _ in range(5)]
    text = render(make_run(calls=calls))
    assert text.count("    a.py") == 1


# --- refusals are surfaced -------------------------------------------------


def test_a_declined_action_is_reported_with_who_declined() -> None:
    run = make_run(
        calls=[call("edit_file", status="declined", path="a.py", old="x", new="y")]
    )
    text = render(run)
    assert "you declined" in text
    assert "1 declined" in text


def test_a_blocked_action_is_reported_as_the_blocklist() -> None:
    run = make_run(calls=[call("run_command", status="blocked", command="rm -rf /")])
    text = render(run)
    assert "blocklist" in text


def test_a_run_that_looks_finished_shows_its_refusals() -> None:
    """The whole reason refusals are in the summary.

    A run can end with outcome "finished" and still have declined
    three edits. Without this the two are indistinguishable.
    """
    calls = [call("edit_file", status="declined", path=f"{n}.py") for n in range(3)]
    text = render(make_run(calls=calls, outcome="finished"))
    assert "finished" in text
    assert "3 declined" in text


# --- unhappy endings -------------------------------------------------------


def test_hitting_the_step_limit_says_how_to_raise_it() -> None:
    text = render(make_run(outcome="step_limit"))
    assert "AGENT_MAX_STEPS" in text


def test_repeating_suggests_rephrasing() -> None:
    text = render(make_run(outcome="repeating"))
    assert "Rephrasing" in text


def test_a_failed_run_does_not_claim_success() -> None:
    text = render(make_run(outcome="failed"))
    assert "finished" not in text.split("\n")[1]


# --- live streaming -------------------------------------------------------


def test_each_call_is_printed_once_not_once_per_notification() -> None:
    """The observer fires on every progress event, not once per call.

    A plain function taking the run would reprint the whole log on each
    notification and bury the output in duplicates.
    """
    buffer = io.StringIO()
    log = report.LiveLog(out=buffer)
    run = make_run(calls=[])
    log(run)
    run.calls.append(call("read_file", path="a.py"))
    log(run)
    log(run)  # a second notification with nothing new
    run.calls.append(call("read_file", path="b.py"))
    log(run)
    text = buffer.getvalue()
    assert text.count("a.py") == 1
    assert text.count("b.py") == 1


def test_a_streamed_line_says_how_long_the_call_took() -> None:
    buffer = io.StringIO()
    run = make_run(calls=[call("read_file", path="a.py")])
    report.LiveLog(out=buffer)(run)
    assert "0.10s" in buffer.getvalue()


def test_a_refusal_is_visible_while_streaming_not_just_at_the_end() -> None:
    """Otherwise a decline is invisible until the run is over."""
    buffer = io.StringIO()
    run = make_run(calls=[call("write_file", status="declined", path="a.py")])
    report.LiveLog(out=buffer)(run)
    assert "declined" in buffer.getvalue()


# --- listing past runs -----------------------------------------------------


def test_an_empty_history_says_where_it_would_look() -> None:
    buffer = io.StringIO()
    report.listing([], out=buffer)
    assert "No runs recorded" in buffer.getvalue()


def test_the_listing_shows_the_outcome_and_the_task() -> None:
    buffer = io.StringIO()
    report.listing([make_run()], out=buffer)
    text = buffer.getvalue()
    assert "finished" in text
    assert "do the thing" in text


def test_the_listing_caps_a_very_long_task() -> None:
    """A single line per run, or the table stops being scannable."""
    buffer = io.StringIO()
    run = make_run(task="x" * 400)
    report.listing([run], out=buffer)
    assert "..." in buffer.getvalue()
    assert "x" * 400 not in buffer.getvalue()


def test_the_listing_shows_the_id_that_run_takes() -> None:
    """`--run` takes an id, so the listing has to show one.

    The two used to disagree: `--run` documented its argument as "the
    first column of `--history`" and the first column was the timestamp.
    Nothing about that is a crash, which is what made it easy to miss -
    the feature simply could not be used.
    """
    buffer = io.StringIO()
    report.listing([make_run(run_id="20260930-141203-a7f1")], out=buffer)
    assert "20260930-141203-a7f1" in buffer.getvalue()


def test_the_listing_still_identifies_a_run_with_no_file_behind_it() -> None:
    """A run in progress has no id yet, and must not print a bare gap."""
    buffer = io.StringIO()
    report.listing([make_run(run_id="")], out=buffer)
    text = buffer.getvalue()
    assert "do the thing" in text
    assert text.splitlines()[-1].split()[0], "the row starts with nothing"


def test_detail_names_the_run_so_a_copy_of_it_can_be_found() -> None:
    buffer = io.StringIO()
    report.detail(make_run(run_id="20260930-141203-a7f1"), out=buffer)
    assert "20260930-141203-a7f1" in buffer.getvalue()


def test_detail_shows_every_call_and_the_answer() -> None:
    buffer = io.StringIO()
    run = make_run(
        calls=[call("read_file", path="a.py"), call("write_file", path="b.py")]
    )
    report.detail(run, out=buffer)
    text = buffer.getvalue()
    assert "a.py" in text
    assert "b.py" in text
    assert "Done." in text
    assert "fake" in text


def test_detail_says_so_when_a_run_made_no_calls() -> None:
    buffer = io.StringIO()
    report.detail(make_run(calls=[]), out=buffer)
    assert "No tool calls" in buffer.getvalue()


def test_detail_shows_the_error_of_a_failed_run() -> None:
    buffer = io.StringIO()
    report.detail(make_run(outcome="failed", error="ConnectionError: reset"), out=buffer)
    assert "ConnectionError" in buffer.getvalue()
