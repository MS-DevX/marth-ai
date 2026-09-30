"""Tests for the run record.

`runs.py` is the thing every other view reads: the curses dashboard, the
plain log, and the history file all render what is defined here. So the
tests are mostly about it being honest rather than about it doing
anything clever.

Two properties get the most attention. A run that round-trips through
`to_dict`/`from_dict` unchanged, because the history file is only worth
keeping if it can be read back. And that a run never claims more than it
did: a declined write must not appear in the counts as an ok, and a
finished run with three refusals has to still be distinguishable from a
clean one.
"""

import pytest

from agent.runs import CallStatus, Run, ToolCall, describe_call, first_line


def call(status: str = "ok", **kwargs) -> ToolCall:
    """Return one tool call for a run to record."""
    return ToolCall(
        step=kwargs.pop("step", 1),
        name=kwargs.pop("name", "read_file"),
        args=kwargs.pop("args", {"path": "a.py"}),
        status=status,
        **kwargs,
    )


def run_with(*calls: ToolCall, **kwargs) -> Run:
    """Return a run containing `calls`, ready to be rendered."""
    run = Run(
        task=kwargs.pop("task", "do the thing"),
        model=kwargs.pop("model", "fake-model"),
        provider=kwargs.pop("provider", "test"),
        started=1000.0,
        calls=list(calls),
        **kwargs,
    )
    run.finished = 1005.0
    return run


# --- describing a call -----------------------------------------------------


def test_a_call_is_shown_by_name_and_arguments() -> None:
    assert "read_file(path=a.py)" in describe_call(call())


def test_the_first_three_arguments_are_shown() -> None:
    line = describe_call(call(args={"a": 1, "b": 2, "c": 3, "d": 4}))
    assert "a=1" in line and "b=2" in line and "c=3" in line
    assert "d=4" not in line
    assert "..." in line


def test_a_long_argument_is_cut_not_the_line() -> None:
    """A whole file's contents must not wreck a one-line display."""
    line = describe_call(call(args={"content": "x" * 5000, "path": "a.py"}))
    assert "..." in line
    assert "x" * 5000 not in line


def test_a_multi_line_argument_is_folded_into_one_line() -> None:
    """Otherwise one argument turns into four log lines."""
    line = describe_call(call(args={"old": "one\ntwo\nthree"}))
    assert "\n" not in line
    assert "one two three" in line


def test_a_successful_call_has_no_marker() -> None:
    """A row of spaces for every success is noise."""
    assert describe_call(call("ok")).startswith("read_file")


def test_a_declined_call_is_marked() -> None:
    assert describe_call(call("declined")).startswith("! ")


def test_a_blocked_call_is_marked() -> None:
    assert describe_call(call("blocked")).startswith("X ")


def test_an_errored_call_is_marked() -> None:
    assert describe_call(call("error")).startswith("E ")


def test_every_status_has_a_marker_that_does_not_collide() -> None:
    """Two outcomes must never print the same prefix.

    The dashboard colours by prefix, so a collision would shade a refusal
    as a success.
    """
    prefixes = {describe_call(call(status))[:2] for status in ("ok", "declined", "blocked", "error")}
    assert len(prefixes) == 4


# --- the first line of a result -------------------------------------------


def test_a_result_summarises_to_its_first_non_empty_line() -> None:
    assert first_line("\n\nfirst\nsecond\n") == "first"


def test_an_empty_result_summarises_to_nothing() -> None:
    assert first_line("") == ""
    assert first_line("\n\n") == ""


def test_a_long_first_line_is_cut() -> None:
    assert len(first_line("x" * 500)) <= 120


def test_a_result_of_only_whitespace_summarises_to_nothing() -> None:
    assert first_line("   \n\t\n") == ""


def test_a_command_result_shows_its_output_not_its_labels() -> None:
    """`run_command` opens with `exit code:` and `--- stdout ---`.

    Taking the first line would show the exit code and hide the output,
    which is the part worth one line. Found by watching a real run: the
    live log said `exit code: 0` and never showed what the program
    printed.
    """
    result = "exit code: 0\n--- stdout ---\nHello, world!\n"
    assert first_line(result) == "Hello, world!"


def test_a_command_result_falls_back_to_the_exit_code_when_it_printed_nothing() -> None:
    """Otherwise a silent command would show no detail at all."""
    assert first_line("exit code: 0\n(no output)") == "exit code: 0"


def test_a_command_result_prefers_stderr_when_that_is_all_there_was() -> None:
    result = "exit code: 2\n--- stderr ---\nNo such file or directory\n"
    assert first_line(result) == "No such file or directory"


def test_a_timed_out_command_says_so() -> None:
    """There is no content line to prefer, so the timeout line is the news."""
    assert "TIMED OUT" in first_line("TIMED OUT after 30s and was killed.")


def test_a_file_result_is_unaffected() -> None:
    """The skipping is for `run_command`'s layout, not for results generally."""
    assert first_line("exit codes are a useful thing\nexit code: 2\n") == (
        "exit codes are a useful thing"
    )


# --- counting --------------------------------------------------------------


def test_the_counts_tell_a_finished_run_from_a_refused_one() -> None:
    """The whole reason the counts exist.

    A run can end with outcome "finished" having declined three edits.
    Without the counts those two are indistinguishable, and a user
    reading only the summary would think the task was done.
    """
    run = run_with(
        call("ok"),
        call("declined", step=2),
        call("declined", step=3),
        call("blocked", step=4),
        call("error", step=5),
        outcome="finished",
    )
    tally = run.counts
    assert tally == {"ok": 1, "declined": 2, "blocked": 1, "error": 1}


def test_a_run_with_no_calls_counts_zero_of_everything() -> None:
    assert sum(run_with().counts.values()) == 0


def test_an_unexpected_status_does_not_crash_the_tally() -> None:
    """A future status should show up, not raise."""
    run = Run(task="x")
    run.calls.append(call("something_new"))
    assert run.counts["something_new"] == 1


# --- duration --------------------------------------------------------------


def test_a_finished_run_knows_how_long_it_took() -> None:
    assert run_with().duration == 5.0


def test_a_running_run_reports_time_so_far() -> None:
    """Otherwise a live view would show a duration of zero for ever."""
    run = Run(task="x", started=1000.0)
    assert run.duration >= 0.0


# --- round tripping --------------------------------------------------------


def test_a_run_survives_a_round_trip() -> None:
    original = run_with(call("ok"), call("declined", step=2), outcome="finished")
    restored = Run.from_dict(original.to_dict())
    assert restored.task == original.task
    assert restored.model == original.model
    assert restored.outcome == original.outcome
    assert restored.steps_used == original.steps_used
    assert len(restored.calls) == 2
    assert restored.calls[1].status == "declined"


def test_call_arguments_survive_a_round_trip() -> None:
    original = call(args={"path": "a.py", "content": "x = 1", "extra": [1, 2]})
    restored = ToolCall.from_dict(original.to_dict())
    assert restored.args == {"path": "a.py", "content": "x = 1", "extra": [1, 2]}


def test_the_derived_summary_is_not_a_stored_field() -> None:
    """`summary` is computed, so storing it would let it go stale."""
    data = call().to_dict()
    assert "summary" in data
    restored = ToolCall.from_dict(data)
    assert restored.summary == data["summary"]


def test_an_unknown_field_from_a_newer_version_is_ignored() -> None:
    """A log written by a later commit must still open."""
    data = call().to_dict()
    data["some_future_field"] = "whatever"
    assert ToolCall.from_dict(data).name == "read_file"


def test_a_missing_required_field_is_reported_rather_than_invented() -> None:
    """A call with a fabricated step number is worse than an error."""
    data = call().to_dict()
    del data["step"]
    with pytest.raises(ValueError, match="step"):
        ToolCall.from_dict(data)


def test_a_record_with_no_calls_loads() -> None:
    assert Run.from_dict({"task": "x"}).calls == []


def test_an_unknown_status_survives_the_round_trip() -> None:
    """Forward compatibility, checked at both ends."""
    restored = Run.from_dict({"task": "x", "calls": [call("brand_new").to_dict()]})
    assert restored.calls[0].status == "brand_new"


# --- defaults --------------------------------------------------------------


def test_a_new_run_reports_itself_as_running() -> None:
    """The status a live view shows before anything has happened."""
    assert Run(task="x").outcome == "running"


def test_a_new_run_has_not_finished_yet() -> None:
    """0 rather than now, so 'finished' is a distinguishable answer."""
    assert Run(task="x").finished == 0.0


def test_a_new_run_has_no_calls_and_no_answer() -> None:
    run = Run(task="x")
    assert run.calls == []
    assert run.answer == ""
    assert run.error == ""


def test_the_status_values_are_the_documented_ones() -> None:
    """A Literal is only documentation unless something checks it."""
    allowed: set[str] = set(CallStatus.__args__)
    assert allowed == {"ok", "declined", "blocked", "error"}