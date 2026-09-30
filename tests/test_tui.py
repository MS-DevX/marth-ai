"""Tests for the live dashboard.

Curses cannot draw without a terminal, so the drawing code is driven
against a fake screen that records what it was asked to write. That is
enough to check the things that actually go wrong: a line off the edge
of the screen, a header that overwrites the log, a prompt that leaves
the terminal in curses mode.

What is deliberately not tested here is whether curses itself works;
that needs a terminal, and the fallback path is what matters when there
is not one.
"""

import curses

import pytest

from agent import config, safety, tui
from agent.runs import Run, ToolCall


class FakeScreen:
    """A stand-in for a curses window that records what was drawn.

    Args:
        rows: The screen height.
        columns: The screen width.
    """

    def __init__(self, rows: int = 24, columns: int = 80) -> None:
        self._size = (rows, columns)
        self.lines: dict[int, str] = {}
        self.refreshes = 0
        self.cleared = 0

    def getmaxyx(self) -> tuple[int, int]:
        """Return the screen size as (rows, columns)."""
        return self._size

    def erase(self) -> None:
        """Blank the recorded screen."""
        self.cleared += 1
        self.lines.clear()

    def clear(self) -> None:
        """Blank the recorded screen, as `erase` does."""
        self.erase()

    def addnstr(self, row: int, column: int, text: str, limit: int, attr: int) -> None:
        """Record a write, clipped to the limit curses would apply.

        Raises:
            curses.error: when the write starts off the screen, which is
                what curses itself does rather than clipping.
        """
        if not 0 <= row < self._size[0] or not 0 <= column < self._size[1]:
            raise curses.error("addnstr() returned ERR")
        self.lines[row] = self.lines.get(row, "")[:column] + text[:limit]

    def refresh(self) -> None:
        """Count a repaint."""
        self.refreshes += 1

    def text(self) -> str:
        """Return everything drawn, as one blob."""
        return "\n".join(self.lines[row] for row in sorted(self.lines))


@pytest.fixture
def dashboard():
    """Return a dashboard wired to a fake screen."""
    board = tui.Dashboard(Run(task="do the thing", model="fake-model"))
    board.attach(FakeScreen())
    return board


def make_run(*calls: ToolCall) -> Run:
    """Return a run in progress, with the given calls already made."""
    return Run(
        task="do the thing",
        model="fake-model",
        provider="test",
        started=1000.0,
        steps_used=len(calls),
        calls=list(calls),
    )


def read(name: str, status: str = "ok", step: int = 1, **args) -> ToolCall:
    """Return a read-only tool call."""
    return ToolCall(step=step, name=name, args=args, status=status, output="a.py:1")


# --- what is on screen -----------------------------------------------------


def test_the_task_and_model_are_visible(dashboard) -> None:
    """Otherwise a long run has no context on screen."""
    dashboard(make_run())
    text = dashboard._screen.text()
    assert "do the thing" in text
    assert "fake-model" in text


def test_the_step_counter_is_visible(dashboard) -> None:
    """The model needs to know how much budget is left, and so does the user."""
    run = make_run(read("list_files", path="."))
    run.steps_used = 4
    dashboard(run)
    text = dashboard._screen.text()
    assert f"4/{config.MAX_STEPS}" in text


def test_a_completed_tool_call_appears_in_the_log(dashboard) -> None:
    dashboard(make_run(read("list_files", path=".")))
    assert "list_files" in dashboard._screen.text()


def test_a_refusal_is_marked_in_the_log(dashboard) -> None:
    """The point of the live view: see a decline as it happens."""
    call = ToolCall(
        step=2,
        name="write_file",
        args={"path": "a.py", "content": "x"},
        status="declined",
        output="Declined by the user.",
    )
    dashboard(make_run(call))
    assert "Declined" in dashboard._screen.text()


def test_a_blocked_command_is_marked(dashboard) -> None:
    call = ToolCall(
        step=2,
        name="run_command",
        args={"command": "rm -rf /"},
        status="blocked",
        output="Refused: this command is blocked.",
    )
    dashboard(make_run(call))
    assert "Refused" in dashboard._screen.text()


def test_a_status_this_version_does_not_know_is_not_drawn_as_a_success() -> None:
    """An unknown outcome is not a green one.

    A log written by a later version is exactly what somebody opens the
    history to read. Defaulting an unrecognised status to the success
    colour would report it as though the call worked.
    """
    assert tui._shade("quarantined") == tui.C_WARN
    assert tui._shade("quarantined") != tui.C_OK


def test_counts_include_refusals(dashboard) -> None:
    """A run that looks busy but was refused twice is worth seeing."""
    run = make_run(
        ToolCall(step=1, name="write_file", args={"path": "a"}, status="declined"),
        ToolCall(step=2, name="write_file", args={"path": "b"}, status="declined"),
        ToolCall(step=3, name="read_file", args={"path": "c"}, status="ok"),
    )
    dashboard(run)
    assert "2 declined" in dashboard._screen.text()


# --- not drawing off the screen -------------------------------------------


def test_nothing_is_drawn_past_the_last_row() -> None:
    """A short terminal must not raise on the last line.

    Curses returns an error rather than clipping, and one line too many
    would take the whole run down with it.
    """
    board = tui.Dashboard(make_run())
    board.attach(FakeScreen(rows=4, columns=30))
    board(make_run())  # must not raise


def test_a_very_narrow_terminal_does_not_raise() -> None:
    board = tui.Dashboard(make_run(read("list_files", path=".")))
    board.attach(FakeScreen(rows=10, columns=8))
    board(make_run(read("list_files", path=".")))  # must not raise


def test_a_long_task_line_does_not_overflow_the_screen() -> None:
    """A long task must be cut, not wrapped onto the log."""
    run = make_run()
    run.task = "x" * 300
    board = tui.Dashboard(run)
    board.attach(FakeScreen(rows=10, columns=40))
    board(run)
    for row, line in board._screen.lines.items():
        assert len(line) <= 40, f"row {row} overflowed: {len(line)}"


def test_a_very_long_argument_list_is_cut() -> None:
    call = ToolCall(
        step=1,
        name="write_file",
        args={"content": "y" * 5000, "path": "a.py"},
        status="ok",
    )
    run = make_run(call)
    board = tui.Dashboard(run)
    board.attach(FakeScreen(rows=10, columns=40))
    board(run)
    for line in board._screen.lines.values():
        assert len(line) <= 40


# --- updating as the run goes ---------------------------------------------


def test_the_log_scrolls_rather_than_growing_without_limit() -> None:
    """A run of 20 steps must not need a 20-line window."""
    run = make_run()
    board = tui.Dashboard(run)
    board.attach(FakeScreen(rows=10, columns=60))
    for n in range(40):
        run.calls.append(read("list_files", path=f"{n}.py", step=n + 1))
        run.steps_used = n + 1
        board(run)
    assert len(board._log) == 80  # two lines per call
    assert "older lines" in board._screen.text()


def test_the_newest_call_is_always_on_screen() -> None:
    """The log scrolls; the current step is what the user is watching."""
    run = make_run()
    board = tui.Dashboard(run)
    board.attach(FakeScreen(rows=10, columns=60))
    for n in range(40):
        run.calls.append(read("list_files", path=f"last{n}.py", step=n + 1))
        run.steps_used = n + 1
        board(run)
    assert "last39.py" in board._screen.text()


def test_a_call_is_not_drawn_twice(dashboard) -> None:
    """The observer fires per event; the same call must not be reprinted."""
    run = make_run(read("list_files", path="a.py"))
    dashboard(run)
    dashboard(run)
    dashboard(run)
    assert [text for text, _ in dashboard._log] == [
        "step  1  list_files(path=a.py)",
        "           a.py:1",
    ]


# --- falling back ----------------------------------------------------------


def test_a_drawing_failure_does_not_stop_the_run(monkeypatch) -> None:
    """A terminal hiccup must not cost the user the work."""
    run = make_run()
    board = tui.Dashboard(run)
    board.attach(FakeScreen())
    monkeypatch.setattr(
        tui.curses, "init_pair", lambda *a: (_ for _ in ()).throw(curses.error("no"))
    )
    board(run)  # must not raise


def test_available_is_false_without_a_terminal(monkeypatch) -> None:
    """A pipe cannot be drawn on, and must fall back to plain output."""
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: False, raising=False)
    assert tui.available() is False


def test_available_is_false_when_nothing_needs_approving(monkeypatch) -> None:
    """With --yes there is no prompt to make room for."""
    monkeypatch.setattr(config, "AUTO_APPROVE", True)
    assert tui.available() is False


def test_available_is_false_on_a_terminal_too_small(monkeypatch) -> None:
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(tui.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(tui.curses, "setupterm", lambda: None)
    monkeypatch.setattr(tui, "terminal_size", lambda: (4, 20))
    assert tui.available() is False


def test_available_is_true_on_a_usable_terminal(monkeypatch) -> None:
    monkeypatch.setattr(tui.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(tui.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(tui.curses, "setupterm", lambda: None)
    monkeypatch.setattr(tui, "terminal_size", lambda: (40, 120))
    assert tui.available() is True


# --- the confirmation prompt ----------------------------------------------


def test_the_dashboard_can_take_over_the_prompt(monkeypatch) -> None:
    """Curses owns the screen, so the question needs its own handler."""
    board = tui.Dashboard(make_run())
    board.attach(FakeScreen())
    monkeypatch.setattr(tui.curses, "endwin", lambda: None)
    monkeypatch.setattr(tui, "input", lambda _p: "y", raising=False)
    safety.set_ask_handler(board.ask)
    try:
        assert safety._ask("Apply this edit?") is True
    finally:
        safety.set_ask_handler(None)


def test_the_handler_is_removed_afterwards(monkeypatch) -> None:
    """A leftover handler would break the next plain run in the process."""
    board = tui.Dashboard(make_run())
    board.attach(FakeScreen())
    monkeypatch.setattr(tui.curses, "endwin", lambda: None)
    monkeypatch.setattr(tui, "input", lambda _p: "n", raising=False)
    safety.set_ask_handler(board.ask)
    safety.set_ask_handler(None)
    assert safety._ASK_HANDLER is None


def test_anything_but_yes_declines(monkeypatch) -> None:
    """A stray keystroke must not approve a change to the user's files."""
    board = tui.Dashboard(make_run())
    board.attach(FakeScreen())
    monkeypatch.setattr(tui.curses, "endwin", lambda: None)
    for answer in ("", "n", "no", "Y E S", "sure", "why not"):
        monkeypatch.setattr(tui, "input", lambda _p, a=answer: a, raising=False)
        safety.set_ask_handler(board.ask)
        try:
            assert safety._ask("Apply?") is False, f"{answer!r} was treated as yes"
        finally:
            safety.set_ask_handler(None)


def test_an_interrupted_prompt_declines(monkeypatch) -> None:
    """Half a question is not consent."""
    board = tui.Dashboard(make_run())
    board.attach(FakeScreen())
    monkeypatch.setattr(tui.curses, "endwin", lambda: None)

    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(tui, "input", interrupt, raising=False)
    safety.set_ask_handler(board.ask)
    try:
        assert safety._ask("Apply?") is False
    finally:
        safety.set_ask_handler(None)


def test_the_screen_is_taken_back_after_a_prompt(monkeypatch) -> None:
    """Otherwise the log has a hole where the question was."""
    board = tui.Dashboard(make_run())
    board.attach(FakeScreen())
    monkeypatch.setattr(tui.curses, "endwin", lambda: None)
    monkeypatch.setattr(tui, "input", lambda _p: "y", raising=False)
    board.ask("Apply?")
    assert board._screen.refreshes > 0, "nothing was redrawn after the prompt"


# --- helpers ---------------------------------------------------------------


def test_a_multi_line_result_is_summarised_to_one_line(dashboard) -> None:
    """The log has room for one line; the rest is in the history file."""
    run = make_run(
        ToolCall(
            step=1,
            name="read_file",
            args={"path": "a.py"},
            output="first line\nsecond line\nthird line",
        )
    )
    dashboard(run)
    text = dashboard._screen.text()
    assert "first line" in text
    assert "second line" not in text


def test_a_call_with_no_result_shows_only_the_call(dashboard) -> None:
    """A blank detail line would be noise on a small screen."""
    run = make_run(ToolCall(step=1, name="read_file", args={"path": "a.py"}))
    dashboard(run)
    assert [text for text, _ in dashboard._log] == ["step  1  read_file(path=a.py)"]


def test_the_colour_travels_with_the_line_not_with_the_text() -> None:
    """The marker sits after the step number, so sniffing text is wrong.

    Parsing the rendered string to work out the status was the previous
    approach and it put the refusal colour on a success as soon as the
    log format moved. Carrying the colour alongside is what makes that
    class of bug impossible.
    """
    assert tui._shade("ok") == tui.C_OK
    assert tui._shade("declined") == tui.C_WARN
    assert tui._shade("error") == tui.C_WARN
    assert tui._shade("blocked") == tui.C_BAD


def test_a_blocked_call_is_coloured_differently_from_a_declined_one() -> None:
    """Different reasons, so a different reading of the log."""
    assert tui._shade("blocked") != tui._shade("declined")
