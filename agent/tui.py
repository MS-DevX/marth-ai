"""A live view of a run, drawn with curses from the standard library.

Full-screen on purpose: a long run scrolls past in a plain terminal and
there is no way to see what has already happened without scrolling back
through it. Here the log scrolls and the header stays put, so the
current step and the running totals are always on screen.

The hard part is the confirmation prompt. A curses program owns the
terminal, so writing a question into that screen produces unreadable
output. `Dashboard.ask` handles it the way curses programs normally do:
give the terminal back, ask on a clean screen, then take the terminal
back and redraw. The user sees an ordinary prompt; the dashboard sees a
suspension.

Nothing here imports a third-party TUI library. `curses` is in the
standard library, and the alternative is a dependency for what amounts
to a few lines of escape codes.

Falls back rather than breaking: if the terminal cannot do curses,
`available()` returns False and the caller prints plainly instead. A
dashboard is not worth a crash.
"""

import curses
import os
import sys
from collections.abc import Callable
from typing import Any

from . import config, safety
from .runs import Run, describe_call, first_line

# Below this the header and log stop fitting, so plain output is used
# instead. A split terminal pane on a laptop is a real size.
MIN_WIDTH = 40
MIN_HEIGHT = 8

# Colour pair numbers. Applied with `curses.color_pair`, which needs an
# init_pair call per pair; `_init_colours` does that once per session.
C_HEADER = 1
C_OK = 2
C_WARN = 3
C_BAD = 4
C_DIM = 5
C_ACCENT = 6

_PAIR_DEFS = {
    C_HEADER: (curses.COLOR_CYAN, -1),
    C_OK: (curses.COLOR_GREEN, -1),
    C_WARN: (curses.COLOR_YELLOW, -1),
    C_BAD: (curses.COLOR_RED, -1),
    C_DIM: (curses.COLOR_WHITE, -1),
    C_ACCENT: (curses.COLOR_MAGENTA, -1),
}


def _shade(status: str) -> int:
    """Return the colour for a call that ended this way.

    Args:
        status: The call's status, as recorded by the loop.

    An unrecognised status gets the warning colour rather than the
    success one. It comes from a log written by a later version of this
    program, and colouring it green would report an outcome nobody has
    confirmed as a success. `ok` is the only status that earns the
    success colour.
    """
    if status == "ok":
        return C_OK
    if status == "blocked":
        return C_BAD
    return C_WARN  # declined, errored, or not one this version knows


def available() -> bool:
    """Return whether this terminal can show the dashboard.

    Checked rather than assumed: a piped stdout, a CI log, and a window
    too small to be useful all have to fall back to plain output rather
    than draw over it.

    With `--yes` there is nothing to approve, so there is no reason to
    take over the screen either.
    """
    if config.AUTO_APPROVE:
        return False
    if not (sys.stdout.isatty() and sys.stdin.isatty()):
        return False
    try:
        curses.setupterm()
    except curses.error:
        return False
    rows, columns = terminal_size()
    return columns >= MIN_WIDTH and rows >= MIN_HEIGHT


def terminal_size() -> tuple[int, int]:
    """Return the terminal size as (rows, columns).

    Raises:
        OSError: if the size cannot be determined, which is the normal
            answer when stdout is a pipe.
    """
    columns, rows = os.get_terminal_size(sys.stdout.fileno())
    return rows, columns


class Dashboard:
    """Draws a run as it happens.

    Args:
        record: The run the loop is filling in. Held by reference, so
            every redraw shows the current state rather than a copy
            taken at construction.

    Construct through `run_with_dashboard` rather than directly: the
    terminal has to be handed back afterwards whatever happens, and
    that is the only function that can guarantee it.
    """

    def __init__(self, record: Run) -> None:
        self._run = record
        self._screen: Any = None
        self._log: list[str] = []
        self._shown = 0  # calls already turned into log lines
        self._colours_ready = False

    def attach(self, screen: Any) -> None:
        """Take the curses screen to draw on.

        Args:
            screen: The window handed over by `curses.wrapper`.
        """
        self._screen = screen
        self._colours_ready = False

    # -- the redraw --------------------------------------------------------

    def __call__(self, record: Run) -> None:
        """Redraw for a progress event.

        Args:
            record: The run. The same object every time, still being
                filled in; taken as an argument so the loop's observer
                signature works directly.

        Drawing errors are swallowed. A run that cannot be drawn is
        still a run that should happen, and a terminal hiccup must not
        cost the user their work.
        """
        self._run = record
        try:
            self._add_new_calls()
            self._draw()
        except curses.error:
            pass

    def _add_new_calls(self) -> None:
        """Turn tool calls we have not shown yet into log lines."""
        for call in self._run.calls[self._shown :]:
            self._log.append(
                (f"step {call.step:>2}  {describe_call(call)}", _shade(call.status))
            )
            detail = first_line(call.output)
            if detail:
                self._log.append((f"           {detail}", C_DIM))
            self._shown += 1

    def _draw(self) -> None:
        """Repaint the whole screen.

        There is no partial-update story worth relying on here and the
        screen is small, so it clears and repaints every frame. The
        alternative is tracking what changed, which is where this kind
        of code usually goes wrong.
        """
        self._init_colours()
        self._screen.erase()
        height, width = self._screen.getmaxyx()
        used = self._draw_header(width)
        self._draw_log(height, width, used)
        self._draw_footer(height, width)
        self._screen.refresh()

    def _init_colours(self) -> None:
        """Define the colour pairs, once per session."""
        if self._colours_ready:
            return
        for number, (foreground, background) in _PAIR_DEFS.items():
            try:
                curses.init_pair(number, foreground, background)
            except (curses.error, ValueError):
                # A terminal with too few pairs, or none at all, still
                # draws; `_attribute` falls back to plain text.
                pass
        self._colours_ready = True

    def _attribute(self, colour: int) -> int:
        """Return the curses attribute for one of the `C_` pair numbers.

        Falls back to `A_NORMAL` when colour is unavailable. Colour is
        decoration, and a terminal that cannot do it still has to be able
        to run the agent; an exception here would take the whole run down
        over presentation.
        """
        try:
            return curses.color_pair(colour)
        except (curses.error, ValueError):
            return curses.A_NORMAL

    def _draw_header(self, width: int) -> int:
        """Draw the model, the task and the running totals.

        Args:
            width: The screen width in columns.

        Returns:
            The number of rows used, so the log starts below them.
        """
        self._put(0, 0, f" marth-ai  {self._run.model or 'no model'} ", C_HEADER)
        status = f"{self._run.outcome}  step {self._run.steps_used}/{config.MAX_STEPS}  "
        self._put(
            0,
            max(0, width - len(status) - 1),
            status,
            C_ACCENT if self._run.outcome != "finished" else C_OK,
        )
        self._put(1, 0, f" {self._run.task}", C_DIM)
        tally = self._run.counts
        self._put(
            2,
            0,
            f" {len(self._run.calls)} calls"
            f"   {tally['ok']} ok"
            f"   {tally['declined']} declined"
            f"   {tally['blocked']} blocked"
            f"   {tally['error']} errors",
            C_DIM,
        )
        return 3

    def _draw_log(self, height: int, width: int, top: int) -> None:
        """Draw the log, newest at the bottom of the window.

        Args:
            height: Total screen height.
            width: Screen width.
            top: First row available, below the header.
        """
        room = height - top
        if room <= 0:
            return
        for offset, (text, colour) in enumerate(self._log[-room:]):
            self._put(top + offset, 0, text, colour)
        hidden = len(self._log) - room
        if hidden > 0:
            self._put(0, max(0, width - 30), f" {hidden} older lines ", C_DIM)

    def _draw_footer(self, height: int, width: int) -> None:
        """Draw the key hints along the bottom row."""
        if height < 2:
            return
        hint = " y approve   n decline   ^C stop "
        self._put(height - 1, max(0, width - len(hint) - 1), hint, C_DIM)

    def _put(self, row: int, column: int, text: str, colour: int) -> None:
        """Write one line, tolerating the screen edges.

        Args:
            row: Row to write on.
            column: Column to start at.
            text: The text to write.
            colour: One of the `C_` pair numbers.

        Curses raises rather than clipping a write that runs past the
        last cell, and a log line is not worth losing because it was
        two characters too wide for the current window.
        """
        if self._screen is None:
            return
        height, width = self._screen.getmaxyx()
        if not 0 <= row < height or column >= width:
            return
        try:
            self._screen.addnstr(
                row, column, text, width - column - 1, self._attribute(colour)
            )
        except curses.error:
            pass

    # -- suspension for prompts -------------------------------------------

    def ask(self, question: str) -> bool:
        """Ask a y/n question with the dashboard suspended.

        Args:
            question: The question to put to the user.

        Returns:
            True for y/yes, False for anything else. An interrupted
            prompt declines: a half-typed answer is not consent, and
            declining is the direction that loses nothing.
        """
        # Hand the terminal back first. Writing the question into the
        # curses screen produces unreadable output, and the keystrokes
        # would fight the dashboard's own.
        curses.endwin()
        try:
            try:
                answer = input(f"\n{question}\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("", file=sys.stderr)
                return False
        finally:
            # Take it back and repaint, so the log has no hole where the
            # prompt was.
            self._resume()
        return answer in {"y", "yes"}

    def _resume(self) -> None:
        """Restart curses after a suspended prompt and redraw.

        `doupdate` rather than `refresh`, because the screen was left in
        whatever state the normal prompt made of it. The redraw happens
        even if that fails: a screen the dashboard cannot restore is
        still worth a fresh paint, and skipping it would leave a hole
        where the question was.
        """
        try:
            curses.doupdate()
        except (curses.error, ValueError):
            pass
        try:
            self._draw()
        except curses.error:
            pass


def run_with_dashboard(
    record: Run, work: Callable[[Callable[[Run], None]], Run]
) -> Run:
    """Run `work` with the dashboard live, then return the finished run.

    Args:
        record: The run to draw. `work` fills it in and returns it.
        work: Takes an observer callback and does the work. Normally
            `partial(loop.run_record, task)`, whose `on_event` argument
            is the observer.

    Returns:
        The finished run, the same object that was passed in.

    The terminal is always handed back. That is the one thing a
    full-screen program must never get wrong: leaving the terminal in
    curses mode makes the shell unusable until the user knows to type
    `reset`, and `curses.wrapper` is what guarantees it happens even if
    drawing or the run itself raises.
    """
    dashboard = Dashboard(record)

    def entry(screen: Any) -> None:
        """Curses entry point: draw, run the work, stop drawing."""
        dashboard.attach(screen)
        try:
            curses.curs_set(0)
        except curses.error:
            pass  # some terminals have no cursor to hide
        safety.set_ask_handler(dashboard.ask)
        try:
            work(dashboard)
        finally:
            safety.set_ask_handler(None)

    try:
        curses.wrapper(entry)
    except curses.error as exc:
        # No usable terminal. Say why rather than failing silently, then
        # do the work anyway; only the live view is lost.
        print(f"[live view unavailable: {exc}]", file=sys.stderr)
        return work(lambda _run: None)
    return record
