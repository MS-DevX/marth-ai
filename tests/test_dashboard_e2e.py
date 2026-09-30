"""End-to-end tests of the live dashboard, inside a real terminal.

Curses needs a terminal, so these fork a pty and run the agent for real
against a scripted model. That is the only way to check the parts that
matter and cannot be unit tested: that the dashboard starts, that the
confirmation prompt can still be answered, and above all that the
terminal is handed back in a usable state.

The last of those is the one that must never fail. A curses program
that exits without restoring the terminal leaves the user's shell
unable to echo, and they have to know to type `reset`. Everything else
here checks that the feature works; that one checks that it fails
safely.
"""

import os
import pty
import select
import signal
import sys
import time
from pathlib import Path

import pytest

from agent import main, safety
from agent.llm import LLMResponse, ToolCall

# Generous, because a scripted run is fast but a pty on a loaded machine
# is not. A test that waits for a timeout rather than a result is a test
# that stops being run.
TIMEOUT = 60.0

# What a terminal emulator uses to enter and leave the alternate screen.
# Seeing the second one means curses tidied up after itself.
LEAVE_ALT_SCREEN = "\x1b[?1049l"
ENTER_ALT_SCREEN = "\x1b[?1049h"

# The size the fake terminal reports. `COLUMNS` and `LINES` in the
# environment are not enough: curses asks the kernel for the window
# size, and a pty starts at 0x0. COLUMNS and LINES are only a fallback
# for `shutil.get_terminal_size`, which the agent deliberately does not
# use, because a fallback that disagrees with the real size produces
# drawing code that is tested against a screen nobody will ever see.
LINES = 30
COLUMNS = 100


def _set_window_size(fd: int, rows: int, columns: int) -> None:
    """Tell the pty how big it is, the way a real terminal would.

    Args:
        fd: The pty's master file descriptor.
        rows: Terminal height.
        columns: Terminal width.
    """
    import fcntl
    import struct
    import termios

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))


class ScriptedModel:
    """A model that replays fixed responses, for driving the CLI.

    Args:
        responses: What to return on each successive call to `send`.
    """

    model_name = "scripted-model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    def send(self, history, tool_schemas=None):  # type: ignore[no-untyped-def]
        """Return the next scripted reply.

        Raises:
            AssertionError: if the agent asked for more turns than were
                scripted, which is a bug in the test rather than the run.
        """
        if not self._responses:
            raise AssertionError("the agent asked for more turns than scripted")
        return self._responses.pop(0)

    def list_model_names(self) -> list[str]:  # type: ignore[no-untyped-def]
        """Unused here; present to satisfy the provider interface."""
        return ["scripted-model"]


def text_reply(text: str) -> LLMResponse:
    """Return a reply with prose and no tool calls."""
    return LLMResponse(text=text)


def tool_call(name: str, **args) -> LLMResponse:
    """Return a reply asking for one tool call."""
    return LLMResponse(text="", tool_calls=(ToolCall(id="1", name=name, args=args),))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Return a throwaway sandbox for the agent to work in.

    A real directory with a file in it, because the sandbox refuses to
    hand out paths outside the workspace root and a test that only
    worked because the repository happened to contain the right files
    is testing the repository, not the agent.
    """
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "README.md").write_text("# fixture\n")
    return root


def drive(
    responses: list[LLMResponse],
    *extra_args: str,
    answers: list[str] | None = None,
    approve: bool = False,
    max_steps: int = 6,
    workspace: Path | None = None,
    interrupt_after: float | None = None,
) -> str:
    """Run the agent in a pty and return everything it printed.

    Args:
        responses: The scripted model replies.
        extra_args: Extra command line arguments.
        answers: Answers for confirmation prompts, in order. Any prompt
            beyond these is declined, which is the safe default.
        approve: Answer every prompt with y, whatever the order.
        max_steps: The loop's step limit. Patched on `config` rather
            than passed as `AGENT_MAX_STEPS`, because that value is read
            when `config` is first imported - in this process, long
            before an environment variable set here would be seen.
        workspace: Where the agent is allowed to work. Patched the same
            way `max_steps` is, and for the same reason. These tests
            really do run `write_file`, and the default workspace is
            this repository, so leaving it alone means every test run
            litters the working tree with whatever the model was told to
            write.
        interrupt_after: Send SIGINT after this many seconds, to test
            that a cancelled run still restores the terminal.

    Returns:
        The full output, escape sequences included.
    """
    model = ScriptedModel(responses)
    pending = list(answers or [])

    def answering_ask(_question: str) -> bool:
        """Answer a confirmation from the script rather than a keypress."""
        answer = "y" if approve else (pending.pop(0) if pending else "n")
        return answer.strip().lower() in {"y", "yes"}

    return _run_in_pty(
        ["--model", "scripted-model", "do the thing", *extra_args],
        model=model,
        ask=answering_ask,
        max_steps=max_steps,
        workspace=workspace,
        interrupt_after=interrupt_after,
    )


def _run_in_pty(
    argv: list[str],
    model: ScriptedModel,
    ask,
    max_steps: int = 6,
    workspace: Path | None = None,
    interrupt_after: float | None = None,
) -> str:
    """Run the CLI in a pty with a swapped-in model and prompt handler.

    Args:
        argv: Arguments for `python -m agent.main`.
        model: The model to give the agent in place of a real one.
        ask: What `_ask` should return.
        max_steps: The loop's step limit.
        workspace: The sandbox root for the run.
        interrupt_after: Send SIGINT after this many seconds.

    The swaps are undone in a `finally`: this runs inside the pytest
    process, and a leaked one would make the next test use a scripted
    model without knowing it.
    """
    import agent.loop as loop_module
    from agent import config

    real_run_record = loop_module.run_record
    real_build_llm = main.build_llm
    real_ask = safety._ask
    real_max_steps = config.MAX_STEPS
    real_workspace = config.WORKSPACE_ROOT

    def fake_run_record(task, llm, on_event=None, record=None):  # type: ignore[no-untyped-def]
        """Run the real loop against the scripted model."""
        return real_run_record(task, model, on_event, record)

    loop_module.run_record = fake_run_record
    main.build_llm = lambda _name: model
    safety._ask = ask
    safety.set_ask_handler(None)
    config.MAX_STEPS = max_steps
    if workspace is not None:
        config.WORKSPACE_ROOT = workspace
    try:
        return _capture(argv, interrupt_after)
    finally:
        loop_module.run_record = real_run_record
        main.build_llm = real_build_llm
        safety._ask = real_ask
        safety.set_ask_handler(None)
        config.MAX_STEPS = real_max_steps
        config.WORKSPACE_ROOT = real_workspace


def _capture(argv: list[str], interrupt_after: float | None = None) -> str:
    """Run the CLI in a pty and return what it printed.

    Args:
        argv: Arguments for `python -m agent.main`.
        interrupt_after: Send SIGINT after this many seconds.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Set in the parent so the forked child inherits it. `os.environ`
    # is copied by fork, which is why this is not passed to the child
    # explicitly the way an execve would need.
    os.environ.update(
        TERM="xterm-256color",
        AGENT_NO_HISTORY="1",  # the log would go in the sandbox, but
    )
    pid, fd = pty.fork()
    if pid == 0:  # child
        # Deliberately not execve. `pty.fork` already gives us a new
        # process with a controlling terminal, which is all curses needs,
        # and continuing in this copy is what keeps the model and prompt
        # swaps above. An execve would start a fresh interpreter and
        # throw them away, and the child would go looking for a real API
        # key.
        os.chdir(root)
        # `pty.fork` put the pty slave on fds 0-2, but `sys.stdout` is
        # still pytest's own object, pointing at its capture file. Rebind
        # it or the run writes there and curses sees no terminal.
        sys.stdout = os.fdopen(1, "w", buffering=1, errors="replace")
        sys.stderr = os.fdopen(2, "w", buffering=1, errors="replace")
        sys.stdin = os.fdopen(0, "r", buffering=1)
        try:
            main.main(argv)
        finally:
            os._exit(0)

    _set_window_size(fd, rows=LINES, columns=COLUMNS)

    output = bytearray()
    started = time.time()
    interrupted = False
    deadline = started + TIMEOUT
    while time.time() < deadline:
        if interrupt_after is not None and not interrupted:
            if time.time() - started > interrupt_after:
                os.kill(pid, signal.SIGINT)
                interrupted = True
        ready, _, _ = select.select([fd], [], [], 0.2)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        output += chunk
        if _looks_finished(output):
            # Drain the rest, so the summary is not cut off mid-line.
            time.sleep(0.3)
            output += _drain(fd)
            break
    _reap(pid, fd)
    return output.decode("utf-8", "replace")


def _drain(fd: int) -> bytes:
    """Read whatever is left on the pty, briefly."""
    tail = bytearray()
    deadline = time.time() + 2.0
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if not ready:
            break
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        tail += chunk
    return bytes(tail)


def _looks_finished(output: bytearray) -> bool:
    """Return whether the agent has printed its summary."""
    return b"steps," in output or b"Configuration error" in output


def _reap(pid: int, fd: int) -> None:
    """Close the pty and collect the child, without hanging on either."""
    try:
        os.close(fd)
    except OSError:
        pass
    for _ in range(20):
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if done:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except (OSError, ChildProcessError):
        pass


# --- the dashboard starts at all ------------------------------------------


def test_the_dashboard_takes_over_the_screen(workspace: Path) -> None:
    """The whole feature: escape sequences and an alternate screen."""
    output = drive([text_reply("all done")], workspace=workspace)
    assert ENTER_ALT_SCREEN in output, "curses never started"


def test_the_task_and_model_reach_the_screen(workspace: Path) -> None:
    output = drive([text_reply("all done")], workspace=workspace)
    assert "do the thing" in output
    assert "scripted-model" in output


def test_tool_calls_are_shown_while_the_run_works(workspace: Path) -> None:
    responses = [
        tool_call("list_files", path="."),
        tool_call("read_file", path="README.md"),
        text_reply("read them"),
    ]
    output = drive(responses, workspace=workspace)
    assert "list_files" in output
    assert "read_file" in output


def test_the_tool_result_is_shown_too(workspace: Path) -> None:
    """The log has to say what came back, not just what was asked."""
    responses = [tool_call("read_file", path="README.md"), text_reply("read it")]
    output = drive(responses, "--plain", workspace=workspace)
    assert "# fixture" in output


def test_the_final_answer_is_printed_after_the_dashboard_ends(workspace: Path) -> None:
    """Curses is gone by then, so the answer has to survive it."""
    output = drive([text_reply("everything is fine")], workspace=workspace)
    assert "everything is fine" in output


# --- the terminal is always handed back ------------------------------------


def test_the_terminal_is_restored_after_a_clean_run(workspace: Path) -> None:
    """Otherwise the user's shell stops echoing and needs `reset`."""
    output = drive([text_reply("all done")], workspace=workspace)
    assert LEAVE_ALT_SCREEN in output, "curses never left the alternate screen"


def test_the_terminal_is_restored_when_the_provider_fails(workspace: Path) -> None:
    """A crash mid-run is the case that strands a shell."""

    class Exploding(ScriptedModel):
        def send(self, history, tool_schemas=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("provider fell over")

    output = _run_in_pty(
        ["--model", "x", "task"],
        Exploding([]),
        ask=lambda _q: True,
        workspace=workspace,
    )
    assert LEAVE_ALT_SCREEN in output, "a crash left the terminal in curses mode"


def test_the_terminal_is_restored_when_the_agent_is_interrupted(
    workspace: Path,
) -> None:
    """Ctrl-C during a run must not strand the terminal either."""
    output = drive(
        [tool_call("list_files", path="."), text_reply("done")],
        workspace=workspace,
        interrupt_after=0.4,
    )
    assert LEAVE_ALT_SCREEN in output or "Traceback" in output


# --- approval inside the dashboard -----------------------------------------


def test_a_declined_write_is_reported_at_the_end(workspace: Path) -> None:
    """A run that stopped because you said no must not look successful."""
    responses = [
        tool_call("write_file", path="nope.txt", content="x"),
        text_reply("understood, I will not do that"),
    ]
    output = drive(responses, answers=["n"], workspace=workspace)
    assert "declined" in output or "Refused" in output


def test_a_refusal_does_not_stop_the_terminal_being_restored(workspace: Path) -> None:
    """Suspension for the prompt is the riskiest moment for the screen."""
    responses = [
        tool_call("edit_file", path="README.md", old="x", new="y"),
        text_reply("left it alone"),
    ]
    output = drive(responses, answers=["n"], workspace=workspace)
    assert LEAVE_ALT_SCREEN in output


def test_a_declined_write_does_not_happen(workspace: Path) -> None:
    """The refusal has to be real, not just reported."""
    responses = [
        tool_call("write_file", path="nope.txt", content="x"),
        text_reply("understood"),
    ]
    drive(responses, answers=["n"], workspace=workspace)
    assert not (workspace / "nope.txt").exists(), "wrote a file the user declined"


def test_approving_everything_still_runs_to_completion(workspace: Path) -> None:
    responses = [
        tool_call("write_file", path="approved.txt", content="yes"),
        text_reply("wrote it"),
    ]
    output = drive(responses, approve=True, workspace=workspace)
    assert "wrote it" in output


def test_an_approved_write_really_happens(workspace: Path) -> None:
    """The approval has to be doing something."""
    responses = [
        tool_call("write_file", path="approved.txt", content="yes"),
        text_reply("wrote it"),
    ]
    drive(responses, approve=True, workspace=workspace)
    assert (workspace / "approved.txt").read_text() == "yes"


def test_a_blocked_command_is_refused_even_when_everything_is_approved(
    workspace: Path,
) -> None:
    """`--yes`-style approval must not lift the blocklist.

    `approve=True` here stands in for `--yes`: it answers every
    confirmation with yes. A destructive command has to be refused by
    the blocklist regardless, because that is a refusal and not a
    question.
    """
    responses = [
        tool_call("run_command", command="rm -rf /"),
        text_reply("I cannot run that"),
    ]
    output = drive(responses, approve=True, workspace=workspace)
    assert "blocklist" in output or "blocked" in output


# --- plain mode ------------------------------------------------------------


def test_plain_mode_never_starts_curses(workspace: Path) -> None:
    """--plain has to be honoured, or piping output draws over the pipe."""
    output = drive([text_reply("done")], "--plain", workspace=workspace)
    assert ENTER_ALT_SCREEN not in output


def test_plain_mode_still_prints_the_summary(workspace: Path) -> None:
    output = drive([text_reply("done")], "--plain", workspace=workspace)
    assert "steps," in output


def test_the_workspace_is_still_shown_in_plain_mode(workspace: Path) -> None:
    """The sandbox boundary is printed on every run, dashboard or not."""
    output = drive([text_reply("done")], "--plain", workspace=workspace)
    assert "Workspace:" in output


# --- stop conditions -------------------------------------------------------


def test_the_step_limit_is_named_in_the_summary(workspace: Path) -> None:
    # Different paths on purpose: eight identical calls would trip the
    # repeat check first, and that is a different stop condition.
    responses = [tool_call("read_file", path=f"file{n}.py") for n in range(8)]
    output = drive(responses, "--plain", workspace=workspace)
    assert "step_limit" in output
    assert "AGENT_MAX_STEPS" in output


@pytest.mark.parametrize("flag", ["--plain"])
def test_plain_mode_is_accepted_before_or_after_the_task(
    flag: str, workspace: Path
) -> None:
    """Argparse order should not matter to a user in a hurry."""
    output = drive([text_reply("done")], flag, workspace=workspace)
    assert "steps," in output


# --- these tests must not touch the repository -----------------------------


def test_a_run_writes_nothing_outside_its_workspace(
    workspace: Path, tmp_path: Path
) -> None:
    """The sandbox is the point, and it has to hold for these tests too.

    Without this the suite writes `approved.txt` and `nope.txt` into the
    repository it is running in, which is both litter and a sign that the
    boundary is not being enforced anywhere it is not being measured.
    """
    responses = [
        tool_call("write_file", path="inside.txt", content="ok"),
        text_reply("done"),
    ]
    drive(responses, approve=True, workspace=workspace)
    assert (workspace / "inside.txt").exists()
    assert not (tmp_path / "inside.txt").exists()
    assert not (Path(__file__).resolve().parent.parent / "inside.txt").exists()


def test_a_write_outside_the_workspace_is_refused(workspace: Path) -> None:
    """The refusal has to reach the summary, not just the tool result."""
    responses = [
        tool_call("write_file", path="../escaped.txt", content="x"),
        text_reply("I cannot write outside the workspace"),
    ]
    output = drive(responses, "--plain", approve=True, workspace=workspace)
    assert "workspace" in output.lower()
    assert not (workspace.parent / "escaped.txt").exists()
