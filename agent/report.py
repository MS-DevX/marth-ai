"""Plain terminal output: what is printed when there is no dashboard.

Two jobs. While the agent works, `LiveLog` puts one line per tool call
on stdout, which is all a non-curses terminal can usefully show. At the
end, `summary` prints what happened: how it finished, what it changed,
and anything that was refused. The refusals are in the summary on
purpose, because a run that looks finished but had three declines is
not the same as one that did what it was asked.

The same summary is printed after a dashboard run too. The live view is
gone once curses ends, and the part worth reading afterwards is exactly
the part that scrolled past.
"""

import sys
import time
from typing import TextIO

from . import config
from .runs import Run, ToolCall, describe_call, first_line

# Tools that change a specific file. Used to summarise what a run
# touched; `run_command` is deliberately absent, because a command is
# not a path and listing `python main.py` under "Changed:" is a lie.
FILE_TOOLS = frozenset({"write_file", "edit_file"})


class LiveLog:
    """Prints each tool call, and its result, as it happens.

    Kept as an object rather than a function because it is an observer
    called on every progress event, and has to remember what it has
    already printed. A plain function taking the run would reprint the
    whole log each time and bury the output in duplicates.

    Two lines per call: what was asked, and what came back. The result
    matters because without a dashboard this is the only place the user
    can see what the agent actually read, and a log that shows only the
    questions is not much use. One line of the result, not all of it,
    for the same reason the curses log shows one.

    Args:
        out: Where to print. Defaults to stdout.
    """

    def __init__(self, out: TextIO | None = None) -> None:
        self._out = out or sys.stdout
        self._shown = 0

    def __call__(self, run: Run) -> None:
        """Print any calls made since the last time this was called."""
        for call in run.calls[self._shown :]:
            print(_call_line(call), file=self._out, flush=True)
            detail = first_line(call.output)
            if detail:
                print(f"           {detail}", file=self._out, flush=True)
            self._shown += 1


def _call_line(call: ToolCall) -> str:
    """Return the one-line report of a single tool call."""
    return f"  [{call.status:>8}] {describe_call(call)}  ({call.seconds:.2f}s)"


def touched(run: Run) -> list[str]:
    """Return the paths a run wrote to, in order and without repeats.

    Args:
        run: The run to inspect.

    Only writes that succeeded count. A refused or blocked write did
    not touch anything, and listing it would misreport what happened.
    """
    seen: list[str] = []
    for call in run.calls:
        if call.status != "ok" or call.name not in FILE_TOOLS:
            continue
        path = call.args.get("path")
        if isinstance(path, str) and path not in seen:
            seen.append(path)
    return seen


def commands(run: Run) -> list[str]:
    """Return the shell commands a run actually ran.

    Args:
        run: The run to inspect.

    Kept apart from `touched` because a command is not a file. Listing
    `python main.py` under a heading that says "Changed" would have the
    reader believe a file of that name had been written.

    A command that exited non-zero is included, because it ran. One
    that was declined or blocked is not, because it did not: those are
    listed under "Refused, so not done:" instead, and repeating them
    here would report work that never happened.
    """
    return [
        str(call.args.get("command", ""))
        for call in run.calls
        if call.name == "run_command" and call.status in {"ok", "error"}
    ]


def refusals(run: Run) -> list[ToolCall]:
    """Return the calls a person or the blocklist refused."""
    return [call for call in run.calls if call.status in {"declined", "blocked"}]


def summary(run: Run, out: TextIO | None = None) -> str:
    """Return the end-of-run block, and print it.

    Args:
        run: The finished run.
        out: Where to print. Defaults to stdout.

    Returns:
        The same text, so a caller can reuse it.
    """
    stream_out = out or sys.stdout
    tally = run.counts
    lines = [
        "",
        f"  {run.outcome} in {len(run.calls)} calls, {run.steps_used} steps, "
        f"{_duration(run.duration)}",
        f"  {tally['ok']} ok"
        + (f", {tally['declined']} declined" if tally["declined"] else "")
        + (f", {tally['blocked']} blocked" if tally["blocked"] else "")
        + (f", {tally['error']} error" if tally["error"] == 1 else "")
        + (f", {tally['error']} errors" if tally["error"] > 1 else ""),
    ]

    written = touched(run)
    if written:
        lines.append("")
        lines.append("  Changed:")
        for path in written:
            lines.append(f"    {path}")

    ran = commands(run)
    if ran:
        lines.append("")
        lines.append("  Commands run:")
        for command in ran:
            lines.append(f"    {command}")

    refused = refusals(run)
    if refused:
        lines.append("")
        lines.append("  Refused, so not done:")
        for call in refused:
            why = "you declined" if call.status == "declined" else "blocklist"
            lines.append(f"    {call.name}({_arg_preview(call.args)})  [{why}]")

    if run.outcome == "step_limit":
        lines.append("")
        lines.append("  Ran out of steps. Raise AGENT_MAX_STEPS to continue.")
    if run.outcome == "repeating":
        lines.append("")
        lines.append("  The model repeated itself. Rephrasing the task may help.")

    text = "\n".join(lines)
    print(text, file=stream_out)
    return text


def _duration(seconds: float) -> str:
    """Return a duration as a short human string."""
    whole = int(seconds)
    if whole < 60:
        return f"{whole}s"
    minutes, rest = divmod(whole, 60)
    return f"{minutes}m {rest:02d}s"


def _arg_preview(args: dict) -> str:
    """Return a short preview of a call's arguments, for the refusal list."""
    return describe_call(ToolCall(step=0, name="call", args=args))


# --- listing past runs -----------------------------------------------------


def listing(runs: list[Run], out: TextIO | None = None) -> str:
    """Return a table of past runs, and print it.

    Args:
        runs: The runs, newest first.
        out: Where to print. Defaults to stdout.

    Returns:
        The same text, so a caller can reuse it.
    """
    stream_out = out or sys.stdout
    if not runs:
        text = (
            f"  No runs recorded in {config.WORKSPACE_ROOT / config.HISTORY_DIRNAME}."
        )
        print(text, file=stream_out)
        return text

    lines = [
        f"  {'run id':<24} {'when':<12} {'outcome':<11} {'steps':>5}  task",
        f"  {'-' * 24} {'-' * 12} {'-' * 11} {'-' * 5}  {'-' * 40}",
    ]
    for run in runs:
        stamp = time.strftime("%m-%d %H:%M", time.localtime(run.started))
        task = " ".join(run.task.split())
        if len(task) > 52:
            task = task[:49] + "..."
        # The id is the first column because `--run` takes it, and a
        # feature you cannot name is a feature you cannot use. Falls back
        # to the timestamp so a run with no file behind it is still
        # identifiable rather than printing a bare gap.
        run_id = run.run_id or stamp.replace(" ", "-").replace(":", "")
        lines.append(
            f"  {run_id:<24} {stamp:<12} {str(run.outcome):<11} {run.steps_used:>5}  {task}"
        )
    text = "\n".join(lines)
    print(text, file=stream_out)
    return text


def detail(run: Run, out: TextIO | None = None) -> str:
    """Return one past run in full, and print it.

    Args:
        run: The run to show.
        out: Where to print. Defaults to stdout.

    Returns:
        The same text, so a caller can reuse it.
    """
    stream_out = out or sys.stdout
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run.started))
    lines = [
        "",
        f"  run id   {run.run_id}",
        f"  task     {run.task}",
        f"  when     {stamp}",
        f"  model    {run.model}  ({run.provider})",
        f"  outcome  {run.outcome}",
        "",
    ]
    if not run.calls:
        lines.append("  No tool calls were made.")
    for call in run.calls:
        lines.append(f"  step {call.step:>2}  {_call_line(call).strip()}")
        output = call.output.strip()
        if output:
            for line in output.splitlines()[:6]:
                lines.append(f"            {line}")
    if run.error:
        lines += ["", f"  error    {run.error}"]
    if run.answer:
        lines += ["", "  answer", *[f"  {line}" for line in run.answer.splitlines()]]
    text = "\n".join(lines)
    print(text, file=stream_out)
    return text
