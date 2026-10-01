"""What a run did, in a form that can be shown and stored.

The loop used to print a line per step and return a string. That is
enough to watch a run and impossible to review afterwards. This module
is the record: the loop fills one of these, `history.py` writes it to
disk, and `tui.py` reads it while the run is happening.

Deliberately plain data. No behaviour, no file handles, nothing to keep
alive between calls, so a run can be serialised with `dataclasses.asdict`
and compared in a test without a terminal anywhere in sight.
"""

import time
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Literal

# Why a run ended. The loop has exactly three exits and the user needs to
# tell them apart: "the agent finished" and "the agent gave up" look the
# same in a log otherwise.
Outcome = Literal["finished", "repeating", "step_limit", "failed"]

# How a single tool call ended. `declined` and `blocked` are kept
# distinct because they mean different things: one was a person saying
# no, the other was the blocklist refusing regardless.
CallStatus = Literal["ok", "declined", "blocked", "error"]


@dataclass
class ToolCall:
    """One tool call the model asked for, and what came of it.

    Args:
        step: Which step of the loop this was, 1-based.
        name: The tool name, as the model spelled it.
        args: The arguments, as the model sent them.
        status: How it ended.
        output: The tool's return value, or the refusal message.
        seconds: How long the tool took. Only meaningful for the tool
            itself; a declined call is near-zero because nothing ran.
        approved: Whether a person said yes, or None if the tool did not
            need approval.
    """

    step: int
    name: str
    args: dict[str, Any]
    status: CallStatus = "ok"
    output: str = ""
    seconds: float = 0.0
    approved: bool | None = None

    @property
    def summary(self) -> str:
        """Return a one-line description of the call and its result.

        Built here rather than in the renderer because both the TUI and
        the history viewer want it, and two versions of "how do I say
        this in a line" would drift apart.
        """
        return describe_call(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the call as plain data, ready for JSON."""
        data = asdict(self)
        data["summary"] = self.summary
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """Rebuild a call from stored JSON.

        Args:
            data: A call as read back from disk.

        Returns:
            The call. Unknown keys are ignored, so a log written by a
            version with more fields still loads.

        Raises:
            ValueError: if a field with no default is missing. That
                means the record is not a tool call at all, and the
                caller is better off being told than handed a call with
                a fabricated step number.
        """
        known = {field.name for field in fields(cls)}
        usable = {key: value for key, value in data.items() if key in known}
        missing = [
            field.name
            for field in fields(cls)
            if field.name not in usable
            and field.default is MISSING
            and field.default_factory is MISSING
        ]
        if missing:
            raise ValueError(f"tool call record is missing: {', '.join(missing)}")
        return cls(**usable)


@dataclass
class Run:
    """Everything about one agent run, from task to final answer.

    Args:
        task: What the user asked for.
        model: The model that ran it.
        provider: Which provider served it. Defaults to "ollama" because
            that is the only one there is now; it stays a field rather
            than being deleted because history files already carry it, and
            a run that recorded "openai" is a true fact about a run that
            really happened.
        started: Unix time the run began.
        finished: Unix time it ended. 0 while it is still going.
        outcome: Why it ended. "running" until something is known.
        answer: The model's final text, or the reason it stopped.
        calls: Every tool call, in order.
        steps_used: Loop iterations used.
        error: Set when the run died rather than stopped.
    """

    task: str
    model: str = ""
    provider: str = "ollama"
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    outcome: Outcome | str = "running"
    answer: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    steps_used: int = 0
    error: str = ""
    run_id: str = ""
    """The `--run` argument for this run, filled in when it is loaded.

    Set from the filename rather than stored in the file, because the
    filename is where it already is: storing it in both places means the
    two can disagree, and the copy inside the file is the one a user
    cannot see. Empty for a run still in progress, which has no file yet.
    """

    @property
    def duration(self) -> float:
        """Return how long the run took in seconds.

        Uses `time.time()` for a running call rather than a monotonic
        clock, so it matches the stored timestamps; a clock jump
        mid-run is not worth the extra plumbing here.
        """
        return (self.finished or time.time()) - self.started

    @property
    def counts(self) -> dict[str, int]:
        """Return how many calls ended each way.

        The decline and block counts are the ones worth surfacing: a run
        that looks finished but had three refusals is not the same as
        one that did what it was asked.
        """
        tally = {"ok": 0, "declined": 0, "blocked": 0, "error": 0}
        for call in self.calls:
            tally[call.status] = tally.get(call.status, 0) + 1
        return tally

    def to_dict(self) -> dict[str, Any]:
        """Return the run as plain data, ready for JSON."""
        return {
            "task": self.task,
            "model": self.model,
            "provider": self.provider,
            "started": self.started,
            "finished": self.finished,
            "outcome": self.outcome,
            "answer": self.answer,
            "steps_used": self.steps_used,
            "error": self.error,
            "calls": [call.to_dict() for call in self.calls],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        """Rebuild a run from stored JSON.

        Tolerant of records written by an older version: an unknown key
        is ignored and a missing one falls back to a default, so opening
        a log from a previous commit does not raise.
        """
        run = cls(
            task=data.get("task", ""),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            started=data.get("started", 0.0),
            finished=data.get("finished", 0.0),
            outcome=data.get("outcome", "unknown"),
            answer=data.get("answer", ""),
            steps_used=data.get("steps_used", 0),
            error=data.get("error", ""),
        )
        for raw in data.get("calls", []):
            run.calls.append(ToolCall.from_dict(raw))
        return run


# A leading marker per outcome, so a glance down the log separates a
# refusal from a success. The value for an unrecognised status is a
# question mark rather than a crash: a log written by a later version of
# this file is exactly what somebody opens the history viewer to read,
# and raising KeyError on it would make that impossible.
STATUS_MARKERS = {"ok": " ", "declined": "!", "blocked": "X", "error": "E"}
UNKNOWN_MARKER = "?"


def describe_call(call: ToolCall) -> str:
    """Return a one-line description of a tool call.

    The arguments are folded onto one line and cut if long, because a
    display column cannot hold a file's contents. Read-only calls are
    unmarked and anything else is tagged with `STATUS_MARKERS`, so a
    glance separates what the agent looked at from what it changed.
    """
    args = ", ".join(
        f"{key}={_shorten(value)}" for key, value in list(call.args.items())[:3]
    )
    if len(call.args) > 3:
        args += ", ..."
    marker = STATUS_MARKERS.get(call.status, UNKNOWN_MARKER)
    line = f"{call.name}({args})"
    return f"{marker} {line}" if marker != " " else line


def _shorten(value: Any, limit: int = 40) -> str:
    """Return `value` as a short single-line string.

    Arguments go into a one-line display, so a multi-line file body or a
    whole file's contents have to be cut rather than wrecking the layout.
    """
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def first_line(text: str, limit: int = 120) -> str:
    """Return the most informative single line of `text`, trimmed to fit.

    A tool result is often several lines - a file, a diff, a command's
    output. Every live view wants one line of it and none of them want a
    different one, so it lives here rather than in each renderer. The
    rest is in the history file, one keypress away.

    "Most informative" is not the same as "first". A command's result
    opens with its own labels:

        exit code: 0
        --- stdout ---
        Hello, world!

    Taking the first line would show the exit code and hide the output,
    which is the one part worth seeing. So label lines are skipped in
    favour of the first line carrying content, and only if there is no
    such line does the first line of anything win - otherwise a command
    that printed nothing would show no detail at all instead of saying
    it exited cleanly.

    The labels are recognised by prefix because `run_command` writes
    them. If that format changes, this is the one place to change with
    it, and the tests below are what notice.
    """
    lines = [line.strip() for line in text.splitlines()]
    for stripped in lines:
        if stripped and not _is_label(stripped):
            return stripped[:limit]
    for stripped in lines:
        if stripped:
            return stripped[:limit]
    return ""


# The structural lines `run_command` puts at the top of its result, and
# the timeout line that replaces them. Content, not structure.
_RESULT_LABELS = ("exit code:", "--- stdout ---", "--- stderr ---", "TIMED OUT", "(no output)")


def _is_label(line: str) -> bool:
    """Return whether `line` is a heading rather than content."""
    return line.startswith(_RESULT_LABELS)
