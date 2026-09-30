"""Past runs, written to disk as they happen.

Nothing here is required for the agent to work. If history fails to
write, the run carries on: losing a log is not a reason to lose the
work. Every write is therefore wrapped so a full disk or a read-only
checkout cannot turn into an exception mid-task.

Layout, under `<workspace>/.marth-ai/runs/`:

    20260930-141205-a3f1.jsonl

One file per run, named with the start time so the files sort
chronologically by name and the oldest can be trimmed without opening
any of them. Each line is one event, appended as it happens, so a run
killed halfway still leaves a readable record of what it got through.
That is the reason for JSONL over one JSON document per run: a
half-written JSON file tells you nothing, while a truncated final line
costs one entry.

The header line carries the task and the model. The call lines follow.
The last line is the footer, carrying the outcome and the final answer.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from . import config
from .runs import Run, ToolCall

# The three kinds of line in a run file. Named so `load_run` can reject
# anything else and so a truncated final line is recognisable.
HEADER = "header"
CALL = "call"
FOOTER = "footer"

FILENAME_TIME = "%Y%m%d-%H%M%S"
_FILENAME = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}\.jsonl$")


def runs_dir(root: Path | None = None) -> Path:
    """Return the directory run files are written to.

    Args:
        root: The workspace to look in. Defaults to the configured one.

    Returns:
        The runs directory. Not created; writing is what creates it.
    """
    base = (root or config.WORKSPACE_ROOT) / config.HISTORY_DIRNAME / "runs"
    return base


def new_run(task: str, model: str, provider: str) -> Run:
    """Return an empty run record with the header fields filled in."""
    return Run(task=task, model=model, provider=provider, started=time.time())


class RunLog:
    """Appends one run's events to a file as they happen.

    Args:
        run: The record to write. Held by reference and rewritten in
            place, so a change to the record is picked up by the next
            `write_call`.
        directory: Where to write. Defaults to the runs directory.
    """

    def __init__(self, run: Run, directory: Path | None = None) -> None:
        self._run = run
        self._dir = directory or runs_dir()
        self._path = self._dir / _filename_for(run)
        self._enabled = config.HISTORY_ENABLED
        self._failed = False
        self.written = 0

    @property
    def path(self) -> Path:
        """Return the file this run is being written to."""
        return self._path

    @property
    def failed(self) -> bool:
        """Return whether writing has given up.

        Once a write fails the log stops trying. A run that is going to
        fail to log every step would otherwise try once per step and
        report nothing, which is slower and no more informative.
        """
        return self._failed

    def open(self) -> None:
        """Create the file and write the header line.

        The header carries only what cannot change during the run: the
        task, the model, and when it started. The outcome and the answer
        go in the footer, so a file with no footer unambiguously means
        the run did not finish. Writing the whole record up front would
        bake in whatever the answer was at the time, which is nothing.

        Silently does nothing when history is disabled or the directory
        cannot be created. A log is a convenience; failing to make one
        must not stop the agent from doing the task.
        """
        if not self._enabled or self._failed:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            # A run whose name already exists would be appended to,
            # producing one file containing two runs. Two runs in the
            # same second is possible when scripted, so add a suffix
            # rather than assume the timestamp is unique.
            self._path = self._unique_path()
            self._append(
                {
                    "type": HEADER,
                    "task": self._run.task,
                    "model": self._run.model,
                    "provider": self._run.provider,
                    "started": self._run.started,
                }
            )
        except OSError:
            self._failed = True

    def write_call(self, call: ToolCall) -> None:
        """Append one finished tool call.

        Args:
            call: The call that just ran.
        """
        if not self._enabled or self._failed:
            return
        self._append({"type": CALL, **call.to_dict()})
        self.written += 1

    def close(self, answer: str, outcome: str, error: str = "") -> None:
        """Append the footer and trim old runs.

        Args:
            answer: The model's final text, or why the run stopped.
            outcome: One of the `Outcome` values.
            error: Set if the run died rather than stopped cleanly.

        Carries the step count as well as the outcome. It is not
        derivable from the calls: a step can produce several calls, and
        the last step of a run often produces none, because the model
        replied with prose instead.
        """
        if not self._enabled or self._failed:
            return
        self._append(
            {
                "type": FOOTER,
                "outcome": outcome,
                "answer": answer,
                "error": error,
                "steps_used": self._run.steps_used,
                "finished": self._run.finished or time.time(),
            }
        )
        self._trim()

    # -- internals ---------------------------------------------------------

    def _unique_path(self) -> Path:
        """Return a path not already taken, adding a counter if needed."""
        if not self._path.exists():
            return self._path
        stem = self._path.stem
        for n in range(1, 100):
            candidate = self._dir / f"{stem}-{n}.jsonl"
            if not candidate.exists():
                return candidate
        return self._path

    def _append(self, record: dict[str, Any]) -> None:
        """Write one JSON line, opening and closing each time.

        Opened per line rather than held open, so a crash cannot leave a
        half-written buffer, and so several runs in parallel do not
        interleave bytes in one handle.
        """
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except (OSError, TypeError, ValueError):
            self._failed = True

    def _trim(self) -> None:
        """Delete the oldest run files beyond the configured limit.

        Sorted by filename, which is chronological by construction. A
        failure here is ignored: refusing to record a run because the
        directory is full of old ones would be the wrong way round.
        """
        try:
            files = sorted(
                path
                for path in self._dir.iterdir()
                if path.suffix == ".jsonl" and path.is_file()
            )
            for old in files[: max(0, len(files) - config.MAX_HISTORY_RUNS)]:
                old.unlink(missing_ok=True)
        except OSError:
            pass


def _filename_for(run: Run) -> str:
    """Return the filename a run is stored under.

    The suffix is a short random tag. Timestamps collide when a script
    starts two runs in the same second, and a collided filename would
    silently merge two runs into one file.
    """
    stamp = time.strftime(FILENAME_TIME, time.localtime(run.started))
    return f"{stamp}-{os.urandom(2).hex()}.jsonl"


def list_runs(root: Path | None = None) -> list[Path]:
    """Return run files, newest first.

    Args:
        root: The workspace to look in. Defaults to the configured one.

    Returns:
        Every readable run file, newest first by name. An absent
        directory is an empty list, not an error: a project that has
        never run the agent has no history, and that is not a problem.
    """
    directory = runs_dir(root)
    if not directory.is_dir():
        return []
    try:
        files = [path for path in directory.iterdir() if path.suffix == ".jsonl"]
    except OSError:
        return []
    return sorted(files, reverse=True)


def load_run(path: Path) -> Run:
    """Rebuild a run from its file.

    Args:
        path: The run file to read.

    Returns:
        The run, with whatever was recorded. A file truncated mid-write
        yields everything up to the break rather than an error, since a
        partial record of a crashed run is the most useful thing to have.

    Raises:
        ValueError: if the file is not a run file at all.
    """
    header: dict[str, Any] | None = None
    raw_calls: list[dict[str, Any]] = []
    footer: dict[str, Any] | None = None

    with path.open(encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]

    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            # A torn final line is what a crash mid-write leaves behind,
            # so the last line is allowed to be unparseable. Damage
            # anywhere else means the file is not what we think it is,
            # and quietly ignoring it would hide a real problem.
            if index == len(lines) - 1:
                break
            raise ValueError(f"{path.name} line {index + 1}: {exc}") from exc
        kind = record.get("type")
        if kind == HEADER:
            header = record
        elif kind == CALL:
            raw_calls.append(record)
        elif kind == FOOTER:
            footer = record

    # Checked before the calls are built, so a file that is not a run
    # file at all is reported as such rather than as a malformed call.
    if header is None:
        raise ValueError(f"{path.name} has no header line")

    run = Run.from_dict(header)
    run.calls = [ToolCall.from_dict(record) for record in raw_calls]
    # Taken from the filename, which is where `--run` looks for it. Storing
    # it inside the file as well would mean two copies that can disagree,
    # and this is the one the user can actually see.
    run.run_id = path.stem
    if footer is not None:
        run.outcome = footer.get("outcome", "unknown")
        run.answer = footer.get("answer", "")
        run.error = footer.get("error", "")
        run.steps_used = footer.get("steps_used", 0)
        run.finished = footer.get("finished", 0.0)
    else:
        # No footer means the process died before it could record how
        # it ended. Saying "running" is the honest reading; anything
        # else would claim an outcome nobody observed.
        run.outcome = "running"
    return run


def clear(root: Path | None = None) -> int:
    """Delete every stored run and return how many were removed.

    Args:
        root: The workspace to clear. Defaults to the configured one.

    Returns:
        The number of files deleted. Zero when there was nothing to
        delete, which is a success rather than a failure.
    """
    removed = 0
    for path in list_runs(root):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
