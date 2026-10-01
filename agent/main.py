"""Command line entry point.

Starts the agent on a task, or shows what it did on a past one. Three
output modes: a curses dashboard while the run works, one line per tool
call when there is no usable terminal, and the stored history when
asked for. All three end with the same summary, because that is the
part worth reading afterwards.
"""

import argparse
import sys
from collections.abc import Callable

from . import config, history, llm, loop, report, setup, tui
from .llm import LLM, build_llm
from .runs import Run


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser, built but not run.

    Separate from `parse_args` so the set of flags can be asked for
    without parsing anything, which is what keeps the README honest: a
    test can check that every flag the README documents is one this
    parser accepts, and would fail on a renamed one.
    """
    parser = argparse.ArgumentParser(
        prog="marth",
        description="A small terminal coding agent.",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="The task to give the agent. Omit to be prompted for it.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Override the model (default from config: {config.MODEL_NAME}).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List the model names the server has, then exit.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Approve every write, edit and command without asking. "
            "Destructive commands are still refused. Only use this in a "
            "throwaway checkout or a container."
        ),
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="List past runs and exit.",
    )
    parser.add_argument(
        "--run",
        default=None,
        metavar="ID",
        help="Show one past run in full. The ID is the first column of --history.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help=(
            "Install what a run needs: check for Ollama and the local "
            "model, download the model if it is missing, and save the "
            "choice. Then exit."
        ),
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Skip the live dashboard and print one line per tool call.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not record this run. Nothing is written to .marth-ai/.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: The arguments to parse, or None for the real ones.
    """
    return build_parser().parse_args(argv)


def read_task(argument: str | None) -> str:
    """Return the task from the argument, or by asking the user."""
    if argument:
        return argument.strip()
    return input("What should I do? ").strip()


def explain_failure(exc: Exception) -> str:
    """Turn a server error into something the user can act on.

    The raw messages are long, name internal metrics, and bury the one
    fact that matters. Each case below is a different thing being wrong
    with the machine or the model, so each gets its own next step.

    Args:
        exc: Whatever raised.

    Returns:
        A short explanation. Falls back to the original text when the
        cause is not one we recognise.
    """
    text = str(exc)
    low = text.lower()

    # A model that is not on the machine. This is by far the most common
    # failure on a local setup, and Ollama reports it as a plain 404, which
    # on its own looks like a bug in the agent. So it is checked first and
    # answered with the command that fixes it, not with a list to read.
    if "404" in low or ("not found" in low and "model" in low):
        return (
            "That model is not on this machine. A local server answers a "
            "request for a model it does not have with a 404, which is "
            "indistinguishable from a typo in the name.\n\n"
            "Fetch it, then run the task again:\n"
            "  marth --setup\n"
            "Or see what is already downloaded:\n"
            "  marth --list-models"
        )
    if "connection refused" in low or "could not reach" in low:
        return (
            f"Could not reach a model server at {config.OPENAI_BASE_URL}.\n\n"
            "If Ollama is installed but not running, start it:\n"
            "  ollama serve\n"
            "Then fetch a model:\n"
            "  marth --setup"
        )
    if "401" in low or ("invalid" in low and "api key" in low):
        return (
            "The server rejected the API key. Ollama ignores it, so this "
            "only happens if OPENAI_BASE_URL was repointed at a cloud "
            "endpoint. Check OPENAI_API_KEY in the settings file."
        )
    if any(marker in low for marker in llm.PERMANENT_QUOTA_MARKERS):
        return (
            "The provider's quota for this model is used up. It reports that "
            "as a rate limit, but the limit does not reset on the timescale a "
            "retry would take, so waiting will not help.\n\n"
            "A local model has no quota. Either repoint OPENAI_BASE_URL in "
            "the settings file at a local Ollama server, or choose another "
            "model with --model."
        )
    if len(text) > 400:
        # Long server errors are mostly quota links and metric names.
        return f"{text[:400]}\n\n(run with --list-models to check the model name)"
    return f"Error talking to the model: {type(exc).__name__}: {text}"


def list_models(llm: LLM) -> list[str]:
    """Return the model names the server has.

    For a local server that is the list of what is downloaded, so this is
    also the honest answer to "what can I run right now". A model can be
    listed and still be rejected on use, so it is a starting point.
    """
    return llm.list_model_names()


# --- running a task --------------------------------------------------------


def run_task(
    task: str,
    model: LLM,
    *,
    live: bool = True,
    record: bool = True,
) -> Run:
    """Run the agent on a task and return the full record.

    Args:
        task: What the user asked for.
        model: The model to use.
        live: Show the curses dashboard rather than plain lines.
        record: Write the run to the history directory.

    The history log and whatever view is on screen are both observers
    on the same loop, so they are composed here rather than inside
    `loop.run_record`: the loop should not know that either exists.
    """
    run = Run(task=task, model=model.model_name)
    log = history.RunLog(run) if record else None

    def work(view: Callable[[Run], None] | None) -> Run:
        """Run the loop with the log and the view both watching.

        Fills in `run` rather than making a new record, so the object
        the dashboard is drawing and the object that gets summarised are
        the same one.
        """

        def observe(current: Run) -> None:
            """Record the newest call, then let the view redraw."""
            if log is not None and len(current.calls) > log.written:
                log.write_call(current.calls[-1])
            if view is not None:
                view(current)

        if log is not None:
            log.open()
        return loop.run_record(task, model, observe if (log or view) else None, run)

    if live and tui.available():
        finished = tui.run_with_dashboard(run, work)
    else:
        finished = work(report.LiveLog())

    if log is not None:
        # Closed here rather than inside `work`, because the outcome is
        # only known once the loop has returned.
        log.close(finished.answer, str(finished.outcome), finished.error)
    return finished


def run_and_report(
    task: str,
    model: LLM,
    *,
    live: bool = True,
    record: bool = True,
) -> str:
    """Run a task, print the outcome, and return the final text.

    Args:
        task: What the user asked for.
        model: The model to use.
        live: Show the curses dashboard rather than plain lines.
        record: Write the run to the history directory.

    Errors are turned into a short message so the user sees a clean
    failure instead of a stack trace. Each message is written here and
    names a fix, rather than being the raw response body.
    """
    try:
        run = run_task(task, model, live=live, record=record)
    except Exception as exc:  # noqa: BLE001 - CLI boundary, report and stop.
        return explain_failure(exc)

    report.summary(run)
    return run.answer


# --- setup -----------------------------------------------------------------


def show_setup(model: str | None) -> int:
    """Install what a run needs. Returns a process exit code.

    Args:
        model: The model to install, or None for the recommended one.
    """
    print("Setting up marth-ai.\n")
    result = setup.run_setup(model)
    if not result.ok:
        print("\nSetup did not finish:")
        for problem in result.problems:
            print(f"  - {problem}")
        return 1
    print(
        f"\nReady. Run a task in any directory:\n"
        f"  marth \"explain what this repo does\"\n\n"
        f"The agent will only touch the directory you are in. Point it "
        f"somewhere else with AGENT_WORKSPACE=/path/to/repo."
    )
    return 0


# --- history commands ------------------------------------------------------


def show_history() -> int:
    """Print the list of past runs. Returns a process exit code."""
    runs = []
    for path in history.list_runs():
        try:
            runs.append(history.load_run(path))
        except (OSError, ValueError) as exc:
            # One unreadable file should not hide the other 199.
            print(f"  [skipping {path.name}: {exc}]", file=sys.stderr)
    report.listing(runs)
    return 0


def show_run(run_id: str) -> int:
    """Print one past run in full. Returns a process exit code."""
    matches = [path for path in history.list_runs() if path.name.startswith(run_id)]
    if not matches:
        print(f"No run matching {run_id!r}. Try --history to see what is there.")
        return 1
    if len(matches) > 1:
        # Prefixes are allowed because the timestamp part of an id is
        # guessable, but guessing wrong must not silently show a
        # different run: the user asked about one run and would be reading
        # another's output as though it were this one's.
        print(
            f"{run_id!r} matches {len(matches)} runs: "
            f"{', '.join(path.stem for path in matches[:4])}. "
            f"Use more of the id."
        )
        return 1
    try:
        report.detail(history.load_run(matches[0]))
        return 0
    except (OSError, ValueError) as exc:
        print(f"  Could not read {matches[0].name}: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    # The settings file is loaded by config, at import, because the values
    # it holds have to be in the environment before config reads them. A
    # second load here would be too late to matter and would only look
    # like it was doing something.
    args = parse_args(argv)

    if args.history:
        return show_history()
    if args.run:
        return show_run(args.run)
    if args.list_models:
        # A missing server is the common case here -- `--list-models` is
        # often run before `--setup` has ever been run -- and it arrives
        # as a connection error from inside llm.py. Answered with the
        # command that fixes it rather than a stack trace.
        try:
            names = list_models(build_llm(args.model))
        except Exception as exc:  # noqa: BLE001 - CLI boundary, report and stop.
            print(explain_failure(exc), file=sys.stderr)
            return 1
        for name in names:
            print(name)
        return 0
    if args.setup:
        return show_setup(args.model)

    if args.yes:
        config.AUTO_APPROVE = True
    if args.no_history:
        config.HISTORY_ENABLED = False

    task = read_task(args.task)
    if not task:
        print("No task given. Nothing to do.")
        return 1

    # Printed every run so the sandbox boundary and the target model are
    # never a surprise.
    print(f"Workspace: {config.WORKSPACE_ROOT}")
    print(f"Server:    {config.OPENAI_BASE_URL}")
    print(f"Model:     {args.model or config.MODEL_NAME}\n")

    model = build_llm(args.model)
    print(run_and_report(task, model, live=not args.plain, record=config.HISTORY_ENABLED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
