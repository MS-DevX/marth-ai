"""Command line entry point.

Phase 1 behaviour: read a task, send it to Gemini once, print the reply.
"""

import argparse
import sys

from dotenv import load_dotenv

from . import config, loop
from .llm import LLM, build_llm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="agent",
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
        help="List the model names this API key can use, then exit.",
    )
    return parser.parse_args(argv)


def read_task(argument: str | None) -> str:
    """Return the task from the argument, or by asking the user."""
    if argument:
        return argument.strip()
    return input("What should I do? ").strip()


def ask_model(llm: LLM, task: str) -> str:
    """Run the agent on a task and return its reply.

    Errors are turned into a short message so the user sees a clean failure
    instead of a stack trace. The API key is never part of what we print.
    """
    try:
        return loop.run_once(task, llm)
    except RuntimeError as exc:
        # Our own config errors (e.g. missing API key) already read well.
        return f"Configuration error: {exc}"
    except Exception as exc:  # noqa: BLE001 - CLI boundary, report and stop.
        return f"Error talking to the model: {type(exc).__name__}: {exc}"


def list_models(llm: LLM) -> list[str]:
    """Return the model names this API key can see.

    Models can be listed by an account but still be rejected on use (for
    example retired ones), so this is a starting point, not a guarantee.
    """
    return llm.list_model_names()


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    # An explicit path. `load_dotenv()` with no argument walks up from the
    # *calling file*, which is fragile when the code is run from somewhere
    # unusual, so we point it at the project root instead.
    load_dotenv(config.PROJECT_ROOT / ".env")

    args = parse_args(argv)

    if args.list_models:
        llm = build_llm(args.model)
        for name in list_models(llm):
            print(name)
        return 0

    task = read_task(args.task)
    if not task:
        print("No task given. Nothing to do.")
        return 1

    # Printed every run so the sandbox boundary and the target model are
    # never a surprise.
    print(f"Workspace: {config.WORKSPACE_ROOT}")
    print(f"Provider:  {config.PROVIDER}")
    print(f"Model:     {args.model or config.MODEL_NAME}\n")

    llm = build_llm(args.model)
    print(ask_model(llm, task))
    return 0


if __name__ == "__main__":
    sys.exit(main())
