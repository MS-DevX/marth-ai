"""Command line entry point.

Phase 1 behaviour: read a task, send it to Gemini once, print the reply.
"""

import argparse
import sys

from dotenv import load_dotenv

from . import config, llm, loop
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
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Approve every write, edit and command without asking. "
            "Destructive commands are still refused. Only use this in a "
            "throwaway checkout or a container."
        ),
    )
    return parser.parse_args(argv)


def read_task(argument: str | None) -> str:
    """Return the task from the argument, or by asking the user."""
    if argument:
        return argument.strip()
    return input("What should I do? ").strip()


def explain_failure(exc: Exception) -> str:
    """Turn a provider error into something the user can act on.

    The raw SDK messages are long, name internal metrics, and bury the one
    fact that matters. Each case below is a different mistake by the user
    or a different limit being hit, so each gets its own next step.

    Args:
        exc: Whatever the provider raised.

    Returns:
        A short explanation. Falls back to the original text when the
        cause is not one we recognise.
    """
    text = str(exc)
    low = text.lower()

    if any(marker in low for marker in llm.PERMANENT_QUOTA_MARKERS):
        return (
            "The free tier's daily quota for this model is used up. The API "
            "reports it as a rate limit, but the limit resets tomorrow, so "
            "retrying now will not help.\n\n"
            "Use the local model instead, which has no quota:\n"
            "  AGENT_PROVIDER=openai python -m agent.main \"your task\"\n"
            "or try a different model with --model."
        )
    if "401" in low or "invalid" in low and "api key" in low:
        return (
            "The API key was rejected. Check that GEMINI_API_KEY is set in "
            "the project's .env file."
        )
    if "404" in low or "not found" in low and "model" in low:
        return (
            "That model is not available to this key. Run "
            "`python -m agent.main --list-models` to see what is."
        )
    if "could not reach" in low:
        return text
    if len(text) > 400:
        # Long provider errors are mostly quota links and metric names.
        return f"{text[:400]}\n\n(run with --list-models to check the model name)"
    return f"Error talking to the model: {type(exc).__name__}: {text}"


def ask_model(llm: LLM, task: str, auto_approve: bool = False) -> str:
    """Run the agent on a task and return its reply.

    Args:
        llm: The model to use.
        task: What the user asked for.
        auto_approve: Set from `--yes`. Stored on the config object so
            the tools can reach it without every signature carrying it.

    Errors are turned into a short message so the user sees a clean failure
    instead of a stack trace. The API key is never part of what we print.
    """
    if auto_approve:
        config.AUTO_APPROVE = True
    try:
        return loop.run(task, llm)
    except RuntimeError as exc:
        # Our own config errors (e.g. missing API key) already read well.
        return f"Configuration error: {exc}"
    except Exception as exc:  # noqa: BLE001 - CLI boundary, report and stop.
        return explain_failure(exc)


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
    print(ask_model(llm, task, auto_approve=args.yes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
