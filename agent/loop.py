"""The agent loop.

The loop is the whole point of this project, so it is written out longhand
rather than handed to a framework. One step is:

    send the conversation  ->  run whatever tools were asked for  ->  repeat

It stops when the model replies with prose and no tool calls, when it
repeats itself, or when it runs out of steps. Each of those is reported
to the user rather than being swallowed, because "the agent gave up" and
"the agent finished" look identical otherwise.
"""

import json
import time
from collections.abc import Callable
from dataclasses import replace

from . import config, prompt, safety, tools
from .llm import LLM, Message, ToolResult
from .runs import Run, ToolCall

# Shown in place of a tool result that has been dropped to fit the context
# window. It has to say something useful: the model may still want to know
# a call happened even when it can no longer see what came back.
DROPPED_RESULT = "[earlier result dropped to fit the context window]"


def run_tool_call(name: str, args: dict) -> tools.Result:
    """Run one tool by name and return what it produced.

    A tool that fails returns an error message rather than raising, so
    the model gets to read what went wrong and try something else.

    This is the single funnel for every failure a run can have: a missing
    tool, bad arguments, the sandbox, a crash inside a tool. Each becomes
    a `Result` tagged `"error"` here, so none of them can reach the
    dashboard or the history file looking like a success.

    The result is truncated here, at the single point where tool output
    enters the conversation, rather than inside each tool. That way the
    context window cannot be blown by a large file no matter which tool
    produced it, including tools added later.

    Args:
        name: The tool name the model used.
        args: The arguments the model supplied.

    Returns:
        The tool's output, capped in length, tagged with how it went.
    """
    tool = tools.TOOL_REGISTRY.get(name)
    if tool is None:
        available = ", ".join(sorted(tools.TOOL_REGISTRY))
        return tools.Result(f"Unknown tool: {name}. Available tools: {available}", "error")

    # Guard against a tool that does not take the arguments the model sent.
    try:
        outcome = tool(**args)
    except TypeError as exc:
        return tools.Result(f"Bad arguments for {name}: {exc}", "error")
    except (safety.SandboxError, safety.SecretFileError) as exc:
        # These two carry messages written for the model, so the text is
        # passed through as-is. The generic handler below would bury it
        # under a Python exception name, which tells the model nothing
        # it can act on.
        return tools.Result(str(exc), "error")
    except Exception as exc:  # noqa: BLE001 - report to the model, keep going.
        return tools.Result(f"{type(exc).__name__}: {exc}", "error")

    # A tool with no failure mode may return a bare string; that is a
    # success, and a missing `Result` should not fail the call.
    if not isinstance(outcome, tools.Result):
        return tools.Result(safety.truncate(outcome))
    return tools.Result(safety.truncate(outcome), outcome.status, outcome.approved)


def call_signature(name: str, args: dict) -> str:
    """Return a stable key for a tool call, for spotting repeats.

    Args are sorted so that `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` count
    as the same call rather than evading the repeat check.
    """
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def _history_size(history: list[Message]) -> int:
    """Return roughly how many characters the conversation will occupy."""
    total = 0
    for message in history:
        total += len(message.text)
        for call in message.tool_calls:
            total += len(call.name) + len(json.dumps(call.args, default=str))
        for result in message.results:
            total += len(result.content)
    return total


def compact_history(
    history: list[Message], budget: int | None = None
) -> list[Message]:
    """Shrink old tool results until the conversation fits `budget`.

    The conversation is re-sent in full on every step, so it has to stay
    bounded or a long run eventually asks the server for more than it can
    accept. A local 8B model has a window small enough that this is not
    theoretical: an 8k-token task hits it within a dozen steps, so an
    overflow here is a bug the user sees, not one the provider hides.

    Older results are dropped first, for two reasons. They are the bulk
    of the conversation, and they are the ones the model has already read
    and acted on. Prose is never dropped: the task and the model's own
    reasoning are what tie the steps together.

    The most recent tool result is always kept, even if that alone
    exceeds the budget. Truncating the thing the model is currently
    reasoning about would be worse than sending a long history.

    Args:
        history: The conversation so far, oldest first. Not modified.
        budget: Character ceiling. Defaults to `config.MAX_HISTORY_CHARS`.

    Returns:
        A new list, or the original one if it already fits.
    """
    cap = config.MAX_HISTORY_CHARS if budget is None else budget
    if _history_size(history) <= cap:
        return history

    compacted = list(history)
    # Oldest first, but never the final tool turn.
    for index, message in enumerate(compacted):
        if message.role != "tool" or index == len(compacted) - 1:
            continue
        if _history_size(compacted) <= cap:
            break
        compacted[index] = replace(
            message,
            results=tuple(
                replace(result, content=DROPPED_RESULT) for result in message.results
            ),
        )
    return compacted


def run(task: str, llm: LLM, on_event: Callable[[Run], None] | None = None) -> str:
    """Run the agent loop until the model is done asking for tools.

    Args:
        task: What the user asked for.
        llm: The model to use.
        on_event: Called with the in-progress run whenever something
            happens, so a UI can draw while the loop is still working.
            The same object is passed each time and keeps changing, so a
            handler that formats it immediately sees that moment's state.

    Returns:
        The model's final answer, or a plain statement of why the run
        stopped early.
    """
    return run_record(task, llm, on_event).answer


def run_record(
    task: str,
    llm: LLM,
    on_event: Callable[[Run], None] | None = None,
    record: Run | None = None,
) -> Run:
    """Run the loop, returning the whole record rather than just the text.

    `run()` is the thin wrapper most callers want; this is the one that
    produces something a dashboard or the history log can use.

    Args:
        task: What the user asked for.
        llm: The model to use.
        on_event: Called as the run progresses; see `run`.
        record: An existing record to fill in, or None to make one.

        The `record` argument is there so a caller that has to show the
        run before it starts - to draw a dashboard, or to name a history
        file - can hold one object for the whole run instead of a
        placeholder and a separate filled-in copy. Two records means the
        thing on screen is not the thing that happened.

    Returns:
        The same `Run` that was passed in, filled in.
    """
    # The prompt is prepended to the task rather than passed as a
    # separate role. Gemini, the previous provider, took its system prompt
    # in a config field rather than in the conversation, and the
    # OpenAI-compatible shape takes it as a message; plumbing it per
    # provider meant the two paths could drift apart silently. Prepending
    # worked on both, and cannot be dropped by a provider that ignores
    # it.
    history: list[Message] = [
        Message(role="user", text=f"{prompt.SYSTEM_PROMPT}\n\nThe task: {task}")
    ]
    if record is None:
        record = Run(task=task, model=llm.model_name)
    else:
        # Filled in rather than replaced: the caller may already have
        # set fields the loop has no opinion about, such as the name of
        # the file the run is being written to.
        record.task = task
        record.model = llm.model_name
    seen: dict[str, int] = {}

    def notify() -> None:
        """Tell the UI something changed, if anyone is listening."""
        if on_event is not None:
            on_event(record)

    notify()
    for step in range(1, config.MAX_STEPS + 1):
        response = llm.send(compact_history(history), tool_schemas=tools.TOOL_SCHEMAS)
        record.steps_used = step

        # Prose with no tool calls is the model saying it is finished.
        if not response.tool_calls:
            record.outcome = "finished"
            record.answer = response.text.strip() or "(the model returned nothing)"
            record.finished = time.time()
            notify()
            return record

        results: list[ToolResult] = []
        for call in response.tool_calls:
            signature = call_signature(call.name, call.args)
            seen[signature] = seen.get(signature, 0) + 1

            if seen[signature] > config.MAX_REPEATED_CALLS:
                record.outcome = "repeating"
                record.answer = (
                    f"Stopped: the model asked for `{call.name}` with the same "
                    f"arguments {seen[signature] - 1} times and stopped making "
                    f"progress. Last request: {json.dumps(call.args)}. "
                    f"Rephrasing the task, or adding a tool it is missing, "
                    f"usually gets past this."
                )
                record.finished = time.time()
                notify()
                return record

            entry = ToolCall(step=step, name=call.name, args=dict(call.args))
            started = time.monotonic()
            outcome = run_tool_call(call.name, call.args)
            entry.seconds = round(time.monotonic() - started, 3)
            entry.output = outcome
            entry.status = outcome.status
            entry.approved = outcome.approved
            record.calls.append(entry)
            results.append(
                ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content=entry.output,
                )
            )
            notify()

        history.append(
            Message(
                role="model",
                text=response.text,
                tool_calls=response.tool_calls,
            )
        )
        history.append(Message(role="tool", results=tuple(results)))

    record.outcome = "step_limit"
    record.answer = (
        f"Stopped: reached the {config.MAX_STEPS}-step limit without a final "
        f"answer. The last thing the model said was:\n\n"
        f"{history[-2].text.strip() if history[-2].text else '(nothing)'}\n\n"
        f"Raise AGENT_MAX_STEPS to continue, or narrow the task."
    )
    record.finished = time.time()
    notify()
    return record
