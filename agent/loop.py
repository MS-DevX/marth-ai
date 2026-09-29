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
from dataclasses import replace

from . import config, safety, tools
from .llm import LLM, Message, ToolResult

# Shown in place of a tool result that has been dropped to fit the context
# window. It has to say something useful: the model may still want to know
# a call happened even when it can no longer see what came back.
DROPPED_RESULT = "[earlier result dropped to fit the context window]"


def run_tool_call(name: str, args: dict) -> str:
    """Run one tool by name and return its result as a string.

    A tool that fails returns an error string rather than raising, so the
    model gets to read what went wrong and try something else.

    The result is truncated here, at the single point where tool output
    enters the conversation, rather than inside each tool. That way the
    context window cannot be blown by a large file no matter which tool
    produced it, including tools added later.

    Args:
        name: The tool name the model used.
        args: The arguments the model supplied.

    Returns:
        The tool's output, capped in length, or a message describing why
        it could not run.
    """
    tool = tools.TOOL_REGISTRY.get(name)
    if tool is None:
        available = ", ".join(sorted(tools.TOOL_REGISTRY))
        return f"Unknown tool: {name}. Available tools: {available}"

    # Guard against a tool that does not take the arguments the model sent.
    try:
        return safety.truncate(str(tool(**args)))
    except TypeError as exc:
        return f"Bad arguments for {name}: {exc}"
    except safety.SecretFileError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - report to the model, keep going.
        return f"{name} failed: {type(exc).__name__}: {exc}"


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
    bounded or a long run eventually asks the provider for more than it
    can accept. Gemini's window is large enough to ignore this, which is
    why a local model is the honest place to develop against.

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


def run(task: str, llm: LLM) -> str:
    """Run the agent loop until the model is done asking for tools.

    Args:
        task: What the user asked for.
        llm: The model to use.

    Returns:
        The model's final answer, or a plain statement of why the run
        stopped early.
    """
    history: list[Message] = [Message(role="user", text=task)]
    seen: dict[str, int] = {}

    for step in range(1, config.MAX_STEPS + 1):
        response = llm.send(compact_history(history), tool_schemas=tools.TOOL_SCHEMAS)

        # Prose with no tool calls is the model saying it is finished.
        if not response.tool_calls:
            return response.text.strip() or "(the model returned nothing)"

        results: list[ToolResult] = []
        for call in response.tool_calls:
            signature = call_signature(call.name, call.args)
            seen[signature] = seen.get(signature, 0) + 1

            if seen[signature] > config.MAX_REPEATED_CALLS:
                return (
                    f"Stopped: the model asked for `{call.name}` with the same "
                    f"arguments {seen[signature] - 1} times and stopped making "
                    f"progress. Last request: {json.dumps(call.args)}. "
                    f"Rephrasing the task, or adding a tool it is missing, "
                    f"usually gets past this."
                )

            print(f"  [step {step}] {call.name}({json.dumps(call.args)})")
            results.append(
                ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content=run_tool_call(call.name, call.args),
                )
            )

        history.append(
            Message(
                role="model",
                text=response.text,
                tool_calls=response.tool_calls,
                raw=response.raw,
            )
        )
        history.append(Message(role="tool", results=tuple(results)))

    return (
        f"Stopped: reached the {config.MAX_STEPS}-step limit without a final "
        f"answer. The last thing the model said was:\n\n"
        f"{history[-2].text.strip() if history[-2].text else '(nothing)'}\n\n"
        f"Raise AGENT_MAX_STEPS to continue, or narrow the task."
    )
