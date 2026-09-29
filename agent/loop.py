"""The agent loop.

Phase 2 stops after a single tool round trip. The full loop, where the
model may keep asking for tools until it is done, arrives in Phase 3. The
step counter and repeated-call structure below is deliberately laid out so
Phase 3 is a matter of removing the early return, not rewriting anything.
"""

from . import safety, tools
from .llm import LLM, Message, ToolResult


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


def run_once(task: str, llm: LLM) -> str:
    """Send a task, run any tool the model asks for, and reply once.

    This is one round trip: model -> tool -> model. If the model does not
    ask for a tool, the reply is returned immediately.

    Args:
        task: What the user asked for.
        llm: The model to use.

    Returns:
        The model's final text for this step.
    """
    history: list[Message] = [Message(role="user", text=task)]

    # Step 1: ask the model, giving it the tools it may call.
    first = llm.send(history, tool_schemas=tools.TOOL_SCHEMAS)
    if not first.tool_calls:
        return first.text

    for call in first.tool_calls:
        print(f"  [tool] {call.name}({call.args})")

    history.append(
        Message(
            role="model",
            text=first.text,
            tool_calls=first.tool_calls,
            raw=first.raw,
        )
    )

    # Step 2: run the requested tools and report the results back.
    results = tuple(
        ToolResult(
            call_id=call.id,
            name=call.name,
            content=run_tool_call(call.name, call.args),
        )
        for call in first.tool_calls
    )
    history.append(Message(role="tool", results=results))

    # Step 3: the model reads the results and gives its final answer.
    final = llm.send(history, tool_schemas=tools.TOOL_SCHEMAS)

    if final.tool_calls:
        # The model wants to keep going. Looping is Phase 3, so say so
        # rather than printing an empty reply.
        wanted = ", ".join(sorted({c.name for c in final.tool_calls}))
        return (
            f"{final.text}\n\n"
            f"[stopped after one round trip: the model asked for more "
            f"tools ({wanted}). Looping arrives in Phase 3.]"
        ).strip()
    return final.text
