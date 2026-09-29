"""Tests for the agent loop.

The loop is driven by a scripted fake LLM, so these run offline and in
milliseconds. What they check is the loop's own behaviour: when it stops,
what it reports, and that it never sends more than the context budget
allows. The providers themselves are covered in test_llm.py.
"""

import json

import pytest

from agent import config, loop
from agent.llm import LLMResponse, Message, ToolCall, ToolResult


class ScriptedLLM:
    """An LLM that replays a fixed list of responses, in order.

    Args:
        responses: What to return on each successive call to `send`.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.sent: list[list[Message]] = []

    def send(self, history, tool_schemas=None):  # type: ignore[no-untyped-def]
        """Record the history we were given, then return the next reply."""
        self.sent.append(list(history))
        if not self._responses:
            raise AssertionError("the loop asked for more turns than scripted")
        return self._responses.pop(0)

    def list_model_names(self):  # type: ignore[no-untyped-def]
        """Unused by the loop; present to satisfy the protocol."""
        return []


def call(name: str, **args) -> ToolCall:
    """Build a tool call the way a model would send one."""
    return ToolCall(id=f"id-{name}-{len(args)}", name=name, args=args)


def text_reply(text: str) -> LLMResponse:
    """Build a reply that asks for no tools, i.e. the model is finished."""
    return LLMResponse(text=text)


# --- stopping conditions --------------------------------------------------


def test_answers_immediately_when_no_tools_are_asked_for() -> None:
    llm = ScriptedLLM([text_reply("all done")])
    assert loop.run("do a thing", llm) == "all done"


def test_runs_a_tool_then_answers() -> None:
    """The basic shape: model asks, tool answers, model reports."""
    llm = ScriptedLLM(
        [
            LLMResponse(text="looking", tool_calls=(call("list_files", path="."),)),
            text_reply("there are three files"),
        ]
    )
    assert loop.run("how many files?", llm) == "there are three files"
    assert len(llm.sent) == 2


def test_tool_result_is_fed_back_to_the_model() -> None:
    """The whole design depends on this second call seeing the output."""
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=(call("read_file", path="a.py"),)),
            text_reply("done"),
        ]
    )
    loop.run("read a.py", llm)
    tool_turns = [m for m in llm.sent[1] if m.role == "tool"]
    assert tool_turns, "the second request carried no tool result"
    assert "a.py" in tool_turns[0].results[0].content


def test_runs_several_steps_in_a_row() -> None:
    """Phase 3 exists to get past a single round trip."""
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=(call("list_files", path="."),)),
            LLMResponse(tool_calls=(call("read_file", path="a.py"),)),
            LLMResponse(tool_calls=(call("read_file", path="b.py"),)),
            text_reply("read them both"),
        ]
    )
    assert loop.run("read a.py and b.py", llm) == "read them both"
    assert len(llm.sent) == 4


def test_multiple_tools_in_one_turn_all_run() -> None:
    llm = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=(
                    call("read_file", path="a.py"),
                    call("read_file", path="b.py"),
                )
            ),
            text_reply("both read"),
        ]
    )
    loop.run("read both", llm)
    tool_turns = [m for m in llm.sent[1] if m.role == "tool"]
    assert len(tool_turns[0].results) == 2


def test_stops_at_the_step_limit_and_says_so() -> None:
    """Running out of steps is not the same as finishing, and must not
    look like it."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config, "MAX_STEPS", 3)
    monkeypatch.setattr(config, "MAX_REPEATED_CALLS", 99)
    llm = ScriptedLLM(
        [LLMResponse(tool_calls=(call("read_file", path=f"{n}.py"),)) for n in range(10)]
    )
    result = loop.run("read everything", llm)
    assert "3-step limit" in result
    assert len(llm.sent) == 3
    monkeypatch.undo()


def test_step_limit_still_honours_a_wrong_looking_message() -> None:
    """The message must not claim success."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config, "MAX_STEPS", 2)
    monkeypatch.setattr(config, "MAX_REPEATED_CALLS", 99)
    llm = ScriptedLLM(
        [LLMResponse(tool_calls=(call("read_file", path=f"{n}.py"),)) for n in range(9)]
    )
    assert "reached the 2-step limit" in loop.run("go", llm)
    monkeypatch.undo()


def test_empty_reply_is_reported_not_returned_blank() -> None:
    llm = ScriptedLLM([text_reply("")])
    assert "returned nothing" in loop.run("say nothing", llm)


# --- repeat detection -----------------------------------------------------


def test_stops_when_the_model_repeats_the_same_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model re-reading one file is not making progress, and on a CPU
    each repetition costs real time."""
    monkeypatch.setattr(config, "MAX_REPEATED_CALLS", 2)
    llm = ScriptedLLM(
        [LLMResponse(tool_calls=(call("read_file", path="a.py"),)) for _ in range(9)]
    )
    result = loop.run("read a.py", llm)
    assert "Stopped" in result and "read_file" in result
    # Two allowed plus the one that trips the check.
    assert len(llm.sent) == 3


def test_repeat_count_resets_per_run() -> None:
    """A fresh task must not inherit the previous run's suspicion."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config, "MAX_REPEATED_CALLS", 2)
    for _ in range(2):
        llm = ScriptedLLM(
            [LLMResponse(tool_calls=(call("read_file", path="a.py"),)) for _ in range(2)]
            + [text_reply("fine")]
        )
        assert loop.run("read a.py", llm) == "fine"
    monkeypatch.undo()


def test_argument_order_does_not_evade_repeat_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise a model can loop forever just by reshuffling its keys."""
    monkeypatch.setattr(config, "MAX_REPEATED_CALLS", 1)
    llm = ScriptedLLM(
        [
            LLMResponse(
                tool_calls=(
                    ToolCall(id="1", name="read_file", args={"a": 1, "b": 2}),
                )
            ),
            LLMResponse(
                tool_calls=(
                    ToolCall(id="2", name="read_file", args={"b": 2, "a": 1}),
                )
            ),
            LLMResponse(tool_calls=(call("read_file", path="c"),)),
        ]
    )
    result = loop.run("go", llm)
    assert "Stopped" in result
    # The second call is the one that trips it, not the third.
    assert len(llm.sent) == 2
    monkeypatch.undo()


def test_different_arguments_are_not_treated_as_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_REPEATED_CALLS", 2)
    llm = ScriptedLLM(
        [
            LLMResponse(tool_calls=(call("read_file", path="a.py"),)),
            LLMResponse(tool_calls=(call("read_file", path="b.py"),)),
            LLMResponse(tool_calls=(call("read_file", path="c.py"),)),
            text_reply("read all three"),
        ]
    )
    assert loop.run("read three files", llm) == "read all three"
    monkeypatch.undo()


# --- history compaction ---------------------------------------------------


def test_short_history_is_returned_unchanged() -> None:
    history = [Message(role="user", text="hi")]
    assert loop.compact_history(history, budget=10000) is history


def test_old_results_are_dropped_to_fit_the_budget() -> None:
    """Without this a 20-step run asks for more tokens than the model has."""
    history: list[Message] = [Message(role="user", text="task")]
    for n in range(6):
        history.append(Message(role="model", tool_calls=(call("read_file", path="x"),)))
        history.append(
            Message(
                role="tool",
                results=(ToolResult(call_id="1", name="read_file", content="y" * 1000),),
            )
        )
    compacted = loop.compact_history(history, budget=2500)
    assert loop._history_size(compacted) < loop._history_size(history)


def test_the_newest_result_survives_compaction() -> None:
    """Truncating what the model is currently reasoning about would be
    worse than sending a long history."""
    history: list[Message] = [Message(role="user", text="task")]
    for _ in range(5):
        history.append(Message(role="model", tool_calls=(call("read_file", path="x"),)))
        history.append(
            Message(
                role="tool",
                results=(
                    ToolResult(call_id="1", name="read_file", content="NEWEST" + "y" * 500),
                ),
            )
        )
    compacted = loop.compact_history(history, budget=1200)
    assert "NEWEST" in compacted[-1].results[0].content


def test_prose_is_never_dropped() -> None:
    """The task and the model's own words are what tie the steps together."""
    history: list[Message] = [Message(role="user", text="IMPORTANT_TASK")]
    for _ in range(5):
        history.append(Message(role="model", tool_calls=(call("read_file", path="x"),)))
        history.append(
            Message(
                role="tool",
                results=(ToolResult(call_id="1", name="read_file", content="y" * 900),),
            )
        )
    compacted = loop.compact_history(history, budget=1000)
    assert compacted[0].text == "IMPORTANT_TASK"


def test_dropped_results_say_so() -> None:
    """A silent drop reads as an empty file and sends the model hunting."""
    history: list[Message] = [Message(role="user", text="task")]
    for _ in range(5):
        history.append(Message(role="model", tool_calls=(call("read_file", path="x"),)))
        history.append(
            Message(
                role="tool",
                results=(ToolResult(call_id="1", name="read_file", content="y" * 900),),
            )
        )
    compacted = loop.compact_history(history, budget=1000)
    assert any(
        loop.DROPPED_RESULT in r.content for m in compacted if m.role == "tool"
        for r in m.results
    )


def test_compaction_does_not_mutate_the_original() -> None:
    """The loop keeps building on the same list across steps."""
    history: list[Message] = [Message(role="user", text="task")]
    for _ in range(4):
        history.append(Message(role="model", tool_calls=(call("read_file", path="x"),)))
        history.append(
            Message(
                role="tool",
                results=(ToolResult(call_id="1", name="read_file", content="y" * 800),),
            )
        )
    before = loop._history_size(history)
    loop.compact_history(history, budget=1000)
    assert loop._history_size(history) == before


def test_a_twenty_step_run_stays_within_the_context_budget() -> None:
    """The regression that motivates compaction, checked end to end."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config, "MAX_STEPS", 20)
    monkeypatch.setattr(config, "MAX_REPEATED_CALLS", 99)
    monkeypatch.setattr(config, "MAX_HISTORY_CHARS", 24000)
    monkeypatch.setattr(
        "agent.tools.read_file", lambda path: "z" * config.MAX_OUTPUT_CHARS
    )
    llm = ScriptedLLM(
        [LLMResponse(tool_calls=(call("read_file", path=f"{n}.py"),)) for n in range(30)]
    )
    loop.run("read every file", llm)
    assert len(llm.sent) == 20
    for sent in llm.sent:
        assert loop._history_size(sent) <= config.MAX_HISTORY_CHARS
    monkeypatch.undo()


# --- tool execution -------------------------------------------------------


def test_unknown_tool_is_reported_to_the_model_not_raised() -> None:
    """The model should be able to read what went wrong and recover."""
    result = loop.run_tool_call("no_such_tool", {})
    assert "Unknown tool" in result
    assert "list_files" in result


def test_bad_arguments_are_reported_to_the_model() -> None:
    result = loop.run_tool_call("read_file", {})
    assert "Bad arguments" in result


def test_secret_file_is_refused_with_an_explanation() -> None:
    result = loop.run_tool_call("read_file", {"path": ".env"})
    assert "secret" in result.lower()
    assert "AIza" not in result


def test_tool_output_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MAX_OUTPUT_CHARS", 500)
    monkeypatch.setattr("agent.tools.read_file", lambda path: "z" * 100000)
    assert len(loop.run_tool_call("read_file", {"path": "a"})) < 1000
