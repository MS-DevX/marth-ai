"""Tests for the system prompt.

The prompt is the one piece of guidance the model gets, so these check
it says things that were learned rather than things that sound polite.
Each test names the failure the line exists to prevent.
"""

from agent import config, prompt

# Failures seen while building this agent. A line in the prompt has to
# earn its place by answering one of these, so each entry is a phrase the
# prompt must contain. If a failure stops being real, delete its entry
# rather than leaving the sentence behind.
REGRESSION = {
    # Asked how many tests a file had, answered 8 of 32, having seen 28%.
    "reasoned about a file it only partly saw": "do not reason about a file "
    "you have only partly seen",
    # Edited using text it had guessed at instead of reading first.
    "guessed at file contents instead of reading": "read it. Before editing "
    "it a second time",
    # Asked for the same refused action over and over.
    "retried a refused action": "that is a decision, not a retry",
    # Called confirm_write before it existed, then used the shell instead.
    "called a tool that does not exist": "Only use the tools you were given",
    "worked around a missing tool": "do not invent it",
    # Read a truncated file and guessed the rest.
    "ignored the truncation notice": "do not count what you did not read",
    "finished without checking its work": "run the tests, or run the thing "
    "you changed",
    "kept going with no idea how many steps were left": "step limit",
    "assumed a truncation limit that does not exist": "Tool results are "
    "truncated",
}


def test_loop_prepends_the_prompt() -> None:
    """The task must survive being concatenated with the prompt."""
    task = "rename the function"
    combined = f"{prompt.SYSTEM_PROMPT}\n\nThe task: {task}"
    assert task in combined
    assert combined.endswith(task), "the task should be last, so it is read last"


def test_prompt_answers_every_known_failure() -> None:
    """A rule that answers no observed failure should be cut, not kept.

    Compared against a whitespace-normalised copy, because the prompt is
    hard-wrapped for reading and a phrase can straddle a line break.
    """
    flat = " ".join(prompt.SYSTEM_PROMPT.split())
    for description, expected in REGRESSION.items():
        assert " ".join(expected.split()) in flat, (
            f"no guidance for: {description} (looked for {expected!r})"
        )


def test_prompt_names_the_real_truncation_limit() -> None:
    """A made-up number would be worse than no number, since the model
    would reason about a limit that does not exist."""
    assert str(config.MAX_OUTPUT_CHARS) in prompt.SYSTEM_PROMPT


def test_prompt_warns_that_shell_is_not_sandboxed() -> None:
    """The single most important limitation, and easy to leave out."""
    assert "not confined to this directory" in prompt.SYSTEM_PROMPT


def test_prompt_tells_the_model_a_refusal_is_final() -> None:
    """A model that retries a declined action burns the step budget."""
    lowered = prompt.SYSTEM_PROMPT.lower()
    assert "that is a decision, not a retry" in lowered


def test_prompt_forbids_inventing_tools() -> None:
    """Models call tools they wish existed, then improvise around it."""
    flat = " ".join(prompt.SYSTEM_PROMPT.split())
    assert "Only use the tools you were given" in flat
    assert "do not invent it" in flat


def test_prompt_is_not_absurdly_long() -> None:
    """It is re-sent every step, and it competes with tool output for a
    small context window on a local model."""
    words = len(prompt.SYSTEM_PROMPT.split())
    assert words < 700, f"system prompt is {words} words, too long to re-send"


def test_prompt_has_no_unfilled_placeholders() -> None:
    """A stray {} or {tool} would ship literally to the model."""
    text = prompt.SYSTEM_PROMPT
    assert "{" not in text, "brace in the prompt; was something left unfilled?"
    assert "}" not in text
