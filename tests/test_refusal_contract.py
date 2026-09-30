"""Every failure route records itself, and nothing else does.

This file exists because of a bug that had nothing to do with how the code
looked. A tool that refused to act used to say so in prose, and the loop then
guessed the status back out of that prose by looking for particular words.
Two real failures came out of it:

  * a write outside the workspace was recorded as `ok` - the sandbox did
    refuse, the loop just did not notice, and the history file listed a
    refused write under "Changed:";
  * a file the model read that happened to contain the words "declined by
    the user" was recorded as a refusal, because the words were somewhere
    in the text rather than at the start of it.

The status is now reported by the tool that knows it, and travels as
`tools.Result.status`. The tests below are about that holding on every
route a call can fail by, because a route nobody checks is a route that
silently records itself as a success.

They drive `loop.run_tool_call` rather than the tools directly, because
that is what the loop does and it is where the sandbox, the argument
check and the crash handler sit. Testing `tools.write_file` on its own
would miss all three.
"""

import pytest

from agent import config, loop, safety, tools


@pytest.fixture
def auto_yes(project, monkeypatch: pytest.MonkeyPatch) -> None:
    """Approve every confirmation, so these tests can reach the failures."""
    monkeypatch.setattr(config, "AUTO_APPROVE", True)


@pytest.fixture
def answering(project, monkeypatch: pytest.MonkeyPatch):
    """Return a factory setting whether the next prompt is answered yes."""

    def _set(yes: bool) -> None:
        monkeypatch.setattr(config, "AUTO_APPROVE", False)
        monkeypatch.setattr(safety, "_ASK_HANDLER", lambda _q: yes)

    return _set


# --- the blocklist ---------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    ["rm -rf /", "rm -rf ~", "rm -rf *", "mkfs.ext4 /dev/sda", "shutdown -h now"],
)
def test_a_blocked_command_is_recorded_as_blocked_not_declined(
    auto_yes, command: str
) -> None:
    """Not "declined" and not "ok": the blocklist is a refusal, not a question.

    A blocked command was never put to the user, so calling it "declined"
    would say a person said no. And `--yes` skips confirmations, never the
    blocklist, so it must not change the outcome here.
    """
    result = loop.run_tool_call("run_command", {"command": command})
    assert result.status == "blocked"
    assert result.approved is None, "nobody was asked, so nobody declined"


def test_a_blocked_command_still_explains_itself_to_the_model(
    auto_yes,
) -> None:
    """The status is for the log; the model still needs the reason."""
    result = loop.run_tool_call("run_command", {"command": "rm -rf /"})
    assert "blocked" in result.lower()
    assert result.startswith("Refused:")


# --- the sandbox -----------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["../escaped.py", "/etc/passwd", "subdir/../../escaped.py"]
)
def test_a_write_outside_the_workspace_is_an_error(auto_yes, path: str) -> None:
    result = loop.run_tool_call("write_file", {"path": path, "content": "x = 1"})
    assert result.status == "error", f"refusal recorded as success: {result}"


@pytest.mark.parametrize("path", ["../escaped.py", "/etc/passwd"])
def test_a_read_outside_the_workspace_is_an_error(auto_yes, path: str) -> None:
    result = loop.run_tool_call("read_file", {"path": path})
    assert result.status == "error", f"refusal recorded as success: {result}"


def test_a_secret_file_is_an_error_not_a_read(auto_yes) -> None:
    """Refused by name, before any content is read."""
    (config.WORKSPACE_ROOT / ".env").write_text("KEY=secret\n")
    result = loop.run_tool_call("read_file", {"path": ".env"})
    assert result.status == "error"
    assert "secret" in result.lower()


def test_a_write_to_a_secret_file_is_an_error(auto_yes) -> None:
    (config.WORKSPACE_ROOT / ".env").write_text("KEY=old\n")
    result = loop.run_tool_call("write_file", {"path": ".env", "content": "KEY=new"})
    assert result.status == "error"
    assert "KEY=old" in (config.WORKSPACE_ROOT / ".env").read_text(), "wrote anyway"


# --- a decline -------------------------------------------------------------


def test_a_declined_write_is_recorded_as_declined(answering) -> None:
    answering(False)
    result = loop.run_tool_call("write_file", {"path": "new.py", "content": "x = 1"})
    assert result.status == "declined"
    assert result.approved is False


def test_a_declined_edit_is_recorded_as_declined(answering) -> None:
    answering(False)
    (config.WORKSPACE_ROOT / "a.py").write_text("x = 1\n")
    result = loop.run_tool_call("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"})
    assert result.status == "declined"


def test_a_declined_command_is_recorded_as_declined(answering) -> None:
    """A harmless command, so only the prompt can be the reason it stopped."""
    answering(False)
    result = loop.run_tool_call("run_command", {"command": "echo hello"})
    assert result.status == "declined"
    assert result.approved is False


def test_an_approved_write_is_recorded_as_a_success(answering) -> None:
    """The other direction: a success must not look like a refusal."""
    answering(True)
    result = loop.run_tool_call("write_file", {"path": "new.py", "content": "x = 1"})
    assert result.status == "ok"
    assert result.approved is True
    assert (config.WORKSPACE_ROOT / "new.py").exists()


def test_an_approved_edit_is_recorded_as_a_success(answering) -> None:
    answering(True)
    (config.WORKSPACE_ROOT / "a.py").write_text("x = 1\n")
    result = loop.run_tool_call("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"})
    assert result.status == "ok"
    assert result.approved is True


# --- approvals that were never asked for ----------------------------------


@pytest.mark.parametrize(
    "name, args",
    [
        ("read_file", {"path": "a.py"}),
        ("list_files", {"path": "."}),
        ("grep", {"pattern": "x", "path": "."}),
    ],
)
def test_a_read_only_tool_records_that_nobody_was_asked(
    auto_yes, name: str, args: dict
) -> None:
    """`None` means nobody was asked, which is not the same as a yes.

    Three states, and the history has to tell them apart: approved,
    declined, and not applicable.
    """
    (config.WORKSPACE_ROOT / "a.py").write_text("x = 1\n")
    result = loop.run_tool_call(name, args)
    assert result.approved is None
    assert result.status == "ok"


def test_an_approved_command_records_the_approval(answering) -> None:
    answering(True)
    result = loop.run_tool_call("run_command", {"command": "echo hi"})
    assert result.status == "ok"
    assert result.approved is True


# --- arguments and tools that do not exist --------------------------------


def test_an_unknown_tool_is_an_error() -> None:
    result = loop.run_tool_call("no_such_tool", {})
    assert result.status == "error"
    assert "Available tools" in result, "the model needs to know what it can use"


def test_a_missing_argument_is_an_error() -> None:
    result = loop.run_tool_call("read_file", {})
    assert result.status == "error"
    assert "Bad arguments" in result


def test_an_unexpected_argument_is_an_error() -> None:
    result = loop.run_tool_call("read_file", {"path": "a.py", "colour": "blue"})
    assert result.status == "error"


def test_a_tool_that_crashes_is_an_error_and_not_a_crash(project) -> None:
    """The run has to survive it: the model should be told, then continue."""
    monkey = tools.TOOL_REGISTRY["read_file"]

    def explode(*_a, **_k):
        raise RuntimeError("kaboom")

    tools.TOOL_REGISTRY["read_file"] = explode
    try:
        result = loop.run_tool_call("read_file", {"path": "a.py"})
    finally:
        tools.TOOL_REGISTRY["read_file"] = monkey
    assert result.status == "error"
    assert "kaboom" in result, "the model cannot act on a bare exception name"


# --- tool-level failures ---------------------------------------------------


@pytest.mark.parametrize(
    "name, args, why",
    [
        ("list_files", {"path": "nope"}, "not a directory"),
        ("read_file", {"path": "nope.py"}, "not a file"),
        ("grep", {"pattern": "x", "path": "nope"}, "no such file"),
        ("edit_file", {"path": "nope.py", "old": "a", "new": "b"}, "no such file"),
    ],
)
def test_a_tool_that_cannot_do_its_job_says_so(
    auto_yes, name: str, args: dict, why: str
) -> None:
    """These return a message rather than raising, so nothing else marks them."""
    result = loop.run_tool_call(name, args)
    assert result.status == "error", f"{why} recorded as a success: {result}"


def test_writing_through_a_directory_is_an_error(auto_yes) -> None:
    """The directory has to exist first.

    `write_file("sub/x")` creates `sub/` on the way, so a path like this is
    a perfectly good write unless the directory is already there - and then
    the tool has to notice and say so.
    """
    (config.WORKSPACE_ROOT / "sub").mkdir()
    result = loop.run_tool_call("write_file", {"path": "sub", "content": "x"})
    assert result.status == "error", f"recorded as a success: {result}"


def test_an_edit_with_no_match_is_an_error(auto_yes) -> None:
    """The model has to know its guess at the file was wrong."""
    (config.WORKSPACE_ROOT / "a.py").write_text("x = 1\n")
    result = loop.run_tool_call("edit_file", {"path": "a.py", "old": "nope", "new": "y"})
    assert result.status == "error"
    assert "No match" in result


def test_an_edit_matching_several_places_is_an_error(auto_yes) -> None:
    (config.WORKSPACE_ROOT / "a.py").write_text("x = 1\nx = 1\n")
    result = loop.run_tool_call("edit_file", {"path": "a.py", "old": "x = 1", "new": "y"})
    assert result.status == "error"
    assert "twice" in result or "2 times" in result


def test_a_command_that_times_out_is_an_error(auto_yes, monkeypatch) -> None:
    """A timeout is the one command result that is a tool failure.

    A non-zero exit is not, and this is the difference: `pytest` failing
    is the answer the model asked for, whereas a command killed at 30
    seconds never produced one.
    """
    monkeypatch.setattr(config, "COMMAND_TIMEOUT_SECONDS", 0.05)
    result = loop.run_tool_call("run_command", {"command": "sleep 5"})
    assert result.status == "error"
    assert "TIMED OUT" in result


def test_a_command_with_a_non_zero_exit_is_not_an_error(auto_yes) -> None:
    """Otherwise a failing test run would be indistinguishable from a broken tool."""
    result = loop.run_tool_call("run_command", {"command": "exit 1"})
    assert result.status == "ok"
    assert "exit code: 1" in result


def test_a_search_that_found_nothing_is_not_an_error(auto_yes) -> None:
    """An empty result is an answer. Colouring it red would cry wolf."""
    result = loop.run_tool_call("grep", {"pattern": "zzz-not-here", "path": "."})
    assert result.status == "ok"


# --- the bug this file was written for ------------------------------------


def test_a_file_containing_the_refusal_words_is_still_a_successful_read(
    auto_yes,
) -> None:
    """The reason the status stopped being read back out of the text.

    Under the old scheme the words were matched anywhere in the result, so
    this read was recorded as a decline. Anything the model reads can
    contain any text at all - including this repository's own source.
    """
    (config.WORKSPACE_ROOT / "note.txt").write_text("declined by the user\n")
    result = loop.run_tool_call("read_file", {"path": "note.txt"})
    assert result.status == "ok", f"a file's contents decided the status: {result}"


def test_a_file_containing_the_blocked_words_is_still_a_successful_read(
    auto_yes,
) -> None:
    (config.WORKSPACE_ROOT / "note.txt").write_text(
        "Refused: this command is blocked because it matches.\n"
    )
    result = loop.run_tool_call("read_file", {"path": "note.txt"})
    assert result.status == "ok"


def test_a_file_containing_the_words_failed_is_still_a_successful_read(
    auto_yes,
) -> None:
    (config.WORKSPACE_ROOT / "note.txt").write_text("Failed: everything\n")
    result = loop.run_tool_call("read_file", {"path": "note.txt"})
    assert result.status == "ok"


def test_a_command_whose_output_looks_like_a_refusal_is_not_a_refusal(
    auto_yes,
) -> None:
    """The other direction: a command that really ran must not read as blocked.

    `echo` cannot be a blocked command, because it already ran.
    """
    result = loop.run_tool_call("run_command", {"command": "echo 'refused: this command is blocked'"})
    assert result.status == "ok", f"a command's own output decided its status: {result}"


def test_a_source_file_of_this_project_is_not_a_status_of_its_own(auto_yes) -> None:
    """The source of the old rule is still in this repository.

    `tools.py` contains the refusal wording itself, so reading it used to
    depend on where in the file the phrase appeared.
    """
    import pathlib

    from agent import config as cfg

    project = pathlib.Path(__file__).resolve().parent.parent
    real = cfg.WORKSPACE_ROOT
    cfg.WORKSPACE_ROOT = project
    try:
        result = loop.run_tool_call("read_file", {"path": "agent/tools.py"})
    finally:
        cfg.WORKSPACE_ROOT = real
    assert result.status == "ok"


# --- nothing escapes the type ---------------------------------------------


def test_no_route_produces_a_status_outside_the_documented_set(
    project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever happens, the value has to be one the views can colour.

    The dashboard maps the status to a colour and the history file stores
    it, so an unexpected string would reach both. Enumerated over the
    routes that do not succeed, because those are the ones that matter.
    """
    (config.WORKSPACE_ROOT / "a.py").write_text("x = 1\n")
    (config.WORKSPACE_ROOT / ".env").write_text("K=v\n")
    seen: set[str] = set()

    def record(name: str, args: dict) -> None:
        seen.add(loop.run_tool_call(name, args).status)

    for approve in (True, False):
        monkeypatch.setattr(config, "AUTO_APPROVE", approve)
        monkeypatch.setattr(safety, "_ASK_HANDLER", lambda _q: approve)
        record("read_file", {"path": "a.py"})
        record("read_file", {"path": "/etc/passwd"})
        record("read_file", {"path": ".env"})
        record("list_files", {"path": "."})
        record("list_files", {"path": "nope"})
        record("grep", {"pattern": "x", "path": "."})
        record("grep", {"pattern": "x", "path": "nope"})
        record("write_file", {"path": "new.py", "content": "x = 1"})
        record("write_file", {"path": "../out.py", "content": "x"})
        record("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 2"})
        record("edit_file", {"path": "a.py", "old": "nope", "new": "y"})
        record("run_command", {"command": "echo hi"})
        record("run_command", {"command": "rm -rf /"})
        record("run_command", {"command": "sleep 5"})

    allowed = set(CallStatus_values())
    assert seen <= allowed, f"unexpected statuses: {seen - allowed}"


def test_all_four_statuses_are_reachable(
    project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `ok` were the only one ever produced, the tests above prove nothing."""
    monkeypatch.setattr(config, "AUTO_APPROVE", True)
    ok = loop.run_tool_call("read_file", {"path": "a.py"})
    (config.WORKSPACE_ROOT / "a.py").write_text("x = 1\n")
    ok = loop.run_tool_call("read_file", {"path": "a.py"})
    blocked = loop.run_tool_call("run_command", {"command": "rm -rf /"})
    error = loop.run_tool_call("read_file", {"path": "/etc/passwd"})
    monkeypatch.setattr(config, "AUTO_APPROVE", False)
    monkeypatch.setattr(safety, "_ASK_HANDLER", lambda _q: False)
    declined = loop.run_tool_call("write_file", {"path": "new.py", "content": "x"})

    assert {ok.status, blocked.status, error.status, declined.status} == {
        "ok",
        "blocked",
        "error",
        "declined",
    }


def CallStatus_values() -> list[str]:
    """Return the statuses `CallStatus` documents, read from the type itself.

    A Literal is only documentation unless something checks it, and this
    is what stops a fifth status being added without the views being
    taught to draw it.
    """
    from agent.runs import CallStatus

    return list(CallStatus.__args__)


# --- a `Result` behaves like the string the model sees --------------------


def test_a_result_is_the_text_the_model_sees(auto_yes) -> None:
    """Otherwise the model would be shown a Python repr of an object."""
    (config.WORKSPACE_ROOT / "a.py").write_text("x = 1\n")
    result = loop.run_tool_call("read_file", {"path": "a.py"})
    assert result == "x = 1\n"
    assert isinstance(result, str)
    assert "x = 1" in result
    assert f"{result}" == "x = 1\n"


# --- and the whole thing is recorded in the run ---------------------------


def test_a_run_records_a_refused_write_as_refused(
    project, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the status reaches the object the dashboard and history read.

    The tests above check the tool result. This checks that nothing
    between the tool and the record throws the status away, because a
    correct `Result` that never arrives is worth nothing.
    """
    from agent.llm import LLMResponse, ToolCall as LlmToolCall
    from agent.runs import Run
    from tests.test_loop import ScriptedLLM

    (config.WORKSPACE_ROOT / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(config, "AUTO_APPROVE", False)
    monkeypatch.setattr(safety, "_ASK_HANDLER", lambda _q: False)

    model = ScriptedLLM(
        [
            LLMResponse(
                text="",
                tool_calls=(
                    LlmToolCall(id="1", name="write_file", args={"path": "b.py", "content": "y"}),
                ),
            ),
            LLMResponse(text="I will not do that."),
        ]
    )
    record = Run(task="t", model="m", provider="test", started=0.0)
    run = loop.run_record("do it", model, record=record)
    assert run is record
    write = next(c for c in run.calls if c.name == "write_file")
    assert write.status == "declined"
    assert write.approved is False
    assert run.counts["declined"] == 1
    assert run.counts["ok"] == 0
