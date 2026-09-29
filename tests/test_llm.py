"""Tests for the provider boundary in llm.py.

These never touch the network. The request building and response parsing
are pure functions of their inputs, so they can be checked against canned
payloads, which keeps the suite fast and usable offline.
"""

import pytest

from agent import config, llm

def make_client(**kwargs) -> llm.OpenAICompatLLM:
    """Return an OpenAI-compatible client with test-friendly settings."""
    kwargs.setdefault("model", "test-model")
    kwargs.setdefault("base_url", "http://localhost:11434/v1")
    kwargs.setdefault("api_key", "test-key")
    return llm.OpenAICompatLLM(**kwargs)


# --- request building -----------------------------------------------------


def test_user_turn_becomes_a_user_message() -> None:
    messages = make_client()._build_messages([llm.Message(role="user", text="hi")])
    assert messages == [{"role": "user", "content": "hi"}]


def test_model_tool_call_arguments_are_a_json_string() -> None:
    """OpenAI encodes arguments as a JSON string, not an object.

    This is the single biggest difference from Gemini's shape, and getting
    it wrong produces a validation error from the API.
    """
    history = [
        llm.Message(
            role="model",
            tool_calls=(llm.ToolCall(id="c1", name="read_file", args={"path": "a.py"}),),
        )
    ]
    message = make_client()._build_messages(history)[0]
    assert message["tool_calls"][0]["function"]["name"] == "read_file"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"path": "a.py"}'
    # It must be a string on the wire, so json.loads is what comes back.
    import json

    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}


def test_tool_result_carries_the_call_id_it_answers() -> None:
    """Without `tool_call_id` the model cannot pair result with request."""
    history = [
        llm.Message(
            role="tool",
            results=(llm.ToolResult(call_id="c1", name="read_file", content="body"),),
        )
    ]
    message = make_client()._build_messages(history)[0]
    assert message == {"role": "tool", "tool_call_id": "c1", "content": "body"}


def test_payload_wraps_schemas_as_openai_tools() -> None:
    schemas = [
        {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    payload = make_client()._build_payload([llm.Message(role="user", text="x")], schemas)
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "read_file"


def test_payload_omits_tools_when_none_given() -> None:
    payload = make_client()._build_payload([llm.Message(role="user", text="x")], None)
    assert "tools" not in payload


def test_payload_requests_the_configured_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CONTEXT_TOKENS", 4096)
    payload = make_client()._build_payload([llm.Message(role="user", text="x")], None)
    assert payload["options"]["num_ctx"] == 4096


def test_thinking_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hybrid models waste seconds per turn reasoning before each call."""
    monkeypatch.setattr(config, "THINKING_ENABLED", False)
    payload = make_client()._build_payload([llm.Message(role="user", text="x")], None)
    assert payload["think"] is False


# --- response parsing -----------------------------------------------------


def test_parses_text_and_tool_calls() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": "reading it now",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "a.py"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    result = make_client()._read_response(data)
    assert result.text == "reading it now"
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].args == {"path": "a.py"}
    assert result.tool_calls[0].id == "call_1"


def test_survives_a_malformed_arguments_string() -> None:
    """A weak local model can emit broken JSON; that must not crash us."""
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "read_file", "arguments": "{not json"},
                        }
                    ],
                }
            }
        ]
    }
    result = make_client()._read_response(data)
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].args == {"_raw": "{not json"}


def test_arguments_that_decode_to_a_list_are_not_treated_as_args() -> None:
    """`json.loads` can succeed and still return the wrong type."""
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c",
                            "function": {"name": "read_file", "arguments": '["a.py"]'},
                        }
                    ],
                }
            }
        ]
    }
    result = make_client()._read_response(data)
    assert result.tool_calls[0].args == {"_raw": '["a.py"]'}


def test_missing_choices_yields_an_empty_reply() -> None:
    assert make_client()._read_response({}).text == ""


# --- reasoning traces -----------------------------------------------------


def test_thinking_trace_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`lfm2.5:8b` always emits a <think> block; the loop must not see it."""
    monkeypatch.setattr(config, "THINKING_ENABLED", False)
    data = {
        "choices": [
            {
                "message": {
                    "content": "<think>\nreasoning here\n</think>\nthe answer"
                }
            }
        ]
    }
    assert make_client()._read_response(data).text == "the answer"


def test_thinking_trace_is_kept_when_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENT_THINKING=1 exists to debug the model, so keep the trace."""
    monkeypatch.setattr(config, "THINKING_ENABLED", True)
    raw = "<think>\nreasoning\n</think>\nanswer"
    data = {"choices": [{"message": {"content": raw}}]}
    assert make_client()._read_response(data).text == raw


def test_answer_is_kept_when_stripping_would_empty_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing the entire reply is worse than showing a messy one."""
    monkeypatch.setattr(config, "THINKING_ENABLED", False)
    raw = "<think>\nonly reasoning, no answer"
    data = {"choices": [{"message": {"content": raw}}]}
    assert make_client()._read_response(data).text == raw


def test_text_without_a_think_block_is_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "THINKING_ENABLED", False)
    data = {"choices": [{"message": {"content": "just an answer"}}]}
    assert make_client()._read_response(data).text == "just an answer"


def test_multiline_thinking_block_is_fully_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The block is regex-based, so it must span newlines."""
    monkeypatch.setattr(config, "THINKING_ENABLED", False)
    raw = "<think>\nline one\nline two\n\nline three\n</think>\nanswer"
    data = {"choices": [{"message": {"content": raw}}]}
    assert make_client()._read_response(data).text == "answer"


def test_thinking_only_turn_with_a_tool_call_is_stripped_to_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model often thinks, calls a tool, and says nothing after.

    The text is then only commentary, so the trace must go even though
    that leaves an empty string. Falling back to the raw text here is
    what leaks reasoning into the transcript.
    """
    monkeypatch.setattr(config, "THINKING_ENABLED", False)
    data = {
        "choices": [
            {
                "message": {
                    "content": "<think>\nplanning the call\n</think>",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "list_files",
                                "arguments": '{"path": "."}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    result = make_client()._read_response(data)
    assert result.text == ""
    assert result.tool_calls[0].name == "list_files"


# --- rate limit handling --------------------------------------------------


def test_retry_delay_is_parsed_from_a_google_error() -> None:
    error = Exception("429 RESOURCE_EXHAUSTED ... 'retryDelay': '19.2s'")
    assert llm.parse_retry_delay(error) == pytest.approx(19.2)


def test_retry_delay_is_parsed_from_an_openai_error() -> None:
    error = Exception('HTTP 429: {"error":{"message":"slow down","retryDelay":"2s"}}')
    assert llm.parse_retry_delay(error) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "message",
    [
        "HTTP 400 malformed request",
        "HTTP 401 invalid api key",
        "HTTP 403 permission denied",
        "HTTP 404 model not found",
    ],
)
def test_permanent_errors_are_not_retried(message: str) -> None:
    """These fail the same way on attempt two, so retrying just wastes time."""
    assert llm.parse_retry_delay(Exception(message)) is None


def test_server_overload_is_retried() -> None:
    """Google calls a 503 'usually temporary'; the run should survive it."""
    delay = llm.parse_retry_delay(
        Exception("ServerError: 503 UNAVAILABLE. Spikes in demand are temporary.")
    )
    assert delay is not None


def test_bad_gateway_and_gateway_timeout_are_retried() -> None:
    assert llm.parse_retry_delay(Exception("HTTP 502 Bad Gateway")) is not None
    assert llm.parse_retry_delay(Exception("HTTP 504 Gateway Timeout")) is not None


def test_backoff_grows_with_each_attempt() -> None:
    """An unnamed 5xx gives no hint, so the wait has to widen itself."""
    error = Exception("HTTP 503 UNAVAILABLE")
    assert llm.parse_retry_delay(error, 0) < llm.parse_retry_delay(error, 1)
    assert llm.parse_retry_delay(error, 1) < llm.parse_retry_delay(error, 2)


def test_backoff_is_capped() -> None:
    """An unbounded backoff would stall a run indefinitely."""
    assert llm.parse_retry_delay(Exception("HTTP 503"), 50) == 30.0


def test_server_hint_beats_our_backoff() -> None:
    """When the API says how long to wait, that wins."""
    error = Exception("503 UNAVAILABLE retryDelay: 3s")
    assert llm.parse_retry_delay(error, 5) == pytest.approx(3.0)


def test_transient_error_eventually_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 503 case end to end: fail twice, succeed on the third try."""
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    monkeypatch.setattr(config, "MAX_RATE_LIMIT_RETRIES", 4)
    calls = {"n": 0}

    def overloaded() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("ServerError: 503 UNAVAILABLE")
        return "ok"

    assert llm.retry_on_rate_limit(overloaded) == "ok"
    assert calls["n"] == 3


def test_permanent_error_fails_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad key must not cost the user five minutes of waiting."""
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    monkeypatch.setattr(config, "MAX_RATE_LIMIT_RETRIES", 5)
    calls = {"n": 0}

    def bad_key() -> None:
        calls["n"] += 1
        raise RuntimeError("HTTP 400 INVALID_ARGUMENT: malformed request")

    with pytest.raises(RuntimeError):
        llm.retry_on_rate_limit(bad_key)
    assert calls["n"] == 1


def test_retry_on_rate_limit_returns_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    monkeypatch.setattr(config, "MAX_RATE_LIMIT_RETRIES", 3)
    assert llm.retry_on_rate_limit(lambda: "ok") == "ok"


def test_retry_on_rate_limit_eventually_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    monkeypatch.setattr(config, "MAX_RATE_LIMIT_RETRIES", 3)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("HTTP 429 retryDelay: 1s")
        return "ok"

    assert llm.retry_on_rate_limit(flaky) == "ok"
    assert calls["n"] == 3


def test_retry_on_rate_limit_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    monkeypatch.setattr(config, "MAX_RATE_LIMIT_RETRIES", 2)

    def always_limited() -> None:
        raise RuntimeError("HTTP 429 retryDelay: 1s")

    with pytest.raises(RuntimeError):
        llm.retry_on_rate_limit(always_limited)


def test_wait_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server asking for 10 minutes must not hang the run."""
    monkeypatch.setattr(config, "MAX_RATE_LIMIT_RETRIES", 1)
    monkeypatch.setattr(config, "MAX_RETRY_WAIT_SECONDS", 3)
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)

    def limited() -> str:
        if not slept:
            raise RuntimeError("HTTP 429 retryDelay: 600s")
        return "ok"

    assert llm.retry_on_rate_limit(limited) == "ok"
    assert slept == [3.0]


# --- provider selection ---------------------------------------------------


def test_build_llm_follows_the_configured_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "PROVIDER", "openai")
    assert isinstance(llm.build_llm(), llm.OpenAICompatLLM)

    monkeypatch.setattr(config, "PROVIDER", "gemini")
    assert isinstance(llm.build_llm(), llm.GeminiLLM)
