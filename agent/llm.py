"""The LLM provider boundary.

Nothing outside this module talks to a model. The rest of the project
goes through the `LLM` protocol, so the loop has no idea what is on the
other end of the socket.

One provider ships:

  * `OpenAICompatLLM` - anything speaking the OpenAI HTTP API. Ollama
                        locally is the intended target; Groq or OpenRouter
                        still work by repointing `OPENAI_BASE_URL`. Built
                        on `urllib` from the standard library, so the
                        agent ships with no SDK dependency at all.

Gemini was the other provider and has been removed. `google-genai` pulled
in pydantic, httpx, requests, websockets and cryptography, none of which
this agent used, and a local model costs nothing per call.

`OllamaModels` is also here, and is not a second provider: it does not
implement `LLM` and never sees a conversation. It is the model manager
`marth --setup` uses to answer "is this model installed yet" and "download
it if not", which is an HTTP call like any other and so has to live on
this side of the boundary.

The conversation is represented with the small dataclasses below rather
than any SDK's own message types. That is the whole point of this module:
the loop should not know or care who is on the other end.
"""

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from . import config


# --- Conversation model ---------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A tool the model has asked us to run."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The outcome of running a tool, on its way back to the model."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    """One turn in the conversation.

    A turn holds at most one thing, which keeps the translation to and from
    the wire format straightforward:

      * role="user"  + text          what the human typed
      * role="model" + tool_calls    the model asking for tools
      * role="model" + text          the model's final answer
      * role="tool"  + results       what the tools returned
    """

    role: str
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True)
class LLMResponse:
    """What the model sent back: some text, and possibly tool calls."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


# --- Interface ------------------------------------------------------------


class LLM(Protocol):
    """The minimum interface the agent loop needs from a model."""

    @property
    def model_name(self) -> str:
        """Return the model this instance will send to."""
        ...

    def send(
        self, history: list[Message], tool_schemas: list[dict] | None = None
    ) -> LLMResponse:
        """Send the conversation so far and return the model's reply.

        Args:
            history: The conversation so far, oldest first.
            tool_schemas: The tools the model may call, or None for no
                tools. The parameter exists on the protocol, not just on
                the implementation, because the loop passes it: a class
                written to the narrower signature would accept the call
                at runtime only by accident of a default value, and would
                silently never see a tool.
        """
        ...

    def list_model_names(self) -> list[str]:
        """Return the model names the server has."""
        ...


# --- Transient failures ---------------------------------------------------

# Failures worth waiting out. A 429 means we are going too fast, and a 5xx
# means the far end is briefly overloaded or restarting. Both usually clear
# on their own, so retrying turns a dead run into a slow one.
#
# Everything else is left out on purpose. A 400 bad request, a 401 bad key
# and a 404 unknown model fail identically on the second attempt, so
# retrying them just burns minutes before the same error appears.
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}

# gRPC-style names the SDK surfaces for the same conditions.
TRANSIENT_MARKERS = (
    "resource_exhausted",
    "unavailable",
    "overloaded",
    "deadline_exceeded",
    "internal",
)

# A daily quota is reported with the same 429 and the same `retryDelay` as a
# per-minute one, and the delay is misleading: it names when the *minute*
# resets, not the day. Retrying for five minutes against a limit that resets
# tomorrow cannot succeed, so it is treated as permanent.
PERMANENT_QUOTA_MARKERS = ("perday", "per_day", "per-day", "daily quota")

# Word-boundary match on the status code. A plain `"503" in body` would also
# fire on a model name or a retryDelay value, and `" 503 " in body` misses
# the common case of the code ending the message.
TRANSIENT_PATTERN = re.compile(
    r"\b(" + "|".join(str(code) for code in sorted(TRANSIENT_STATUSES)) + r")\b"
)


def parse_retry_delay(error: Exception, attempt: int = 0) -> float | None:
    """Return how long to wait before retrying, or None if we should not.

    Both providers report a rate limit as HTTP 429 and include a
    `retryDelay` such as `"19s"` somewhere in the body; when one is
    present it is used verbatim, because the server knows better than we
    do. A 5xx carries no such hint, so the wait doubles per attempt.

    Args:
        error: The exception raised by the provider.
        attempt: Zero-based retry count, used to size the backoff.

    Returns:
        Seconds to wait, or None if the error is not transient.
    """
    body = str(error).lower()
    if any(marker in body for marker in PERMANENT_QUOTA_MARKERS):
        return None
    if not TRANSIENT_PATTERN.search(body) and not any(
        marker in body for marker in TRANSIENT_MARKERS
    ):
        return None

    match = re.search(r"retrydelay[\"'\s:]+([0-9.]+)\s*s", body)
    if match:
        return float(match.group(1))
    # No hint given: back off gently rather than hammering immediately.
    return float(min(2**attempt * 2, 30))


def retry_on_rate_limit(send: Callable[[], Any]) -> Any:
    """Call `send`, waiting and retrying if the API is temporarily busy.

    Args:
        send: A zero-argument callable that performs one request.

    Returns:
        Whatever `send` returned.

    Raises:
        The last error, if it is permanent or we run out of retries.
    """
    for attempt in range(config.MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return send()
        except Exception as exc:  # noqa: BLE001 - inspected, then re-raised.
            delay = parse_retry_delay(exc, attempt)
            if delay is None or attempt == config.MAX_RATE_LIMIT_RETRIES:
                raise
            wait = min(delay, config.MAX_RETRY_WAIT_SECONDS)
            print(f"  [provider busy] waiting {wait:.0f}s, then retrying...")
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


# Some local models wrap their reasoning in <think>...</think> tags and
# emit it whether or not the API asks them not to. `lfm2.5:8b` has a
# template of literally `{{ .Prompt }}`, with no support for a thinking
# switch, so the trace comes back inside the message content. There is no
# flag that stops it; the only way to keep it out of the transcript is to
# take it back out.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str, has_tool_calls: bool = False) -> str:
    """Remove `<think>` traces from a model's reply.

    Args:
        text: Raw message content from the provider.
        has_tool_calls: True when the turn also requested tools.

    Returns:
        The text without reasoning blocks. If stripping would leave
        nothing behind *and* the model asked for no tools, the original
        is returned: losing a whole answer is worse than a messy one.
        When a tool call is present the text is only commentary, so it
        is stripped even if that empties it.
    """
    if config.THINKING_ENABLED or "<think>" not in text:
        return text
    stripped = THINK_BLOCK.sub("", text).strip()
    if has_tool_calls:
        return stripped
    return stripped or text


# --- OpenAI-compatible provider ------------------------------------------


class OpenAICompatLLM:
    """`LLM` implementation for any OpenAI-shaped HTTP API.

    Covers Ollama running locally and, by changing `OPENAI_BASE_URL`,
    cloud providers such as Groq or OpenRouter. Uses `urllib` from the
    standard library, so it adds no dependency.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model or config.MODEL_NAME
        self._base_url = (base_url or config.OPENAI_BASE_URL).rstrip("/")
        self._api_key = api_key or config.OPENAI_API_KEY

    @property
    def model_name(self) -> str:
        """Return the model this instance sends to."""
        return self._model

    def _build_messages(self, history: list[Message]) -> list[dict]:
        """Translate our conversation into OpenAI chat messages.

        Two details of the shape matter: tool arguments arrive as a JSON
        *string*, and a tool result has to quote the id of the call it
        answers.
        """
        messages: list[dict] = []
        for message in history:
            if message.role == "user":
                messages.append({"role": "user", "content": message.text})
            elif message.role == "model":
                entry: dict[str, Any] = {"role": "assistant", "content": message.text}
                if message.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                # A JSON string, not an object.
                                "arguments": json.dumps(call.args),
                            },
                        }
                        for call in message.tool_calls
                    ]
                messages.append(entry)
            elif message.role == "tool":
                for result in message.results:
                    messages.append(
                        {
                            "role": "tool",
                            # The id of the call being answered. Without it
                            # the model cannot pair result with request.
                            "tool_call_id": result.call_id,
                            "content": result.content,
                        }
                    )
        return messages

    def _build_payload(
        self, history: list[Message], tool_schemas: list[dict] | None
    ) -> dict:
        """Assemble the JSON body for one chat completion request."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._build_messages(history),
            "stream": False,
        }
        if tool_schemas:
            # Our schemas are already OpenAI-shaped apart from the wrapper.
            payload["tools"] = [
                {"type": "function", "function": schema} for schema in tool_schemas
            ]
        # Ollama-specific: switch off reasoning traces on hybrid models.
        payload["think"] = config.THINKING_ENABLED
        payload["options"] = {"num_ctx": config.CONTEXT_TOKENS}
        return payload

    def _post(self, payload: dict) -> dict:
        """POST the payload to /chat/completions and return parsed JSON."""
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=config.REQUEST_TIMEOUT_SECONDS
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Re-raise with the body attached: the useful detail (a
            # `retryDelay` hint, or why the model was refused) is in there.
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from provider: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach {self._base_url}: {exc.reason}. "
                f"Is the server running?"
            ) from exc

    def _read_response(self, data: dict) -> LLMResponse:
        """Pull text and tool calls out of a chat completion response."""
        choices = data.get("choices") or []
        if not choices:
            return LLMResponse(text="")
        message = choices[0].get("message") or {}

        calls: list[ToolCall] = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            # `arguments` is a JSON string and may be absent or malformed
            # on a weak model, so parse defensively.
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            if not isinstance(args, dict):
                args = {"_raw": raw_args}
            calls.append(
                ToolCall(
                    id=call.get("id") or f"{function.get('name', 'call')}_{index}",
                    name=function.get("name", ""),
                    args=args,
                )
            )
        return LLMResponse(
            text=strip_thinking(
                message.get("content") or "", has_tool_calls=bool(calls)
            ),
            tool_calls=tuple(calls),
        )

    def send(
        self,
        history: list[Message],
        tool_schemas: list[dict] | None = None,
    ) -> LLMResponse:
        """Send the conversation so far and return the model's reply.

        Args:
            history: Every turn so far, oldest first.
            tool_schemas: Tools the model may call.

        Returns:
            The model's text and any tool calls it requested.
        """
        payload = self._build_payload(history, tool_schemas)
        return self._read_response(retry_on_rate_limit(lambda: self._post(payload)))

    def list_model_names(self) -> list[str]:
        """Return the model names the provider offers, sorted."""
        request = urllib.request.Request(
            f"{self._base_url}/models",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=config.REQUEST_TIMEOUT_SECONDS
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise RuntimeError(f"Could not list models: {exc}") from exc
        return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))


class OllamaModels:
    """Asks a local Ollama server what it has, and downloads what it does not.

    Not an `LLM`. This never sees a prompt and implements no part of the
    protocol; it exists so that installing a model is a question with a
    checkable answer rather than a hopeful download. It is here because it
    makes HTTP calls, and this is the only module allowed to.

    Uses Ollama's own API rather than the OpenAI-shaped one, because the
    OpenAI surface can list models but cannot download them.
    """

    def __init__(self, base_url: str | None = None) -> None:
        """Point at a server.

        Args:
            base_url: The server root, e.g. http://localhost:11434. Defaults
                to the configured OpenAI-compatible base with its /v1
                stripped, so one variable configures both.
        """
        root = base_url or config.OPENAI_BASE_URL
        self._root = root[: -len("/v1")] if root.endswith("/v1") else root

    @property
    def root(self) -> str:
        """The server root this instance talks to."""
        return self._root

    def version(self) -> str:
        """Return the server's version, or raise if it is not reachable.

        Raises:
            RuntimeError: if nothing is listening. A missing server and a
                missing model look identical from outside, so this is what
                tells them apart.
        """
        try:
            with urllib.request.urlopen(
                f"{self._root}/api/version", timeout=config.REQUEST_TIMEOUT_SECONDS
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"No Ollama server answered at {self._root}. Start one with "
                f"`ollama serve`, or install Ollama first."
            ) from exc
        return str(data.get("version", "unknown"))

    def installed_models(self) -> list[str]:
        """Return the model names the server already holds, sorted.

        Raises:
            RuntimeError: if the server cannot be reached.
        """
        try:
            with urllib.request.urlopen(
                f"{self._root}/api/tags", timeout=config.REQUEST_TIMEOUT_SECONDS
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Could not list local models: {exc}") from exc
        return sorted(
            str(m.get("name", "")) for m in data.get("models", []) if m.get("name")
        )

    def has_model(self, name: str) -> bool:
        """Return True if the server already holds exactly `name`.

        Exact, and not a forgiving match on the part before the colon.
        Two shapes of "close enough" both end badly here:

          * `qwen2.5:0.5b` and `qwen2.5:1.5b` share a name and are
            entirely different models, so a match means skipping a
            gigabyte of download and then 404ing on every run.
          * asking for `granite4.1` says nothing about which tag the
            registry considers default, so matching any `granite4.1:*`
            can report a 3b model as satisfying a request for the 8b one.

        The cost of being strict is that `--model granite4.1` re-pulls a
        model that is already on disk. Ollama verifies cached layers
        rather than downloading them, so that costs a second, and it is a
        far better failure than the other direction.

        Args:
            name: The model to look for.
        """
        return name in self.installed_models()

    def pull(
        self, name: str, on_progress: Callable[["PullProgress"], None] | None = None
    ) -> None:
        """Download `name`, reporting progress as it goes.

        Ollama streams newline-delimited JSON, one object per state change,
        so the download can say what it is doing instead of sitting silent
        for several minutes.

        Args:
            name: The model to download.
            on_progress: Called with each update. Optional; without it the
                download is silent, which is what a test wants.

        Raises:
            RuntimeError: if the server reports an error, or cannot be
                reached.
        """
        request = urllib.request.Request(
            f"{self._root}/api/pull",
            data=json.dumps({"name": name, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # A model is gigabytes. The per-request timeout is for a single
        # HTTP exchange, and one chunk of a download is not slow; a stalled
        # one still fails rather than hanging the installer for ever.
        tally = _PullTally()
        try:
            with urllib.request.urlopen(
                request, timeout=config.PULL_TIMEOUT_SECONDS
            ) as response:
                for line in response:
                    update = self._read_progress(line, tally)
                    if update is not None and on_progress is not None:
                        on_progress(update)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Could not download {name}: {exc}") from exc

    def _read_progress(self, line: bytes, tally: "_PullTally") -> "PullProgress | None":
        """Turn one streamed line into an update, or None to skip it.

        Args:
            line: One raw newline-terminated JSON object from Ollama.
            tally: Carries the running byte counts between calls, because
                each line knows only about its own layer.
        """
        text = line.decode("utf-8").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return tally.update(data)


@dataclass
class PullProgress:
    """One update from a model download.

    A structured value rather than a ready-made string, because how this
    should look depends on where it is going: a terminal wants one line
    rewritten in place, a log file wants a line per change. Deciding that
    here would force one of those on the other.
    """

    status: str
    percent: int | None = None


class _PullTally:
    """Adds up a download that arrives one layer at a time.

    Ollama reports `total` and `completed` for whichever layer it is
    currently fetching, and never a size for the model as a whole. Taken
    literally, "45%" means 45% of one layer of nine, which reads as a
    download about halfway when it has barely started. So the layers are
    accumulated here instead.

    Layers already on disk are never mentioned at all, so the total
    under-counts them and the final figure is capped below 100 until
    Ollama says the pull succeeded. A bar that reaches 100 and then keeps
    going is worse than one that stops at 99 and finishes.
    """

    def __init__(self) -> None:
        self._done: dict[str, int] = {}
        self._sizes: dict[str, int] = {}

    def update(self, data: dict) -> PullProgress | None:
        """Fold one streamed object into the running total.

        Args:
            data: The decoded JSON object Ollama sent.
        """
        status = str(data.get("status", ""))
        digest = str(data.get("digest", ""))
        total = data.get("total")
        completed = data.get("completed")

        if digest and isinstance(total, int) and total > 0:
            self._sizes[digest] = total
        if digest and isinstance(completed, int):
            # Never let a re-sent chunk for a layer we already counted
            # inflate the numerator.
            self._done[digest] = max(self._done.get(digest, 0), completed)

        if status == "success":
            return PullProgress(status="success", percent=100)
        if not status:
            return None
        # Ollama's own status already names the layer it is on
        # ("pulling 183715c43589"), so there is nothing to add to it.
        return PullProgress(status=status, percent=self.percent())

    def percent(self) -> int | None:
        """Return the whole-model percentage, or None if still unknown."""
        grand_total = sum(self._sizes.values())
        if grand_total <= 0:
            return None
        done = sum(min(got, self._sizes.get(key, got)) for key, got in self._done.items())
        return min(99, int(done * 100 / grand_total))


def build_llm(model: str | None = None) -> LLM:
    """Create the LLM the agent talks to.

    There is one provider now. It used to be a choice, and the choice was
    a silent trap: an unrecognised `AGENT_PROVIDER` fell through to the
    Gemini branch and produced a 404 about a *model*, which reads like a
    bug in the agent rather than a typo in a setting. One backend cannot
    be configured wrong.

    Args:
        model: The model to use, or None for the configured default.

    Returns:
        An `LLM` pointed at `config.OPENAI_BASE_URL`, which is a local
        Ollama server unless someone deliberately repointed it.
    """
    return OpenAICompatLLM(model=model)
