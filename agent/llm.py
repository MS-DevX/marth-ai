"""The LLM provider boundary.

Nothing outside this module imports `google.genai`. The rest of the project
talks to the model through the `LLM` protocol, so swapping Gemini for
another provider later means writing one new class here and changing one
line in `build_llm`.

Phase 1 scope: `send()` handles a single text message. Later phases widen
the return type to carry tool calls; the call site in loop.py will not
have to change structurally.
"""

from typing import TYPE_CHECKING, Protocol

from . import config

if TYPE_CHECKING:  # Imported for type checkers only, not at runtime.
    from google import genai


class LLM(Protocol):
    """The minimum interface the agent loop needs from a model."""

    def send(self, prompt: str) -> str:
        """Send one user message and return the model's text reply."""
        ...

    def list_model_names(self) -> list[str]:
        """Return the model names available to this API key."""
        ...


class GeminiLLM:
    """`LLM` implementation backed by the Gemini API (google-genai)."""

    def __init__(self, model: str | None = None) -> None:
        """Store the model name; the HTTP client is created on first use."""
        self._model = model or config.MODEL_NAME
        self._client: "genai.Client | None" = None

    @property
    def client(self) -> "genai.Client":
        """Return the SDK client, creating it on first access.

        Created lazily so that importing this module (for example from a
        test) does not require an API key or even the SDK itself.
        """
        if self._client is None:
            from google import genai  # imported here to keep startup cheap

            self._client = genai.Client(api_key=config.get_api_key())
        return self._client

    def send(self, prompt: str) -> str:
        """Send `prompt` to the model and return its reply as plain text.

        Args:
            prompt: The user message to send.

        Returns:
            The model's text reply, or an empty string if it returned none.
        """
        response = self.client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text or ""

    def list_model_names(self) -> list[str]:
        """Return the model names available to this API key, sorted.

        Names come back as `models/gemini-...`; the prefix is stripped so
        they can be passed straight back to `--model`.
        """
        models = self.client.models.list()
        return sorted(m.name.split("/")[-1] for m in models)


def build_llm(model: str | None = None) -> LLM:
    """Create the default LLM. Change this one line to swap providers."""
    return GeminiLLM(model=model)
