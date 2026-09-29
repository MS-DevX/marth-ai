"""Central configuration for the agent.

Every tunable value lives here so that no other module has to hardcode a
model name, a timeout, or a limit. Anything here can be overridden with an
environment variable, which keeps experiments (different model, smaller
step budget) to a one-word change.
"""

import os
from pathlib import Path

# --- Model ---------------------------------------------------------------
# The ONLY place a model name appears in the project.
#
# Note: `gemini-2.5-flash` is listed by the models API but is rejected for
# newer accounts, so it is not a safe default. The Gemini default below was
# verified to serve a real request.
#
# Which provider to talk to: "gemini" or "openai" (Ollama, Groq, ...).
# Gemini stays the default so nothing changes until you opt in.
PROVIDER = os.environ.get("AGENT_PROVIDER", "gemini").strip().lower()

# Per-provider defaults, so a single AGENT_MODEL override works for both.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",
    "openai": "lfm2.5:8b",
}
DEFAULT_MODEL = DEFAULT_MODELS.get(PROVIDER, DEFAULT_MODELS["gemini"])
MODEL_NAME = os.environ.get("AGENT_MODEL", "").strip() or DEFAULT_MODEL

# --- OpenAI-compatible providers -----------------------------------------
# Ollama, Groq, and OpenRouter all expose the same OpenAI-shaped HTTP API,
# so one class in llm.py covers all three.
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL", "http://localhost:11434/v1"
).rstrip("/")
# Local servers ignore this, but cloud providers require it.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")

# Context window requested from the model. Ollama's own default is 4096,
# which is tight once a tool returns a whole file.
CONTEXT_TOKENS = int(os.environ.get("AGENT_CONTEXT_TOKENS", "8192"))

# Hybrid "thinking" models spend seconds per turn emitting reasoning tokens
# before every tool call. Tool-calling turns do not benefit from that, so
# it is off by default. Set to 1 to see the reasoning.
THINKING_ENABLED = os.environ.get("AGENT_THINKING", "0") == "1"

# Cloud APIs answer in seconds; a local model prefilling a large file on
# CPU can take much longer, so the ceiling has to be generous.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("AGENT_TIMEOUT", "300"))

# How many times to retry a request that hit a rate limit, and the longest
# single wait. The API tells us how long to wait via `retryDelay`.
MAX_RATE_LIMIT_RETRIES = int(os.environ.get("AGENT_MAX_RETRIES", "5"))
MAX_RETRY_WAIT_SECONDS = float(os.environ.get("AGENT_MAX_RETRY_WAIT", "60"))

# --- Confirmations --------------------------------------------------------
# Off by default. When on, write_file, edit_file and run_command run
# without asking. The destructive-command blocklist still applies, because
# that is a refusal rather than a confirmation, but everything else is
# trusted. Only sensible in a throwaway checkout or a container.
AUTO_APPROVE = os.environ.get("AGENT_YES", "0") == "1"

# The project directory, used to find .env reliably no matter where the
# agent is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Limits --------------------------------------------------------------
# Hard cap on loop iterations, so a confused model cannot burn through the
# API budget forever.
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "20"))

# Seconds a shell command may run before it is killed.
COMMAND_TIMEOUT_SECONDS = float(os.environ.get("AGENT_COMMAND_TIMEOUT", "30"))

# Tool output longer than this is truncated before it is sent back to the
# model (truncation keeps the conversation small and cheap).
MAX_OUTPUT_CHARS = int(os.environ.get("AGENT_MAX_OUTPUT_CHARS", "4000"))

# read_file refuses anything bigger than this rather than truncating,
# because silently returning half a file can mislead the model.
MAX_FILE_BYTES = int(os.environ.get("AGENT_MAX_FILE_BYTES", "512000"))

# Ceiling on the whole conversation sent to the model, in characters.
# Every step adds a tool result, so without this a 20-step run asks for
# ~20000 tokens and overflows a local model's 8K window around step 6.
# Gemini's context is large enough not to care, which is exactly why this
# needs testing against the local provider and not just the cloud one.
#
# Roughly 4 characters per token, so 24000 leaves a 6K budget inside an
# 8192 window once the tool schemas and the reply are accounted for.
MAX_HISTORY_CHARS = int(os.environ.get("AGENT_MAX_HISTORY_CHARS", "24000"))

# A model that asks for the same tool with the same arguments more than
# this many times is not making progress, and on a CPU that is minutes
# wasted per repetition. The run stops instead.
MAX_REPEATED_CALLS = int(os.environ.get("AGENT_MAX_REPEATED_CALLS", "3"))

# --- Sandbox -------------------------------------------------------------
# The one directory the agent is allowed to read and write: the project
# directory itself. Point the agent at some other repo with
# AGENT_WORKSPACE=/path/to/repo.
#
# Note: this is read once at import time. Code that needs to test against a
# different root should pass the root in as an argument instead of relying
# on this value.
WORKSPACE_ROOT = Path(
    os.environ.get("AGENT_WORKSPACE") or PROJECT_ROOT
).expanduser().resolve()

# The placeholder value shipped in .env.example. If we see it, the user
# copied the example file but never filled in a real key.
_PLACEHOLDER_KEY = "your_key_here"


def get_api_key() -> str:
    """Return the Gemini API key from the environment.

    The key is only ever read here and handed straight to the SDK. It is
    never logged or printed.

    Raises:
        RuntimeError: if the key is missing or still the placeholder value.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or key == _PLACEHOLDER_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env, put your "
            "key in it, and try again."
        )
    return key
