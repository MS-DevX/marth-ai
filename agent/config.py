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
# newer accounts, so it is not a safe default. This default was verified to
# serve a real request. Override it with AGENT_MODEL if you need another.
DEFAULT_MODEL = "gemini-3.5-flash"
MODEL_NAME = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)

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
