"""Central configuration for the agent.

Every tunable value lives here so that no other module has to hardcode a
model name, a timeout, or a limit. Anything here can be overridden with an
environment variable, which keeps experiments (different model, smaller
step budget) to a one-word change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Where this copy of the agent lives ------------------------------------
# The directory holding the agent/ package.
#
# For a checkout that is the project directory. For a pipx install it is
# site-packages inside a throwaway venv, which is why it cannot be used
# on its own to decide anything: see is_checkout below.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where per-user settings live. Override with AGENT_CONFIG_DIR, which is
# what the tests use so they cannot write to a real home directory.
CONFIG_DIR = Path(
    os.environ.get("AGENT_CONFIG_DIR") or Path.home() / ".config" / "marth-ai"
).expanduser()


def is_checkout(project_root: Path) -> bool:
    """Return True if `project_root` is a source checkout.

    A checkout has a pyproject.toml sitting next to the agent package. An
    installed package has none, because the file that built it was not
    shipped - only the package it produced was.

    This is the single fact that separates the two cases, and both
    `env_file_for` and `default_workspace` depend on it. Getting it
    wrong is silent and bad: a workspace root of site-packages sandboxes
    the agent inside a venv it cannot see anything else in, and every run
    writes its history there.

    Args:
        project_root: The directory holding the agent package.
    """
    return (project_root / "pyproject.toml").is_file()


def env_file_for(checkout: bool, project_root: Path, config_dir: Path) -> Path:
    """Return the settings file to read.

    In a checkout that is the project's own .env, so the documented
    workflow does not change. Installed, there is no project to speak
    of, so it is the per-user file that `marth setup` writes.

    Args:
        checkout: The answer from `is_checkout`.
        project_root: The directory holding the agent package.
        config_dir: The per-user settings directory.
    """
    return project_root / ".env" if checkout else config_dir / ".env"


def default_workspace(checkout: bool, project_root: Path, cwd: Path) -> Path:
    """Return the workspace root to use when AGENT_WORKSPACE is unset.

    In a checkout, the project itself: that is where the code being
    worked on lives, and it is what every existing command assumes.

    Installed, the directory the user is standing in. The alternative is
    project_root, which in an install is site-packages - a sandbox that
    contains the agent's own dependencies and none of the user's code.

    Args:
        checkout: The answer from `is_checkout`.
        project_root: The directory holding the agent package.
        cwd: The working directory the agent was launched from.
    """
    return project_root if checkout else cwd


_IS_CHECKOUT = is_checkout(PROJECT_ROOT)

# --- Settings file --------------------------------------------------------
# Loaded here, before anything below, because every value in this file
# can be set in it and reading one before the load would quietly ignore
# the setting. That is not hypothetical: PROVIDER is read a few lines
# down, so an .env naming AGENT_PROVIDER has to be loaded above it and
# not in main().
#
# A real environment variable always beats the file, which is what
# load_dotenv does by default and is what lets a one-off override work
# without editing anything.
ENV_FILE = env_file_for(_IS_CHECKOUT, PROJECT_ROOT, CONFIG_DIR)
if ENV_FILE.is_file():
    load_dotenv(ENV_FILE)

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

# The model `marth setup` installs. Separate from DEFAULT_MODEL above
# because the two answer different questions: that is what this run uses,
# this is what gets put on the machine so there is something to use.
#
# granite4.1 over lfm2.5 because it was the one verified to drive the
# write tools through a real edit-and-command task on this project. An
# installer that quietly handed over a model which cannot edit anything
# would be a worse failure than no installer.
RECOMMENDED_LOCAL_MODEL = "granite4.1:8b"

# How long one chunk of a model download may stall before giving up. The
# per-request timeout above is for a single API exchange; a multi-
# gigabyte pull needs longer, or a slow mirror looks like a hang.
PULL_TIMEOUT_SECONDS = float(os.environ.get("AGENT_PULL_TIMEOUT", "120"))

# How patiently setup waits for an Ollama server it just started. Ten
# seconds is generous for a local process binding a port, and the loop is
# bounded so a machine where the server dies on startup reports that
# rather than hanging.
SETUP_STARTUP_ATTEMPTS = int(os.environ.get("AGENT_SETUP_ATTEMPTS", "20"))
SETUP_STARTUP_POLL_SECONDS = float(os.environ.get("AGENT_SETUP_POLL", "0.5"))

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
# PROJECT_ROOT is defined near the top of this file, because the .env it
# points at has to be loaded before any setting is read.

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
# The one directory the agent is allowed to read and write. Point the
# agent at some other repo with AGENT_WORKSPACE=/path/to/repo.
#
# The default depends on how the agent was installed, which is why this
# is not simply PROJECT_ROOT. In a checkout that is the project itself.
# In a pipx install PROJECT_ROOT is site-packages, and a sandbox rooted
# there would point the agent at its own dependencies and nowhere else,
# so the default becomes the directory the user is standing in - which is
# what "run this on my code" means when they have not said which code.
#
# Note: this is read once at import time. Code that needs to test against a
# different root should pass the root in as an argument instead of relying
# on this value.
DEFAULT_WORKSPACE = default_workspace(_IS_CHECKOUT, PROJECT_ROOT, Path.cwd())
WORKSPACE_ROOT = Path(
    os.environ.get("AGENT_WORKSPACE") or DEFAULT_WORKSPACE
).expanduser().resolve()

# --- Run history ----------------------------------------------------------
# Where past runs are written, relative to the workspace root. Inside the
# workspace on purpose: it sits next to the code it describes, one
# directory holds everything, and deleting it loses nothing but history.
#
# Not subject to the path sandbox, since the sandbox is for the agent's
# file tools and this is the agent writing about itself.
HISTORY_DIRNAME = ".marth-ai"

# Set AGENT_NO_HISTORY=1 to stop recording runs. Useful for a one-off
# command on someone else's repository, where leaving a directory behind
# would be rude.
HISTORY_ENABLED = os.environ.get("AGENT_NO_HISTORY", "0") != "1"

# How many past runs to keep. Each file holds one run, so trimming means
# deleting the oldest files by name; the name starts with a timestamp,
# which sorts chronologically without reading any of them.
MAX_HISTORY_RUNS = int(os.environ.get("AGENT_MAX_HISTORY_RUNS", "200"))

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
