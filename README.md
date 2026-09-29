# marth-ai

A small terminal coding agent, written from scratch so the loop is visible.

You type a task, it calls an LLM, the LLM asks for tools (read a file, run
a command), the agent runs them and feeds the results back, repeating until
the task is done. No agent frameworks — just the Gemini SDK and the standard
library.

## Status

Phase 2 is done. The agent can read files and list directories: it sends
your task to the model, the model asks for a tool, the agent runs it, and
the result goes back for a final answer.

Currently available: `list_files`, `read_file`. Write tools, the full
loop, and confirmations come later — see [Roadmap](#roadmap).

## Setup

Python 3.11 or newer.

```bash
cd marth-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then edit .env and paste your key
```

Get a key from <https://aistudio.google.com/apikey> and put it in `.env`:

```
GEMINI_API_KEY=your_actual_key
```

`.env` is gitignored. The key is only ever read from the environment and
handed to the SDK — it is never printed.

## Usage

```bash
# One-shot task
python -m agent.main "explain what this repo does"

# Or get prompted for the task
python -m agent.main

# Use a different model
python -m agent.main --model gemini-3.1-flash-lite "hello"

# See which models your key can use
python -m agent.main --list-models

# Run entirely on a local model via Ollama (no API key, no rate limit)
AGENT_PROVIDER=openai python -m agent.main "explain what this repo does"
```

Run these from the `marth-ai` directory (or set `PYTHONPATH` to it). Every
run prints the workspace, provider, and model it is using, so the sandbox
scope and the target are always visible.

## Providers

The LLM call sits behind one small interface in `agent/llm.py`, so the
provider is a config value. Two implementations ship:

| `AGENT_PROVIDER` | Backend | Needs | Notes |
| --- | --- | --- | --- |
| `gemini` (default) | Gemini API | `GEMINI_API_KEY` | Free tier is rate limited |
| `openai` | Ollama, or any OpenAI-compatible API | nothing locally | No rate limit; slower on CPU |

### Running fully local with Ollama

Installed under `~/.local/opt/ollama` (no root needed). Start the server
once per session:

```bash
~/.local/opt/ollama/bin/ollama serve &
```

Then pull a model and point the agent at it:

```bash
~/.local/opt/ollama/bin/ollama pull lfm2.5:8b        # primary
~/.local/opt/ollama/bin/ollama pull granite4.1:8b    # alternative

AGENT_PROVIDER=openai python -m agent.main "list the files in agent/"
```

`lfm2.5:8b` is the default local model because it is built for tool
calling, which is all an agent does. `granite4.1:8b` is an Apache-2.0
alternative. Only one model is held in memory at a time.

Ollama's OpenAI-compatible endpoint is used, so the same code path serves
Groq or OpenRouter — just change `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

> On a CPU-only machine the model needs roughly its own file size free in
> RAM. If loading fails or the machine crawls, close a browser and retry.

## Configuration

All settings live in `agent/config.py` and can be overridden with
environment variables (see `.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_PROVIDER` | `gemini` | `gemini` or `openai` (Ollama/Groq/etc) |
| `AGENT_MODEL` | per provider | Model name |
| `AGENT_MAX_STEPS` | `20` | Max loop iterations |
| `AGENT_COMMAND_TIMEOUT` | `30` | Seconds before a shell command is killed |
| `AGENT_MAX_OUTPUT_CHARS` | `4000` | Tool output truncation limit |
| `AGENT_MAX_FILE_BYTES` | `512000` | `read_file` refuses anything larger |
| `AGENT_WORKSPACE` | the project dir | The only directory the agent may touch |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | `ollama` | Ignored locally; required by cloud providers |
| `AGENT_CONTEXT_TOKENS` | `8192` | Context window requested |
| `AGENT_THINKING` | `0` | Set to `1` to see hybrid-model reasoning traces |
| `AGENT_TIMEOUT` | `300` | Per-request timeout; local models are slow |
| `AGENT_MAX_RETRIES` | `5` | Rate-limit retries before giving up |
| `AGENT_MAX_RETRY_WAIT` | `60` | Longest single rate-limit wait |

The workspace root is the sandbox boundary. The agent may only read and
write files that resolve inside it. By default that is the `marth-ai/`
project directory, so the agent cannot wander into your home directory.

## Layout

```
marth-ai/
  AGENTS.md          rules for future sessions
  README.md          this file
  requirements.txt
  .env.example
  agent/
    main.py          CLI entry: read task, run loop
    loop.py          the agent loop
    llm.py           Gemini wrapper behind a swappable interface
    tools.py         tool functions + tool schemas
    safety.py        path sandbox + confirmation prompts
    config.py        model name, max steps, workspace root
  tests/
    test_tools.py
    test_safety.py
```

## Roadmap

- [x] Phase 1 — scaffold, config, one-shot Gemini call
- [x] Phase 2 — `read_file` + `list_files` and one tool round trip
- [ ] Phase 3 — the full agent loop
- [ ] Phase 4 — `write_file`, `edit_file`, `grep`, `run_command` + safety
- [ ] Phase 5 — tests, system prompt, usage docs

## Rate limits

The Gemini free tier allows only a handful of requests per minute, and each
loop step costs at least one. The agent now handles this itself: on a
`429` it reads the `retryDelay` the API returns, waits exactly that long,
and retries. Set `AGENT_MAX_RETRIES` to `0` to disable.

For unlimited testing, use the local provider instead.
