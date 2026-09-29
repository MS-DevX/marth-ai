# marth-ai

A small terminal coding agent, written from scratch so the loop is visible.

You type a task, it calls an LLM, the LLM asks for tools (read a file, run
a command), the agent runs them and feeds the results back, repeating until
the task is done. No agent frameworks — just the Gemini SDK and the standard
library.

## Status

Phase 1 (scaffold) is done. `main.py` sends **one** message to Gemini and
prints the reply. Tools, the agent loop, and the safety layer come in
later phases — see [Roadmap](#roadmap).

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
```

Run these from the `marth-ai` directory (or set `PYTHONPATH` to it). Every
run prints the workspace root and model it is using, so the sandbox scope is
always visible.

## Configuration

All settings live in `agent/config.py` and can be overridden with
environment variables (see `.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_MODEL` | `gemini-3.5-flash` | Model name |
| `AGENT_MAX_STEPS` | `20` | Max loop iterations |
| `AGENT_COMMAND_TIMEOUT` | `30` | Seconds before a shell command is killed |
| `AGENT_MAX_OUTPUT_CHARS` | `4000` | Tool output truncation limit |
| `AGENT_WORKSPACE` | the project dir | The only directory the agent may touch |

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
- [ ] Phase 2 — `read_file` + `list_files` and one tool round trip
- [ ] Phase 3 — the full agent loop
- [ ] Phase 4 — `write_file`, `edit_file`, `grep`, `run_command` + safety
- [ ] Phase 5 — tests, system prompt, usage docs
