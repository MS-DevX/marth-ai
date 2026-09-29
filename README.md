# marth-ai

A small terminal coding agent, written from scratch so the loop is visible.

You type a task, it calls an LLM, the LLM asks for tools (read a file, run
a command), the agent runs them and feeds the results back, repeating until
the task is done. No agent frameworks — just the Gemini SDK and the standard
library.

## Status

Phase 3 is done. The agent runs a real loop: it sends your task, runs
whatever tools the model asks for, feeds the results back, and keeps going
until the model answers. It stops on its own when the model is done, when
the model starts repeating itself, or when it hits the step limit — and it
says which one happened.

Currently available: `list_files`, `read_file` (with line ranges). Write
tools and confirmations come in Phase 4 — see [Roadmap](#roadmap).

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
| `AGENT_MAX_HISTORY_CHARS` | `24000` | Ceiling on the whole conversation |
| `AGENT_MAX_REPEATED_CALLS` | `3` | Identical tool calls before giving up |
| `AGENT_COMMAND_TIMEOUT` | `30` | Seconds before a shell command is killed |
| `AGENT_MAX_OUTPUT_CHARS` | `4000` | Tool output truncation limit |
| `AGENT_MAX_FILE_BYTES` | `512000` | `read_file` refuses anything larger |
| `AGENT_WORKSPACE` | the project dir | The only directory the agent may touch |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | `ollama` | Ignored locally; required by cloud providers |
| `AGENT_CONTEXT_TOKENS` | `8192` | Context window requested |
| `AGENT_THINKING` | `0` | Set to `1` to see hybrid-model reasoning traces |
| `AGENT_TIMEOUT` | `300` | Per-request timeout; local models are slow |
| `AGENT_MAX_RETRIES` | `5` | Retries for a busy or overloaded provider |
| `AGENT_MAX_RETRY_WAIT` | `60` | Longest single retry wait |

The workspace root is the sandbox boundary. The agent may only read and
write files that resolve inside it. By default that is the `marth-ai/`
project directory, so the agent cannot wander into your home directory.

Being inside the workspace is not enough on its own. Secret files live in
there too, and the agent has no reason to see any of them:

| Blocked | Allowed |
| --- | --- |
| `.env`, `.ENV`, `credentials`, `secrets` | `.env.example`, `.env.sample` |
| `id_rsa`, `id_ed25519`, `.netrc`, `.npmrc` | any ordinary source file |
| `*.key`, `*.pem`, `*.p12`, `*.pfx`, `*.keystore` | |

The check runs on the resolved name, so `agent/../.env` is refused the
same as `.env`, and the committed templates stay readable because they
document the variables without holding any values.

This matters more than it looks. A file the agent reads can contain
instructions aimed at the model, and `.env` is the one file worth
stealing. Blocking on the filename means the answer does not depend on
the model choosing to refuse.

### Output limits

Every tool result is truncated to `AGENT_MAX_OUTPUT_CHARS` (4000) before it
reaches the model, at a single choke point in the loop so no tool can skip
it. Without this, reading one 20 KB source file fills most of an 8K context
window and the next turn has nowhere to go. Truncated results say how much
was dropped and suggest `grep` instead of re-reading.

## Layout

```
marth-ai/
  AGENTS.md          rules for future sessions
  README.md          this file
  requirements.txt
  .env.example
  agent/
    main.py          CLI entry: read task, run loop
    loop.py          the agent loop + the tool-output truncation choke point
    llm.py           provider boundary: Gemini + OpenAI-compatible
    tools.py         tool functions + tool schemas
    safety.py        path sandbox, secret blocking, truncation, confirmations
    config.py        model name, max steps, workspace root
  tests/
    test_tools.py
    test_safety.py
    test_llm.py
    test_loop.py
```

## Roadmap

- [x] Phase 1 — scaffold, config, one-shot Gemini call
- [x] Phase 2 — `read_file` + `list_files` and one tool round trip
- [x] Phase 3 — the full agent loop
- [ ] Phase 4 — `write_file`, `edit_file`, `grep`, `run_command` + confirmations
- [ ] Phase 5 — system prompt, usage docs, wider test coverage

## The loop

One step is: send the conversation, run whatever tools were asked for,
send again. It is written out longhand in `loop.py` rather than handed to
a framework, because the loop *is* the project.

```python
for step in range(1, config.MAX_STEPS + 1):
    response = llm.send(compact_history(history), tool_schemas=TOOL_SCHEMAS)
    if not response.tool_calls:
        return response.text          # finished
    history.append(model_turn)
    history.append(tool_turn)
```

It stops in three situations, and says which one happened, because "the
agent finished" and "the agent gave up" look identical otherwise:

| Stop | Cause |
| --- | --- |
| Answered | The model replied with prose and no tool calls |
| Repeating | The same call, with the same arguments, `MAX_REPEATED_CALLS` times |
| Step limit | `MAX_STEPS` reached without a final answer |

Repeat detection sorts the arguments, so a model cannot loop forever just
by reshuffling its keys. On a CPU that matters: each wasted step is real
minutes.

### Keeping the conversation small

The whole conversation is re-sent every step, so it has to stay bounded.
Twenty steps of capped tool output is about 20,000 tokens, which overflows
a local model's 8K window around step 6. Gemini's window is large enough
that it never notices, which is why this is developed against Ollama.

When the history exceeds `AGENT_MAX_HISTORY_CHARS`, the oldest **tool
results** are replaced with a one-line placeholder. Prose is never
dropped — the task and the model's own reasoning are what tie the steps
together — and the most recent result is always kept, since that is what
the model is currently thinking about.

### Reading part of a file

Truncation protects the context but leaves the model blind to the rest of
a file. Asked how many tests `test_llm.py` had, the local model counted
8 out of 32, because it only saw 28% of it and guessed the remainder.

`read_file` takes `start_line` and `end_line` for that reason:

```
read_file("agent/llm.py", 200, 260)   ->  [lines 200-260 of 512]
```

A range small enough to come back whole beats a whole file that gets cut.

## Rate limits

The Gemini free tier allows **20 requests per day** per model, and each
loop step costs at least one, so a real agent loop exhausts it in a
couple of minutes. The agent handles the failures itself rather than
dying. Two kinds are waited out:

- **429 per-minute rate limit** — the `retryDelay` the API returns is used
  verbatim, because the server knows better than we do how long to wait
- **5xx overloaded** — no hint is given, so the wait doubles each attempt,
  up to 30 seconds

Everything else fails immediately, with an explanation and a next step:

- **429 daily quota** — reported identically to a per-minute one, including
  a misleading `retryDelay`, but waiting cannot help. Recognised and
  failed fast rather than retried for five minutes against a limit that
  resets tomorrow.
- **400 / 401 / 403 / 404** — fail the same way on attempt two, so retrying
  just makes the user wait for the same error.

Set `AGENT_MAX_RETRIES` to `0` to disable retrying entirely.

A busy free tier also means a model can be listed and still be overloaded
at the moment you use it. If a specific model keeps returning 503, try
another: `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite` have been
reliable where `gemini-3.5-flash` was not.

For unlimited testing, use the local provider instead.
