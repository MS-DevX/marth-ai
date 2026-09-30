# marth-ai

A small terminal coding agent, written from scratch so the loop is visible.

You type a task, it calls an LLM, the LLM asks for tools (read a file, run
a command), the agent runs them and feeds the results back, repeating until
the task is done. No agent frameworks — just the Gemini SDK and the standard
library.

## Status

Phase 7 is done. All six tools work, the model is told how to use them,
and every run is watchable live and readable afterwards.

## Using it

The agent is a coding assistant with a short leash. Point it at a
directory and give it a task in plain English:

```bash
cd marth-ai
python -m agent.main "explain what agent/loop.py does"
python -m agent.main "add a --verbose flag to main.py, then run the tests"
python -m agent.main                     # prompts for the task
```

It reads, edits and runs commands on its own, asking before each change.
Because it asks, **run it in a terminal** — with no terminal there is
nothing to ask, and the answer is no.

### Watching a run

While the agent works, the screen shows what it is doing: the task, the
model, and a running log of each tool call with one line of what came
back. A refused call is marked `!` in amber, a blocked one `X` in red, and
a call that failed `E` — so a run where the agent kept asking for things
you said no to does not look like a run where everything worked.

Confirmations appear on the same screen, one `y`/`n` keystroke each,
instead of dropping you out to a shell prompt and back.

The dashboard needs a real terminal. With a pipe, a small window, or
`--yes` (nothing to approve) it steps aside and prints a plain log
instead, and says nothing rather than drawing escape sequences over your
output. Confirmations still work either way — the prompt is the same
`y`/`n` you would get without a dashboard:

```bash
python -m agent.main --plain "..."          # force the plain log
```

The plain log is not a lesser view. It prints the same lines, the same
markers and the same summary:

```
  [      ok] read_file(path=stats.py)  (0.01s)
           def total(values):
  [      ok] edit_file(path=stats.py, old=sum(values), new=sum(values, 0))  (0.01s)
           Edited stats.py (1 replacement, 11 -> 14 chars).
  [ declined] write_file(path=report.md, content=...)  (0.00s)
           Declined by the user. Nothing was written to report.md.

  finished in 4 calls, 5 steps, 41s
  3 ok, 1 declined

  Changed:
    stats.py
```

Note the `declined` row is counted separately from `ok`, and the file it
wanted to write is not under **Changed**. A run that looks finished but
had three refusals is not the same as one that did what it was asked.

One line of each result is all either view shows, because a whole file
would be unreadable in a log and useless in a column. The rest is in the
history file.

### Looking at earlier runs

Every run is written to `.marth-ai/runs/` as it happens, so a run can be
read back afterwards — including one that was interrupted, or one you
want to check claims against.

```bash
python -m agent.main --history              # recent runs, newest first
python -m agent.main --run 20260930-141203-a7f1   # one run in full
```

```
  run id                   when         outcome     steps  task
  ------------------------ ------------ ----------- -----  ------------------
  20260930-141203-a7f1     09-30 14:12  finished        6  add a --verbose flag
  20260930-120455-3c09     09-30 12:04  step_limit     2  rename compute_total
  20260930-091203-ee51     09-30 09:12  finished       14  explain loop.py
```

`--run` takes the id from the first column, and a prefix of it works. A
prefix matching more than one run says so rather than quietly showing you
the newest of them, because you asked about one run and would be reading
another's output as though it were this one's.

`--run` prints the whole thing: the task, every call with its arguments
and full output, and the answer. This is the way to find out what actually
happened after a run you did not watch.

The log is written as the run proceeds, not at the end, so a run killed
mid-step still has a record — and one with no final line is shown as
`running`, which is what it was.

Turn it off with `--no-history` (`AGENT_NO_HISTORY=1`) if you would
rather the agent wrote nothing outside the workspace at all. The
directory is gitignored either way.

### Running against your own code

By default the agent is sandboxed to the `marth-ai` directory, which is
not very useful for real work. Point it somewhere else:

```bash
AGENT_WORKSPACE=~/projects/myapp python -m agent.main "add type hints to stats.py"
```

The agent's first move will be `list_files`, so you can see it has found
the right place. It stays inside that directory for file tools; shell
commands do not (see below).

### Reviewing before it happens

Every change is shown before it is made, and you approve with `y`:

```
============================================================
The agent wants to edit: stats.py
------------------------------------------------------------
--- stats.py (current)
+++ stats.py (proposed)
@@ -12,3 +12,3 @@
-    return sum(values)
+    return sum(values, start)
============================================================
Apply this edit? [y/N]
```

Anything else declines. After declining, the agent explains what it
wanted and stops — it does not ask again, so there is no way to get
trapped approving something you did not read.

To approve everything instead (CI, a script, a throwaway checkout):

```bash
python -m agent.main --yes "..."
```

### When it stops

The loop ends in one of three ways, and says which:

| Stop | What it prints |
| --- | --- |
| Finished | The model's own summary of what it did |
| Repeating | The same tool call with the same arguments, three times running |
| Step limit | Told it hit `MAX_STEPS`, with the suggestion to raise it |

A refusal mid-run (you declined an edit) shows up as the agent
describing what it wanted to do and stopping, not as an error.

### When it goes wrong

**It claims something works without checking.** The prompt tells it to
run the tests and to say plainly what it ran. If it reports success
without showing you output, it did not verify.

**It stops early and tells you to raise the limit.** Raise it:

```bash
AGENT_MAX_STEPS=40 python -m agent.main "..."
```

**It reasons about a file it only partly saw.** Tool output is cut at
4000 characters. The truncation notice says how much is missing; the fix
is for the model to re-read with `start_line` and `end_line`. If it
counts things it never fully read, raise `AGENT_MAX_OUTPUT_CHARS` or
have it use `grep`.

**A local model is wrong in a specific way.** 8B models drop entries
from directory listings, miscount truncated files, and call tools that
do not exist. `granite4.1:8b` was noticeably more accurate than
`lfm2.5:8b` on the same tasks. A cloud model makes these far rarer, at
the cost of a daily quota.

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

# Approve every write/edit/command without being asked (see below)
python -m agent.main --yes "add a tests/test_thing.py and run pytest"

# Watch it work on the live dashboard (the default in a real terminal)
python -m agent.main "add type hints to stats.py"

# Print a plain log instead - needed when piping, and honest for scripts
python -m agent.main --plain "add type hints to stats.py"

# Review what previous runs did
python -m agent.main --history
python -m agent.main --run 20260930-141203-a7f1

# Do not write a run log at all
python -m agent.main --no-history "explain what this repo does"

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
| `AGENT_YES` | `0` | `1` is the same as `--yes`; skips confirmations |
| `AGENT_NO_HISTORY` | `0` | `1` is the same as `--no-history`; writes no run log |
| `AGENT_MAX_HISTORY_RUNS` | `200` | Oldest run logs deleted past this many |

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

## Before anything changes

`write_file`, `edit_file` and `run_command` ask first. You see exactly
what will happen — the new contents, a diff, or the full command — and
anything other than `y`/`yes` declines, including just pressing Enter.

`edit_file` needs the text to appear **exactly once**. Zero matches means
the model guessed at the file and is told to read it again (including
what the closest actual line was, when there is one). Several matches
means it cannot be talking about a specific place, so the edit is refused
rather than applied in the wrong spot.

### Commands that are never run

Some commands are refused outright, without even asking, because a
confirmation is not a defence against a command that was never meant to
be run and a model stuck in a loop will happily ask for something
catastrophic:

| Refused | Examples |
| --- | --- |
| Whole-disk deletes | `rm -rf /`, `rm -rf /*`, `rm -rf ~`, `rm -rf $HOME`, `rm -rf ..` |
| Top-level system dirs | `rm -rf /etc`, `rm -rf /usr`, `rm --recursive --force /` |
| Disk and filesystem | `mkfs.ext4 /dev/sda1`, `dd of=/dev/sda`, `fdisk`, `shred` |
| Privilege | `sudo ...`, `passwd`, `chmod 777 /`, `chown -R ... /` |
| Machine state | `shutdown`, `reboot`, `halt`, `kill -9 1` |
| Shell profile writes | `echo x > ~/.bashrc` |
| Piping downloads into a shell | `curl http://x.sh \| sh` |
| Unrecoverable git | `git push --force` |

The rules are deliberately blunt, in both directions. A false positive
costs one refused command you can rewrite; a false negative can cost the
disk. `rm -rf build/` and `rm -rf node_modules` stay allowed, because
those are ordinary housekeeping and a blocklist that cries wolf gets
switched off — at which point it protects nothing at all. `git push
--force-with-lease` is allowed too; only the unrecoverable `--force` is
refused.

### Two limits worth knowing

**`run_command` is not sandboxed by path.** A shell command can read and
write anywhere you can; the workspace boundary that governs the file
tools does not apply to it. The confirmation prompt and the blocklist are
the only limits. Treat approving a command as approving what that command
could reach, not just what it appears to do.

**`--yes` turns off the prompt, not the blocklist.** With `--yes`
(`AGENT_YES=1`) every write, edit and command runs without asking, which
is what makes the agent usable from a script or a pipeline. Destructive
commands are still refused, and a banner says so on every run. Without a
terminal and without `--yes`, the answer is always no: an unattended run
should not silently do whatever the model asked.

### Output limits

Every tool result is truncated to `AGENT_MAX_OUTPUT_CHARS` (4000) before it
reaches the model, at a single choke point in the loop so no tool can skip
it. Without this, reading one 20 KB source file fills most of an 8K context
window and the next turn has nowhere to go. Truncated results say how much
was dropped, and suggest reading the rest by line range rather than
repeating the same request.

Long shell output is truncated the same way, and a command that runs past
`AGENT_COMMAND_TIMEOUT` (30s) is killed and reported as a timeout rather
than left running.

## Layout

```
marth-ai/
  AGENTS.md          rules for future sessions
  README.md          this file
  requirements.txt
  .env.example
  agent/
    main.py          CLI entry: read task, run loop, print a summary
    loop.py          the agent loop + the tool-output truncation choke point
    llm.py           provider boundary: Gemini + OpenAI-compatible
    tools.py         tool functions + tool schemas
    safety.py        path sandbox, secret blocking, truncation, confirmations
    prompt.py        the system prompt, and why each line is there
    config.py        model name, max steps, workspace root
    runs.py          what a run did: the record every view reads
    history.py       writing that record to disk, and reading it back
    tui.py           the live curses dashboard
    report.py        the plain-text log and the end-of-run summary
  tests/
    conftest.py      shared `project` fixture
    test_tools.py
    test_write_tools.py
    test_confirmations.py
    test_safety.py
    test_llm.py
    test_loop.py
    test_prompt.py
    test_refusal_contract.py
    test_runs.py
    test_history.py
    test_history_cli.py
    test_report.py
    test_tui.py
    test_dashboard_e2e.py
  check_duplicates.py   dev check: no silently shadowed definitions
  mutation_check.py     dev check: do the tests notice broken safety code?
```

### Checking the safety code

Two scripts exist because the important tests here are the ones that fail
when the code is wrong.

`python3 check_duplicates.py` fails if any module defines the same name
twice. Python takes the last definition and ignores the rest, so a
duplicated block is dead code that still reads as though it were live.
`safety.py` carried 140 such lines during Phase 4 — a full second copy of
the module — and the tests passed the whole time, because they were
exercising the copy nobody was looking at.

`./.venv/bin/python mutation_check.py` deliberately breaks eleven safety
behaviours one at a time (allow `sudo`, make the confirmation default to
yes, ignore the command timeout, drop the sandbox check) and confirms a
test catches each one. A safety test that passes no matter what the code
does is not a test.

## Roadmap

- [x] Phase 1 — scaffold, config, one-shot Gemini call
- [x] Phase 2 — `read_file` + `list_files` and one tool round trip
- [x] Phase 3 — the full agent loop
- [x] Phase 4 — `write_file`, `edit_file`, `grep`, `run_command` + confirmations
- [x] Phase 5 — system prompt, usage docs, wider test coverage
- [x] Phase 6 — the run record, the history file, the summary
- [x] Phase 7 — the live curses dashboard and the run viewer

Not planned, and worth saying out loud: this is a small agent for one
person's projects. It has no support for structured output beyond what
the tools return, no retry-on-tool-failure logic in the model itself,
no multi-agent anything, and no web access.

## How the model is told what to do

`agent/prompt.py` holds the system prompt. It is prepended to the task
rather than sent as a separate system role, because Gemini takes its
system prompt in a config field and the OpenAI-compatible shape takes
it as a message; prepending works identically on both and cannot be
silently dropped by a provider that ignores it.

Every line in it answers a failure that actually happened while
building this agent:

| Failure | What the prompt says |
| --- | --- |
| Counted 8 of 32 tests in a file it had only seen 28% of | Do not reason about a file you have only partly seen; do not count what you did not read |
| Edited using text it guessed at | Read before you change anything; read what you wrote before editing again |
| Called `confirm_write`, which did not exist, then used the shell instead | Only use the tools you were given; say so if you need another, do not invent it |
| Re-asked for an action after it was declined | A refusal is a decision, not a retry; explain and stop |
| Finished without checking | Run the tests, or run what you changed, and say what it printed |
| Kept going with no idea how many steps were left | You will be told about the step limit |
| Assumed a truncation limit that did not exist | The real number, which is asserted against `config` in a test |

`tests/test_prompt.py` checks each of those phrases is present, so a
rule cannot be quietly deleted. It also fails on an unfilled `{}`
placeholder and on the prompt growing past 700 words, since it is
re-sent on every single step and competes with tool output for a small
context window.

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

### Finding things without reading everything

`grep` is the other half of that. It searches every file under a
directory and returns `path:lineno: line`, so the model can locate where
something is defined without pulling whole files into context one by one.

The pattern is plain text, not a regular expression — models write `.`
and `(` meaning themselves far more often than they mean a pattern, and
a regex that fails silently returns nothing, which is worse than an
exact-but-plain match. It skips `.git`, `.venv`, `node_modules` and
dotfiles, and silently steps over binary files rather than reporting a
search failure for them.

## The dashboard

Two views over one record. `agent/runs.py` defines what a run *is* — the
task, the tool calls, the outcome — and the other three modules render it:
`tui.py` live, `report.py` for the summary, `history.py` for the file on
disk. Nothing in `runs.py` draws anything, so the record can be compared
in a test with no terminal anywhere in sight.

`loop.run_record()` returns that record and takes an `on_event` callback
that fires as each step completes. The dashboard subscribes to that and
redraws; the plain log subscribes to the same callback and prints. One
loop, two observers, and neither of them has to poll or know the other
exists.

### How a confirmation works while curses owns the screen

This is the one genuinely awkward part. `safety.py` asks by calling
`input()`, and `input()` cannot work on a screen curses is drawing.

So `safety` asks through an indirection:

```python
safety.set_ask_handler(Dashboard.ask)   # curses suspends, asks, resumes
```

`Dashboard.ask` calls `curses.endwin()`, runs the normal `input()` prompt
against the real terminal, then `curses.doupdate()` and redraws. The
prompt is the familiar one, and the log has no hole where it was. The
handler is `None` everywhere else, so `safety.py` has no idea a dashboard
exists — and `--plain` needs no special case, because it simply never
installs one.

### How a call knows it failed

Tools return a `Result`, which is a `str` subclass carrying a `status`:
`ok`, `declined`, `blocked` or `error`. The model sees the text; the
dashboard and the history file see the status.

The status used to be guessed afterwards, by searching the text for words
like "declined by the user". That failed in both directions, and both
failures were real:

- a write outside the workspace was refused correctly and then recorded
  as `ok`, so the history listed a refused write under **Changed:**
- a file the model read that happened to contain the words "declined by
  the user" was recorded as a refusal

The status is known at the moment the call happens, so it is reported
there. `tests/test_refusal_contract.py` covers every route a call can
fail by, because a route nobody checks is a route that silently records
itself as a success.

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

## A note on the Gemini role format

Function results go back to Gemini under `role="user"`, not
`role="tool"`. The API used to accept a `tool` role and no longer does;
it answers `400 INVALID_ARGUMENT: Role 'tool' is not supported`, with no
hint about what it wants instead. This was found by running the agent
against the live endpoint during Phase 5, not by reading the docs — the
published function-calling guide now leads with a newer Interactions API
and no longer shows this request shape at all.

Check `llm.py` against the current docs before changing it, and verify
against a real request rather than trusting a snippet that happens to
look plausible.
