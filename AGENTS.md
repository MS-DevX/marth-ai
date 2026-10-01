# AGENTS.md

Rules for this project. Read this before changing anything.

## Goal

A CLI coding agent: the user types a task, the agent calls an LLM, the LLM
requests tools (read/edit files, run commands), the agent executes them and
feeds the results back, looping until the task is done.

## Tech rules

- Python 3.11+, standard library first.
- Allowed dependencies: `python-dotenv`, `pytest`. Ask before adding
  anything else.
- No agent frameworks (no LangChain, no OpenAI Agents SDK). The loop must
  stay visible and hand-written.
- Type hints and short docstrings on every function. Small functions,
  clear names.
- No API key is needed by default: the model runs locally. A key is read
  from `OPENAI_API_KEY` in `.env` only when `OPENAI_BASE_URL` has been
  repointed at a server that wants one. Never hardcode a key and never
  print one.
- The model name is a config value (default lives in `config.py`). Never
  hardcode it anywhere else. A model can be listed by the API and still
  be overloaded on use, so a 503 on one model is not a bug in this code —
  try another before debugging.
- Retry only failures that clear on their own: 429 and 5xx. Never retry a
  400, 401, 403 or 404; they will fail identically and the wait is pure
  cost to the user.
- Check the current OpenAI-compatible chat-completions docs for function
  calling instead of guessing the shape. Model names drift: check
  `python -m agent.main --list-models` before changing the default, since a
  model can be listed yet still be rejected on use.
- The LLM call lives behind a small interface in `llm.py` so the provider
  can be swapped later. Nothing outside `llm.py` may import an LLM SDK or
  make HTTP calls itself. A test scans the package for that.
- One provider ships: `OpenAICompatLLM`, built on stdlib `urllib`, which is
  how the agent reaches a local Ollama server. There is deliberately no
  `AGENT_PROVIDER` setting — a setting that cannot take a second value is
  not a setting, and the old one silently fell through to Gemini on a typo.
  Do not add another provider without agreeing first.

## Structure

```
marth-ai/
  AGENTS.md
  README.md
  pyproject.toml      packaging: the `marth` command, the deps, the version
  requirements.txt    the same deps, for a plain `pip install -r`
  .env.example
  .gitignore
  agent/
    __init__.py       __version__, read by pyproject at build time
    main.py           CLI entry: read task, run loop, print a summary
    loop.py           the agent loop
    llm.py            provider boundary: the OpenAI-compatible client
                      + OllamaModels, the local model manager
    setup.py          `marth --setup`: check for Ollama, fetch the model
    tools.py          tool functions + tool schemas
    safety.py         path sandbox + confirmation prompts
    prompt.py         the system prompt
    config.py         model name, max steps, workspace root, settings file
    runs.py           what a run did: the record every view reads
    history.py        writing that record to disk, and reading it back
    tui.py            the live curses dashboard
    report.py         the plain-text log and the end-of-run summary
  tests/
    conftest.py       shared `project` fixture
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
    test_cli.py       what the CLI says when something goes wrong
    test_tui.py
    test_dashboard_e2e.py
    test_config.py     where the agent looks, installed vs checkout
    test_packaging.py  the install manifest, and that it agrees with
                       requirements.txt
    test_setup.py      `--setup` against a fake Ollama, and in a real pty
  check_duplicates.py  dev check: no silently shadowed definitions
  mutation_check.py    dev check: do the tests notice broken safety code?
```

The default workspace root is the project directory in a checkout and the
current directory when installed. See "Installed, or run from a
checkout" below for why that is not a detail.

## Packaging

`pyproject.toml` is the source of truth for dependencies;
`requirements.txt` mirrors it for the plain `pip install -r` path, and
`tests/test_packaging.py` fails if they drift. `pytest` is a dev extra
and must not become a runtime dependency.

`version` is read from `agent.__version__` at build time, so
`agent/__init__.py` must stay cheap to import — no submodule imports, no
work at import. `packages = ["agent"]` is listed explicitly because
auto-discovery would install `tests/` and the two dev scripts as
top-level importable modules.

## Tools

1. `list_files(path)` — list files, skipping `.git`, `.venv`, `node_modules`
2. `read_file(path, start_line, end_line)` — return contents, truncating
   very large files. The line range exists because truncation hides the
   rest of a file: a model asked how many tests a 448-line file had
   counted 8 of 32. Ranges come back under the cap and therefore whole.
3. `write_file(path, content)` — create or overwrite a file
4. `edit_file(path, old, new)` — replace ONE exact match; error on zero or
   multiple matches
5. `grep(pattern, path)` — search text in files
6. `run_command(command)` — run a shell command with a timeout, return
   stdout / stderr / exit code

Tools 3-6 are Phase 4. Do not add a tool that tells the model to use
another tool that does not exist: that guidance has to be true, because
the model acts on it and cannot tell the difference between a broken
promise and a real failure.

`grep` matches plain text, not a regular expression. Models write `.`
and `(` meaning themselves far more often than they mean a pattern.

## The run record

`runs.py` holds what a run is: the task, the tool calls, the outcome.
`tui.py`, `report.py` and `history.py` all render that one record and
none of them may define a second, slightly different version of it. It is
plain data with no behaviour, so a run can be compared in a test with no
terminal in sight.

`loop.run_record()` returns the record and fires `on_event` as each step
completes. The live views subscribe to the callback. They must not poll
the loop, and the loop must not know a view exists.

## Reporting a tool failure

Tools return a `Result`: a `str` subclass carrying `status` (`ok`,
`declined`, `blocked`, `error`) and `approved`. The model sees the text;
the dashboard and the history file see the status.

Do not read the status back out of the text. That was tried and it failed
both ways: a write refused by the sandbox was recorded as `ok` and listed
under "Changed:", and a file the model read containing the words
"declined by the user" was recorded as a refusal. Anything the model reads
can contain any text at all, including this repository's own source.

A tool that returns a message instead of raising must return
`tools.failed(...)`, or the run records a failure as a success. A search
that found nothing is *not* a failure: it is the answer the model asked
for. A command that exited non-zero **is** one, because a failing
`pytest` is not a clean run and must not be rendered green. A command
killed at the timeout is one as well.

## Confirming while curses owns the screen

`input()` cannot work on a screen curses is drawing, so `safety` asks
through `set_ask_handler()`. The handler is `None` everywhere except under
the dashboard, which suspends curses, asks, and resumes. Keep it that way:
`safety.py` must not import curses, and `--plain` must need no special
case because it never installs a handler.

## The history file

`runs/` under the workspace, one JSONL file per run, appended as the run
proceeds so an interrupted run still has a record. A file with no footer
is a run that was still going, and is shown as such.

Run logs go *inside* the workspace rather than in a user-wide config
directory, and they are gitignored. A user who would rather the agent
wrote nothing outside the task at all can turn them off with
`--no-history`; that is a choice, not a fallback, so do not make the log
optional for reasons of convenience.

## Installed, or run from a checkout

`config.is_checkout()` decides which of two worlds we are in, by asking
whether a `pyproject.toml` sits next to the `agent/` package. An
installed copy has none, because the file that built it was not shipped.

Three things follow from that one fact, and all three would be silently
wrong rather than loudly broken:

- the sandbox root, which is the project directory in a checkout and the
  **current directory** when installed (`config.default_workspace`). A
  sandbox rooted at `site-packages` would hold the agent's own
  dependencies and none of the user's code.
- where settings are read from: the project's `.env` in a checkout,
  `~/.config/marth-ai/.env` when installed (`config.env_file_for`).
- nothing else. Do not add a third `if is_checkout` without a reason.

The `.env` is loaded in `config`, at import, **above** the values that
read it. `MODEL_NAME` is read a few lines below the load, so loading it
in `main()` instead is a silent no-op: the file is read too late to
matter. `load_dotenv` is called without `override=True`, so a real
environment variable always wins and a one-off override needs no file
edit.

## Installing a model

`setup.py` answers "can this machine run a task yet" and fixes it. It
checks, in order, because each failure has a different fix: the `ollama`
binary, then a server, then the model.

The HTTP is in `llm.OllamaModels`, not in `setup.py`, like everything
else that talks to the server. It is not a provider — it never sees a
prompt and implements no part of the `LLM` protocol.

Three things in here were bugs before they were code, so do not undo
them:

- `has_model` matches **exactly**. `qwen2.5:0.5b` and `qwen2.5:1.5b`
  share a name and are different models; a loose match skipped a
  gigabyte of download and 404ed on every run after.
- the progress reporter is given `sys.stdout.write`, not `print`. A
  printer appends a newline, which turns 400 updates into 400 lines
  instead of one rewritten line. Only the pty test catches this.
- Ollama reports `total`/`completed` per *layer*, never per model. Taken
  literally the bar reads 100% after the first of nine layers. `_PullTally`
  accumulates them and caps at 99 until the server says `success`, because
  layers already on disk are never mentioned and the denominator
  under-counts.

Setup will not install Ollama itself: a large third-party binary on the
user's `PATH` is their decision, not an installer's. It prints the link
and stops.

## The system prompt

`agent/prompt.py` holds it, prepended to the task rather than sent as a
system role. The shape takes a system message, but prepending is the one
form a server cannot silently drop: it arrives as ordinary user text.

Every line must answer a failure that actually happened. If a rule no
longer prevents anything real, delete the rule rather than keeping it
because it sounds sensible, and drop its entry from `REGRESSION` in
`tests/test_prompt.py` at the same time. That test is the only thing
stopping the prompt quietly shrinking or quietly growing, and it is
worth updating deliberately rather than deleting when it fails.

## Safety rules

- All paths must resolve inside the workspace root. Reject anything
  outside, including `../` tricks and symlinks.
- Inside the workspace is not enough. Secret files are refused by name
  (`.env`, `*.pem`, `id_rsa`, ...), with committed templates like
  `.env.example` explicitly exempt. Do not rely on the model choosing
  to refuse: a file it reads can contain instructions aimed at it, and
  a key file is the one thing worth stealing.
- Tool output is truncated to `MAX_OUTPUT_CHARS` in `loop.py`, at the one
  point where results enter the conversation. Keep it there. A per-tool
  cap is only as good as the next tool added.
- The whole conversation is re-sent every step, so `loop.py` compacts it
  by dropping the oldest *tool results*. Never drop prose, and never drop
  the most recent result: that is what the model is currently reasoning
  about. The default model has an 8K window, so an overflow shows up
  within a normal task rather than being hidden by a huge context.
- `write_file`, `edit_file`, and `run_command` require an explicit y/n
  confirmation that shows exactly what will happen: a diff for edits, the
  full command for shell.
- Block obviously destructive commands (`rm -rf /`, `format`, etc.) even if
  the user confirms. A confirmation is not a defence against a command
  that was never meant to be run, and a model stuck in a loop will ask.
- Command timeout defaults to 30s. Long output is truncated.
- The loop stops after a max step count (default 20) so a confused model
  cannot burn the API budget forever.

## Two limits that are not code

- `run_command` is **not** sandboxed by path. A shell command reaches
  anywhere the user can; the confirmation and the blocklist are the only
  limits. Keep saying so in the README rather than letting the workspace
  boundary imply more than it covers.
- `--yes` skips the confirmation, never the blocklist. The blocklist is a
  refusal, not a question, so no approval can lift it.

## Verifying safety code

A test that passes no matter what the code does is not a test. Run both
of these after touching `safety.py` or the write tools:

```bash
python3 check_duplicates.py             # no silently shadowed definitions
./.venv/bin/python mutation_check.py     # each safety rule has a test that notices
```

`check_duplicates.py` exists because `safety.py` once carried two
complete copies of itself. Python takes the last definition and ignores
the rest, so the first was dead code, and the tests passed throughout
because they exercised the copy nobody was reading. Do not reintroduce a
block by pasting module-level code into the middle of a file.

`mutation_check.py` breaks eleven safety behaviours one at a time and
requires a test to fail for each one. A mutation that survives means a
property is untested; fix the test, not the mutation.

## Testing the dashboard

`tests/test_dashboard_e2e.py` runs the real CLI in a real pty, because the
three things most likely to break cannot be seen any other way: the
alternate screen being entered, the terminal being handed back, and
confirmations arriving on the screen curses is drawing.

Three requirements, each learned the hard way:

- **Fork, do not `execve`.** The child needs the monkeypatched model and
  the patched ask handler. `execve` replaces the process image and throws
  both away, and the run then tries to reach a real model.
- **Rebind `sys.stdout` to fd 1 in the child.** Curses writes to the file
  descriptor, not to the Python object, so a suite that has captured
  stdout leaves curses drawing into a buffer nobody reads.
- **Set the window size with `TIOCSWINSZ`.** A pty defaults to 0x0, and
  `available()` correctly refuses a window that small. The dashboard never
  starts and the test passes for the wrong reason.

These tests really do run `write_file`, so give them a temporary workspace
and assert the repository was not written to. `config.WORKSPACE_ROOT` is
patched in the parent rather than through `AGENT_WORKSPACE`, because
`config` was imported long before an environment variable set in a test
would be read.

## Working rules

- Work in phases. Stop after each phase for review, running, and committing.
  Do not run ahead into the next phase.
- After each phase: summarise what was built in plain English, list how to
  run and test it, then wait.
- Explain non-obvious code briefly so there is something to learn from.
- Suggest a git commit message after each phase. Do not commit.
- If a requirement is unclear, ask one question instead of guessing.
