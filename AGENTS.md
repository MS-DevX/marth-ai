# AGENTS.md

Rules for this project. Read this before changing anything.

## Goal

A CLI coding agent: the user types a task, the agent calls an LLM, the LLM
requests tools (read/edit files, run commands), the agent executes them and
feeds the results back, looping until the task is done.

## Tech rules

- Python 3.11+, standard library first.
- Allowed dependencies: `google-genai`, `python-dotenv`, `pytest`. Ask
  before adding anything else.
- No agent frameworks (no LangChain, no OpenAI Agents SDK). The loop must
  stay visible and hand-written.
- Type hints and short docstrings on every function. Small functions,
  clear names.
- The API key comes from the `GEMINI_API_KEY` environment variable, loaded
  from `.env`. Never hardcode a key and never print one.
- The model name is a config value (default lives in `config.py`). Never
  hardcode it anywhere else. A model can be listed by the API and still
  be overloaded on use, so a 503 on one model is not a bug in this code —
  try another before debugging.
- Retry only failures that clear on their own: 429 and 5xx. Never retry a
  400, 401, 403 or 404; they will fail identically and the wait is pure
  cost to the user.
- Check the current google-genai docs for function calling instead of
  guessing the API. Model names drift: check
  `python -m agent.main --list-models` before changing the default, since a
  model can be listed yet still be rejected on use.
- The LLM call lives behind a small interface in `llm.py` so the provider
  can be swapped later. Nothing outside `llm.py` may import `google.genai`
  or make HTTP calls itself.
- Two providers ship: `GeminiLLM` (google-genai SDK) and `OpenAICompatLLM`
  (stdlib `urllib`, for Ollama/Groq/OpenRouter). Select with
  `AGENT_PROVIDER`. Do not add a third without agreeing first.

## Structure

```
marth-ai/
  AGENTS.md
  README.md
  requirements.txt
  .env.example
  .gitignore
  agent/
    __init__.py
    main.py          CLI entry: read task, run loop
    loop.py          the agent loop
    llm.py           provider boundary: Gemini + OpenAI-compatible
    tools.py         tool functions + tool schemas
    safety.py        path sandbox + confirmation prompts
    prompt.py        the system prompt
    config.py        model name, max steps, workspace root
  tests/
    conftest.py       shared `project` fixture
    test_tools.py
    test_write_tools.py
    test_confirmations.py
    test_safety.py
    test_llm.py
    test_loop.py
    test_prompt.py
  check_duplicates.py  dev check: no silently shadowed definitions
  mutation_check.py    dev check: do the tests notice broken safety code?
```

The project directory (`marth-ai/`) is also the default workspace root, so
the agent's sandbox boundary is the project itself.

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

## The system prompt

`agent/prompt.py` holds it, prepended to the task rather than sent as a
system role: Gemini takes its prompt in a config field and the
OpenAI-compatible shape takes it as a message, so prepending is the one
form both cannot silently drop.

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
  the API key is the one file worth stealing.
- Tool output is truncated to `MAX_OUTPUT_CHARS` in `loop.py`, at the one
  point where results enter the conversation. Keep it there. A per-tool
  cap is only as good as the next tool added.
- The whole conversation is re-sent every step, so `loop.py` compacts it
  by dropping the oldest *tool results*. Never drop prose, and never drop
  the most recent result: that is what the model is currently reasoning
  about. Develop this against a local model. Gemini's 1M context hides
  every overflow bug in the design.
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

## Working rules

- Work in phases. Stop after each phase for review, running, and committing.
  Do not run ahead into the next phase.
- After each phase: summarise what was built in plain English, list how to
  run and test it, then wait.
- Explain non-obvious code briefly so there is something to learn from.
- Suggest a git commit message after each phase. Do not commit.
- If a requirement is unclear, ask one question instead of guessing.
