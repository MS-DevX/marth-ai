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
  hardcode it anywhere else.
- Check the current google-genai docs for function calling instead of
  guessing the API. Model names drift: check
  `python -m agent.main --list-models` before changing the default, since a
  model can be listed yet still be rejected on use.
- The LLM call lives behind a small interface in `llm.py` so the provider
  can be swapped later. Nothing outside `llm.py` may import `google.genai`.

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
    llm.py           Gemini wrapper
    tools.py         tool functions + tool schemas
    safety.py        path sandbox + confirmation prompts
    config.py        model name, max steps, workspace root
  tests/
    test_tools.py
    test_safety.py
```

The project directory (`marth-ai/`) is also the default workspace root, so
the agent's sandbox boundary is the project itself.

## Tools

1. `list_files(path)` — list files, skipping `.git`, `.venv`, `node_modules`
2. `read_file(path)` — return contents, truncating very large files
3. `write_file(path, content)` — create or overwrite a file
4. `edit_file(path, old, new)` — replace ONE exact match; error on zero or
   multiple matches
5. `grep(pattern, path)` — search text in files
6. `run_command(command)` — run a shell command with a timeout, return
   stdout / stderr / exit code

## Safety rules

- All paths must resolve inside the workspace root. Reject anything
  outside, including `../` tricks and symlinks.
- `write_file`, `edit_file`, and `run_command` require an explicit y/n
  confirmation that shows exactly what will happen: a diff for edits, the
  full command for shell.
- Block obviously destructive commands (`rm -rf /`, `format`, etc.) even if
  the user confirms.
- Command timeout defaults to 30s. Long output is truncated.
- The loop stops after a max step count (default 20) so a confused model
  cannot burn the API budget forever.

## Working rules

- Work in phases. Stop after each phase for review, running, and committing.
  Do not run ahead into the next phase.
- After each phase: summarise what was built in plain English, list how to
  run and test it, then wait.
- Explain non-obvious code briefly so there is something to learn from.
- Suggest a git commit message after each phase. Do not commit.
- If a requirement is unclear, ask one question instead of guessing.
