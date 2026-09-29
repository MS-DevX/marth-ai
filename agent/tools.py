"""Tool functions and their function-calling schemas.

Tools are plain Python functions. Alongside each one we register a JSON
Schema description of it, which is what the model actually sees; the model
never sees the Python signature.

Planned tools:
  1. list_files(path)        Phase 2
  2. read_file(path)         Phase 2
  3. write_file(path, ...)   Phase 4
  4. edit_file(path, ...)    Phase 4
  5. grep(pattern, path)     Phase 4
  6. run_command(command)    Phase 4

Tool dispatch will use a registry mapping name -> callable, so the loop
can run a tool without knowing its name in advance.
"""

# Name -> callable. Populated as each tool is implemented.
TOOL_REGISTRY: dict[str, object] = {}

# Directories that listing and searching skip to keep results readable.
IGNORED_DIRS = {".git", ".venv", "node_modules"}
