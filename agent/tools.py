"""Tool functions and their function-calling schemas.

Two halves live here:

  * the Python functions the agent actually runs (`read_file`, `list_files`)
  * a JSON Schema description of each one, which is what the model sees

The model never sees the Python signature or the docstring. It only sees
the schema, so the `description` fields below are effectively the prompt
for each tool. Write them for the model, not for yourself.

Tools return a plain string. That string is fed straight back to the
model, so it should read like a short report, not like a Python repr.
Returning a string (rather than raising) is deliberate: a tool that fails
should tell the model what went wrong so it can try something else, not
crash the run.
"""

from . import config, safety

# Directories that listing skips, to keep results readable.
IGNORED_DIRS = {".git", ".venv", "node_modules"}

# Name -> callable. The loop looks tools up by the name the model used.
TOOL_REGISTRY: dict[str, object] = {}


def format_size(num_bytes: int) -> str:
    """Return a human-readable file size such as `1.2 KB`."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def list_files(path: str = ".") -> str:
    """List the files in a directory.

    Args:
        path: Directory to list, relative to the workspace root.

    Returns:
        A newline-separated listing of `dir/file` paths, or a short message
        explaining why nothing could be listed.
    """
    target = safety.resolve_path(path)
    if not target.is_dir():
        return f"Not a directory: {path}"

    entries: list[str] = []
    for entry in sorted(target.iterdir()):
        if entry.name in IGNORED_DIRS:
            continue
        marker = "/" if entry.is_dir() else ""
        entries.append(f"{entry.relative_to(target)}{marker}")

    if not entries:
        return f"No files in {path} (skipping {', '.join(sorted(IGNORED_DIRS))})"
    return "\n".join(entries)


def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a text file, or a numbered range of its lines.

    Large results are truncated before reaching the model, so a long file
    arrives incomplete and the model cannot count lines, grep it, or see
    what was cut. Reading a range is the way back: it asks for less than
    the cap, so the answer arrives whole.

    Args:
        path: File to read, relative to the workspace root.
        start_line: First line to return, 1-based and inclusive. 0 means
            from the beginning.
        end_line: Last line to return, inclusive. 0 means to the end.

    Returns:
        The requested lines, or a short message explaining the problem.
    """
    target = safety.resolve_path(path)
    if not target.is_file():
        return f"Not a file: {path}"
    if target.stat().st_size > config.MAX_FILE_BYTES:
        return (
            f"File is too large ({format_size(target.stat().st_size)}). "
            f"Limit is {format_size(config.MAX_FILE_BYTES)}."
        )

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Not a text file (binary content): {path}"

    # A whole-file read is the common case and needs no line handling.
    if not start_line and not end_line:
        return text

    lines = text.splitlines()
    # Checked before the bounds below, because a reversed range also sits
    # past the end of the file and would otherwise be reported as that.
    if end_line and end_line < start_line:
        return f"end_line ({end_line}) is before start_line ({start_line})."

    first = max(1, start_line) - 1
    last = end_line if end_line else len(lines)
    if first >= len(lines):
        return (
            f"{path} has {len(lines)} lines, so there is nothing at line "
            f"{start_line}. Ask for a range within 1-{len(lines)}."
        )

    chosen = lines[first:last]
    shown = f"{first + 1}-{first + len(chosen)} of {len(lines)}"
    return f"[lines {shown}]\n" + "\n".join(chosen)


TOOL_REGISTRY = {
    "list_files": list_files,
    "read_file": read_file,
}

# What the model is shown. `parameters` is standard JSON Schema.
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "list_files",
        "description": (
            "List the files and directories at a path. Use this first to "
            "explore an unfamiliar project. Returns paths relative to the "
            "listed directory, with a trailing / on directories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to list, relative to the workspace root. "
                        "Use '.' for the root itself."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a text file. Long results are cut short, so for a large "
            "file ask for a line range instead to make sure you see all of "
            "the part you care about. Returns an error message if the path "
            "is missing, a directory, too large, or a secret file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to read, relative to the workspace root.",
                },
                "start_line": {
                    "type": "integer",
                    "description": (
                        "First line to return, 1-based and inclusive. "
                        "Omit to start at the beginning."
                    ),
                },
                "end_line": {
                    "type": "integer",
                    "description": (
                        "Last line to return, inclusive. Omit to read to "
                        "the end of the file."
                    ),
                },
            },
            "required": ["path"],
        },
    },
]
