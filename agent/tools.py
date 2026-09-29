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


def read_file(path: str) -> str:
    """Read a text file.

    Args:
        path: File to read, relative to the workspace root.

    Returns:
        The file's contents, or a short message explaining the problem.
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
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Not a text file (binary content): {path}"


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
            "Read the full contents of a text file. Returns an error "
            "message if the path is missing, a directory, or too large."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to read, relative to the workspace root.",
                },
            },
            "required": ["path"],
        },
    },
]
