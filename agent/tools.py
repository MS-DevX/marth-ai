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

import difflib
import subprocess
from pathlib import Path

from . import config, safety

# Directories that listing and searching skip, to keep results readable.
IGNORED_DIRS = {".git", ".venv", "node_modules"}


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


def write_file(path: str, content: str) -> str:
    """Create a file, or replace its entire contents.

    Args:
        path: File to write, relative to the workspace root.
        content: The full text to write.

    Returns:
        A short report of what happened, or why it did not.
    """
    target = safety.resolve_path(path)
    if target.is_dir():
        return f"Not a file: {path}"
    if not safety.confirm_write(target, content):
        return f"Declined by the user. Nothing was written to {path}."

    existed = target.exists()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Could not write {path}: {exc}"

    verb = "Overwrote" if existed else "Created"
    return f"{verb} {path} ({format_size(len(content))}, {content.count(chr(10)) + 1} lines)."


def edit_file(path: str, old: str, new: str) -> str:
    """Replace one exact piece of text in a file.

    Exactly one match, never zero and never several. Zero means the model
    guessed at the file's contents and should read it first. Several
    means the model cannot be talking about a specific place, so applying
    it would edit somewhere it did not mean to.

    Args:
        path: File to edit, relative to the workspace root.
        old: The exact text to look for.
        new: The text to put in its place.

    Returns:
        A short report of what changed, or why it did not.
    """
    target = safety.resolve_path(path)
    if not target.is_file():
        return f"Not a file: {path}"
    try:
        body = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"Could not read {path}: {exc}"

    # Counted before asking, so the user is never prompted for an edit
    # that was never going to happen.
    matches = body.count(old)
    if matches == 0:
        where = _find_near_miss(body, old)
        return (
            f"No match for that text in {path}. Read the file first so the "
            f"replacement matches exactly, including whitespace."
            + (f" {where}" if where else "")
        )
    if matches > 1:
        return (
            f"That text appears {matches} times in {path}, so replacing it "
            f"would change {matches} places. Include more surrounding "
            f"context to make it unique."
        )

    if not safety.confirm_edit(target, old, new):
        return f"Declined by the user. {path} is unchanged."

    try:
        target.write_text(body.replace(old, new, 1), encoding="utf-8")
    except OSError as exc:
        return f"Could not write {path}: {exc}"
    return f"Edited {path} (1 replacement, {len(old)} -> {len(new)} chars)."


def _find_near_miss(body: str, old: str) -> str:
    """Describe the closest thing in `body` to what was asked for.

    A model that guesses at file contents usually gets it nearly right,
    and being told *how* it differs is far more useful than "no match".

    The comparison is a similarity ratio rather than a substring check,
    because the commonest mistake is a single wrong character
    (`compute_total` for `compute_totals`), which no substring test
    will ever catch: neither string contains the other.
    """
    wanted = [line.strip() for line in old.splitlines() if len(line.strip()) >= 8]
    if not wanted:
        return ""
    candidates = [line.strip() for line in body.splitlines() if line.strip()]
    best_ratio, best_line = 0.0, ""
    for target in wanted[:3]:
        for candidate in candidates:
            # quick_ratio is a cheap upper bound; checking it first avoids
            # the full comparison for the vast majority of lines.
            if difflib.SequenceMatcher(None, target, candidate).quick_ratio() < 0.6:
                continue
            ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
            if ratio > best_ratio:
                best_ratio, best_line = ratio, candidate
    if best_ratio >= 0.75:
        return f"The closest line in the file is: {best_line!r}"
    return ""


def grep(pattern: str, path: str = ".", ignore_case: bool = False) -> str:
    """Search for text across files.

    Args:
        pattern: Text to search for. Treated as a plain string, not a
            regular expression: models write `.` and `(` meaning
            themselves more often than they mean a pattern.
        path: File or directory to search, relative to the workspace root.
        ignore_case: Match without regard to case.

    Returns:
        `path:lineno: line` for each hit, or a note saying there were none.
    """
    target = safety.resolve_path(path)
    if not target.exists():
        return f"No such file or directory: {path}"

    needle = pattern.lower() if ignore_case else pattern
    files = [target] if target.is_file() else sorted(_walk_files(target))
    if not files:
        return f"No searchable files under {path}"

    hits: list[str] = []
    files_with_hits = 0
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: not a search failure
        matched = False
        for number, line in enumerate(text.splitlines(), start=1):
            haystack = line.lower() if ignore_case else line
            if needle in haystack:
                hits.append(
                    f"{file_path.relative_to(config.WORKSPACE_ROOT)}:{number}: "
                    f"{line.strip()[:200]}"
                )
                matched = True
        if matched:
            files_with_hits += 1

    if not hits:
        scope = path if path != "." else "the workspace"
        return f"No matches for {pattern!r} in {scope}."
    header = f"{len(hits)} match(es) for {pattern!r} in {files_with_hits} file(s):"
    return "\n".join([header, *hits])


def _walk_files(root: Path) -> list[Path]:
    """Yield every searchable file under `root`, skipping noise."""
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.name.startswith("."):
            continue
        found.append(path)
    return found


def run_command(command: str) -> str:
    """Run a shell command and report what happened.

    The command is shown to the user in full before it runs, and blocked
    commands are refused outright. Note what this does *not* do: the path
    sandbox does not apply here. A shell command can read and write
    anywhere the user can, so the confirmation and the blocklist are the
    only limits, not a directory boundary.

    Args:
        command: The shell command to run.

    Returns:
        Exit code, stdout, and stderr, or a note that it was refused.
    """
    blocked, reason = safety.is_command_blocked(command)
    if blocked:
        return (
            f"Refused: this command is blocked because {reason}. Blocked "
            f"commands are not run even when approved, so rewriting it "
            f"will not help."
        )

    if not safety.confirm_command(command):
        return "Declined by the user. Nothing was run."

    timed_out = False
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=config.WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=config.COMMAND_TIMEOUT_SECONDS,
        )
        code, out, err = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as expired:
        timed_out = True
        code = -1
        out = (expired.stdout or "") if isinstance(expired.stdout, str) else ""
        err = (expired.stderr or "") if isinstance(expired.stderr, str) else ""

    parts = []
    if timed_out:
        parts.append(
            f"TIMED OUT after {config.COMMAND_TIMEOUT_SECONDS:.0f}s and was killed."
        )
    else:
        parts.append(f"exit code: {code}")
    if out.strip():
        parts.append(f"--- stdout ---\n{out.rstrip()}")
    if err.strip():
        parts.append(f"--- stderr ---\n{err.rstrip()}")
    if not out.strip() and not err.strip() and not timed_out:
        parts.append("(no output)")
    return "\n".join(parts)


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


# Name -> callable. The loop looks tools up by the name the model used,
# and builds the schema list from these same keys, so a tool that is not
# in this dict cannot be called and cannot be described to the model.
TOOL_REGISTRY: dict[str, object] = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "grep": grep,
    "run_command": run_command,
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
    {
        "name": "write_file",
        "description": (
            "Create a new file, or replace the entire contents of an "
            "existing one. For a small change to a file that already "
            "exists, prefer edit_file: it is safer, and it shows a diff. "
            "The user is asked to approve before anything is written."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File to write, relative to the workspace root."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The complete text to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace one exact piece of text in an existing file. The text "
            "to find must appear EXACTLY ONCE: if it appears more than "
            "once the edit is refused, so include enough surrounding "
            "context to make it unique. Read the file first so the text "
            "matches exactly, including indentation. The user sees a diff "
            "and is asked to approve."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File to edit, relative to the workspace root."
                    ),
                },
                "old": {
                    "type": "string",
                    "description": (
                        "Exact existing text to replace, character for "
                        "character."
                    ),
                },
                "new": {
                    "type": "string",
                    "description": "Text to put in its place.",
                },
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search for text across files and get back matching lines with "
            "their file and line number. Use this to find where something "
            "is defined or used, instead of reading every file. The "
            "pattern is plain text, not a regular expression."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The text to search for.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "File or directory to search, relative to the "
                        "workspace root. Defaults to the whole workspace."
                    ),
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Match without regard to case.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command and return its exit code, stdout and "
            "stderr. Use it for things no other tool does, such as running "
            "the tests. The user is shown the exact command and asked to "
            "approve it before it runs, and obviously destructive commands "
            "are refused outright. Note this is not limited to the "
            "workspace the way file tools are."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
            },
            "required": ["command"],
        },
    },
]
