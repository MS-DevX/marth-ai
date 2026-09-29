"""Path sandbox, secret-file blocking, output truncation, confirmations.

The rule this module enforces: the agent may only touch files that resolve
to a path INSIDE the workspace root. "Resolve" means after following `..`,
symlinks, and anything else the filesystem does, so a symlink pointing at
`/etc` is rejected just like a literal `../../etc` would be.

Being inside the workspace is necessary but not sufficient. Secret files
live in the workspace too, and the agent has no business reading them.
"""

from pathlib import Path

from . import config


class SandboxError(Exception):
    """Raised when a requested path falls outside the workspace root."""


class SecretFileError(Exception):
    """Raised when the agent tries to read or write a secret file."""


# Files the agent must never touch, even though they sit inside the
# workspace. `.env` holds the API key: the model has no reason to see it,
# and a prompt injection hidden in any file it *does* read could ask for
# it. Blocking on the filename is a blunt instrument, but it is the
# difference between "the model was asked not to" and "the model cannot".
#
# Checked against the filename only, case-insensitively, because a model
# that wants `../` also wants `.ENV`.
SECRET_NAMES = {
    ".env",
    "credentials",
    "secrets",
    "id_rsa",
    "id_ed25519",
    ".netrc",
    ".npmrc",
    ".pypirc",
}
SECRET_SUFFIXES = (".key", ".pem", ".p12", ".pfx", ".keystore")

# Committed templates of a secret file, which contain no secrets and are
# worth reading: they document which variables the project expects.
SECRET_EXEMPT_SUFFIXES = (".example", ".sample", ".template", ".dist")


def is_secret_file(path: Path) -> bool:
    """Return True if `path` looks like a file holding credentials.

    Args:
        path: Resolved path, or any path with a name to inspect.

    Returns:
        True when the name matches a known secret file, key or password
        file. Templates such as `.env.example` are not secret and return
        False.
    """
    name = path.name.lower()
    if name.endswith(SECRET_EXEMPT_SUFFIXES):
        return False
    if name in SECRET_NAMES:
        return True
    return name.endswith(SECRET_SUFFIXES)


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve `path` and confirm it is safe for the agent to touch.

    Args:
        path: The path the model asked for. May be relative to the root.
        root: Workspace root to check against. Defaults to
            `config.WORKSPACE_ROOT`. Tests pass an explicit root.

    Returns:
        The fully resolved absolute `Path`, guaranteed to be inside `root`
        and not a secret file.

    Raises:
        SandboxError: if the path escapes the root, including via `..`,
            an absolute path, or a symlink.
        SecretFileError: if the path names a file holding credentials.
    """
    base = (root or config.WORKSPACE_ROOT).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    # `resolve()` collapses `..` AND follows symlinks, so this is the real
    # location on disk, not the string the model wrote.
    resolved = candidate.resolve()

    # `is_relative_to` is the explicit form of "is it under this dir".
    if resolved != base and not resolved.is_relative_to(base):
        raise SandboxError(
            f"Path is outside the workspace ({base}): {path}"
        )

    # Checked after the boundary, so the check is on the real name on disk
    # and cannot be bypassed by reaching the file from another directory.
    if is_secret_file(resolved):
        raise SecretFileError(
            f"Refusing to touch a secret file: {resolved.name}. The agent "
            f"does not need credentials to read code, and a file it has "
            f"read could contain instructions asking it to fetch them."
        )
    return resolved


def truncate(text: str, limit: int | None = None) -> str:
    """Cut `text` to `limit` characters and say what was dropped.

    Every byte a tool returns goes into the model's context window, and
    that window is finite. A single large file can fill it, which starves
    the next turn. This is applied to tool output as a whole rather than
    inside each tool, so a future tool cannot forget to do it.

    Args:
        text: The text to shorten.
        limit: Maximum characters. Defaults to `config.MAX_OUTPUT_CHARS`.

    Returns:
        The text unchanged if it fits, otherwise the first `limit`
        characters plus a note describing how to get the rest.
    """
    cap = config.MAX_OUTPUT_CHARS if limit is None else limit
    if len(text) <= cap:
        return text
    dropped = len(text) - cap
    return (
        f"{text[:cap]}\n\n"
        f"[truncated: {dropped} more characters not shown, out of {len(text)} "
        f"total. This file is longer than the context allows in one piece, "
        f"so counts and totals derived from it are unreliable. Call "
        f"read_file again with start_line and end_line to read the rest in "
        f"sections rather than repeating this same request.]"
    )


# --- Phase 4 -------------------------------------------------------------
# The following will live here once write_file / edit_file / run_command
# are implemented. They are declared now so the shape of the module is
# visible in advance:
#
#   confirm_write(resolved_path, content) -> bool
#   confirm_edit(resolved_path, old, new) -> bool   # shows a diff
#   confirm_command(command) -> bool                 # shows the full command
#   is_command_blocked(command) -> bool               # rm -rf /, format, ...
#   truncate(text, limit) -> str
