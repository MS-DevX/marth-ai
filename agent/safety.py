"""Path sandbox and confirmation prompts.

The rule this module enforces: the agent may only touch files that resolve
to a path INSIDE the workspace root. "Resolve" means after following `..`,
symlinks, and anything else the filesystem does, so a symlink pointing at
`/etc` is rejected just like a literal `../../etc` would be.
"""

from pathlib import Path

from . import config


class SandboxError(Exception):
    """Raised when a requested path falls outside the workspace root."""


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve `path` and confirm it stays inside the workspace.

    Args:
        path: The path the model asked for. May be relative to the root.
        root: Workspace root to check against. Defaults to
            `config.WORKSPACE_ROOT`. Tests pass an explicit root.

    Returns:
        The fully resolved absolute `Path`, guaranteed to be inside `root`.

    Raises:
        SandboxError: if the path escapes the root, including via `..`,
            an absolute path, or a symlink.
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
    return resolved


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
