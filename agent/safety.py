"""Path sandbox, secret blocking, output truncation, confirmations.

The rule this module enforces: the agent may only touch files that resolve
to a path INSIDE the workspace root. "Resolve" means after following `..`,
symlinks, and anything else the filesystem does, so a symlink pointing at
`/etc` is rejected just like a literal `../../etc` would be.

Being inside the workspace is necessary but not sufficient. Secret files
live in the workspace too, and the agent has no business reading them.

Everything here is read-only or a prompt. Nothing in this module performs
the action the user is being asked about, so a bug in here can only ever
fail to ask, never to act.
"""

import difflib
import sys
from collections.abc import Callable
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


# --- Command blocklist ----------------------------------------------------
#
# These are refused outright. A confirmation prompt exists so the user can
# make a judgement about a command they can see; it is not a defence
# against a command that was never meant to be run, and a model that is
# looping or confused will happily ask for something catastrophic.
#
# Matched against the command with whitespace collapsed and LOWERCASED, so
# every pattern below must be lowercase too. Writing `-R` or `$HOME` in a
# pattern here is a silent hole: the input can never contain an uppercase
# letter, so the rule quietly stops matching.
#
# The patterns are deliberately blunt. A false positive costs one refused
# command the user can rewrite; a false negative can cost the disk.
#
# `rm` in any flag form, short or long, followed by a root-level target.
_RM = r"\brm\b(?:\s+-{1,2}[a-z-]+)*\s+"

# Top-level directories whose removal breaks the machine.
_SYSTEM_DIRS = r"/?(?:bin|boot|dev|etc|home|lib|opt|proc|root|sbin|srv|sys|usr|var)"

DESTRUCTIVE_PATTERNS = (
    # Whole-disk and recursive deletes. The trailing `[/*]*` is
    # what catches `/`, `/*`, `~/`, `../` and `$HOME/`, not just
    # the bare form.
    _RM + r"(?:/|~|\$home|\*|\.\.?)[/*]*(?:\s|$)",
    _RM + _SYSTEM_DIRS + r"(?:/|\s|$)",
    # Disk and filesystem tools.
    r"\bmkfs(?:\.|\s)",
    r"\bdd\b[^|]*\bof=/dev/",
    r">\s*/dev/[sh]d[a-z]\b",
    r"\bfdisk\b",
    r"\bformat\s+[a-z]:",
    r"\bshred\b",
    # Privilege and ownership. Flags are lowercase because the input is.
    r"\bchmod\b(?:\s+-[a-z]+)*\s+(?:777|666)\s+/(?:\s|$)",
    r"\bchmod\b\s+-r\s+777\s+/(?:\s|$)",
    r"\bchown\b\s+-[a-z]*r[a-z]*\s+[^ ]+\s+/(?:\s|$)",
    r"\bpasswd\b",
    r"\bsudo\b",
    # Machine state.
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\binit\s+0\b",
    # Process and history destruction.
    r"\bkill(?:all)?\s+-9\s+1\b",
    r"\bhistory\s+-c\b",
    # A force push that cannot be recovered from. `--force-with-lease` is
    # the careful version and stays allowed.
    r"\bgit\s+push\b[^;|&]*--force(?!-with-lease)",
    # Fork bomb.
    r":\s*\(\s*\)\s*\{.*\|.*&.*\}\s*;?\s*:",
    # Redirecting over a shell profile, which makes the damage persist.
    r">\s*(?:~|\$home)?/?\.(?:bashrc|zshrc|profile|bash_profile)",
    # Piping a download straight into a shell.
    r"\b(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:ba|z|k|)?sh\b",
)


def is_command_blocked(command: str) -> tuple[bool, str]:
    """Check a command against the destructive blocklist.

    Args:
        command: The full command the model wants to run.

    Returns:
        `(blocked, reason)`. `reason` explains what tripped, so the model
        can report something useful instead of just "refused".
    """
    import re

    normalised = " ".join(command.split()).lower()
    for pattern in DESTRUCTIVE_PATTERNS:
        if re.search(pattern, normalised):
            return True, f"it matches a destructive-command rule: /{pattern}/"
    return False, ""


# --- Confirmations --------------------------------------------------------

# An optional replacement for the y/n prompt.
#
# It exists because asking is the one thing here that needs a terminal,
# and there are two situations where this module must not be the thing
# holding it. A test needs to answer without a person, and a full-screen
# curses UI owns the terminal, where `input()` produces unreadable
# output; the dashboard installs a handler that hands the screen back,
# asks on a normal terminal, and takes the screen back afterwards.
#
# None means the built-in prompt, which is what every run that does not
# install a handler gets. A callable takes the question and returns the
# answer, and is responsible for its own display.
_ASK_HANDLER: "Callable[[str], bool] | None" = None


def set_ask_handler(handler: "Callable[[str], bool] | None") -> None:
    """Install a replacement for the y/n prompt, or restore the default.

    Args:
        handler: Called with the question and returning the answer, or
            None to go back to the built-in prompt.
    """
    global _ASK_HANDLER
    _ASK_HANDLER = handler


def _ask(prompt: str) -> bool:
    """Ask a yes/no question on the terminal.

    Args:
        prompt: The question to put to the user.

    Returns:
        True for y/yes, False for anything else.

    If there is no terminal to ask on, the answer is no. Defaulting to
    "yes" here would mean an unattended run silently did whatever the
    model asked, which is the opposite of what a confirmation is for.

    An installed handler replaces the terminal entirely, including the
    no-terminal check: a handler that draws its own UI is responsible
    for refusing when it cannot ask.
    """
    if config.AUTO_APPROVE:
        _warn_auto_approve()
        return True
    if _ASK_HANDLER is not None:
        return bool(_ASK_HANDLER(prompt))
    if not sys.stdin.isatty():
        print(f"\n{prompt}\n[no terminal available - refusing]", file=sys.stderr)
        return False
    try:
        answer = input(f"\n{prompt}\n> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("", file=sys.stderr)
        return False
    return answer in {"y", "yes"}


_AUTO_APPROVE_WARNED = False


def _warn_auto_approve() -> None:
    """Say once, loudly, that nothing is being confirmed."""
    global _AUTO_APPROVE_WARNED
    if _AUTO_APPROVE_WARNED:
        return
    _AUTO_APPROVE_WARNED = True
    print(
        "\n" + "!" * 60 + "\n"
        "AUTO-APPROVAL IS ON (--yes). Every write, edit and command the\n"
        "model asks for runs without asking you. Blocked commands are\n"
        "still refused, but nothing else is. Use this only in a throwaway\n"
        "checkout or a container.\n" + "!" * 60,
        file=sys.stderr,
    )


def confirm_write(path: Path, content: str) -> bool:
    """Ask before creating or overwriting a file.

    Args:
        path: The resolved file that will be written.
        content: Exactly what will be written to it.

    Returns:
        True if the user approved.
    """
    verb = "OVERWRITE" if path.exists() else "create"
    shown = content if len(content) <= 2000 else content[:2000] + "\n... (cut short)"
    print(f"\n{'=' * 60}")
    print(f"The agent wants to {verb}: {path}")
    print(f"{'-' * 60}")
    print(shown)
    print(f"{'=' * 60}")
    return _ask("Write this file? [y/N] ")


def _as_lines(text: str) -> list[str]:
    """Split `text` into lines that all end in a newline.

    `difflib` needs this. An edit often replaces a fragment that has no
    trailing newline (`beta` inside `alpha\\nbeta\\ngamma`), and the
    missing newline runs the `-old` and `+new` lines together into
    something that reads as one mangled line.
    """
    return [line if line.endswith("\n") else line + "\n"
            for line in text.splitlines(keepends=True)] or ["\n"]


def confirm_edit(path: Path, old: str, new: str) -> bool:
    """Ask before editing a file, showing a diff of what will change.

    A diff rather than the two strings, because the question the user is
    answering is "does this do what I expect", and a diff answers that in
    a way two blobs of text do not.

    Args:
        path: The resolved file that will be changed.
        old: The text being replaced.
        new: The text replacing it.

    Returns:
        True if the user approved.
    """
    diff = difflib.unified_diff(
        _as_lines(old),
        _as_lines(new),
        fromfile=f"{path.name} (current)",
        tofile=f"{path.name} (proposed)",
        n=3,
    )
    body = "".join(diff)
    if len(body) > 4000:
        body = body[:4000] + "\n... (diff cut short)"
    print(f"\n{'=' * 60}")
    print(f"The agent wants to edit: {path}")
    print(f"{'-' * 60}")
    print(body)
    print(f"{'=' * 60}")
    return _ask("Apply this edit? [y/N] ")


def confirm_command(command: str) -> bool:
    """Ask before running a shell command, showing it in full.

    Args:
        command: The exact command that will be executed.

    Returns:
        True if the user approved.
    """
    print(f"\n{'=' * 60}")
    print("The agent wants to run this shell command:")
    print(f"{'-' * 60}")
    print(f"  {command}")
    print(f"{'-' * 60}")
    print(f"It runs in: {config.WORKSPACE_ROOT}")
    print(f"Timeout: {config.COMMAND_TIMEOUT_SECONDS:.0f}s")
    print(f"{'=' * 60}")
    return _ask("Run it? [y/N] ")
