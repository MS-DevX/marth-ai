"""Getting the agent to a state where it can actually run a task.

Installing the package is only half of it. The agent also needs a model,
and the two halves fail differently: pip either works or prints why it
did not, whereas a missing model produces a 404 from the API that reads
like a bug in the agent. So this asks the question explicitly, and answers
it by downloading the answer.

The model itself is fetched by `llm.OllamaModels`, because that is the
only module allowed to make HTTP calls. This one decides what to ask and
what to tell the user, and writes the settings file that makes the choice
stick.
"""

import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .llm import OllamaModels, PullProgress

# Where Ollama's installer puts the binary on macOS and Linux when it is
# not on PATH. Checked because a fresh Ollama install frequently is not,
# and "command not found" from a subprocess is a worse answer than a
# path that works.
_FALLBACK_OLLAMA_PATHS = (
    Path.home() / ".local" / "opt" / "ollama" / "bin" / "ollama",
    Path("/usr/local/bin/ollama"),
    Path("/opt/homebrew/bin/ollama"),
)

# What to tell someone whose machine has no Ollama. Deliberately not
# attempted automatically: it is a large third-party install on the
# user's PATH, which is not a thing to do behind their back.
OLLAMA_INSTALL_HINT = (
    "Install Ollama from https://ollama.com/download, then re-run `marth --setup`."
)


@dataclass
class SetupReport:
    """What setup found, and what it changed.

    Returned rather than printed as it happens so the caller decides how
    to present it, and so a test can assert on the result without
    capturing output.
    """

    model: str
    ollama_path: str | None = None
    was_installed: bool = False
    downloaded: bool = False
    settings_written: Path | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True if the agent is ready to run a task."""
        return not self.problems


def _is_terminal() -> bool:
    """Return True if stdout is a terminal that can be redrawn.

    Decided once per setup run rather than per line, because the answer
    cannot change while the process is alive.
    """
    return sys.stdout.isatty()


def find_ollama() -> str | None:
    """Return the path to the ollama binary, or None if it is not installed.

    PATH first, then the places the official installer uses. A local model
    is the whole point of setup, so failing to find a binary that is
    sitting in ~/.local/opt would be a shame on a perfectly good machine.
    """
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in _FALLBACK_OLLAMA_PATHS:
        if candidate.is_file():
            return str(candidate)
    return None


def server_is_running(ollama_path: str | None = None) -> bool:
    """Return True if an Ollama server answers.

    Args:
        ollama_path: The binary to ask. Defaults to the one found on this
            machine.
    """
    models = OllamaModels()
    try:
        models.version()
    except RuntimeError:
        return False
    return True


class DownloadProgress:
    """Renders a model download for whatever it is being written to.

    Ollama sends several hundred updates for a model of any size. A
    terminal gets one line rewritten in place; anything else gets a line
    per state change and one per 10%, so `marth --setup > log` records
    that a download happened without filling the file with the same line
    forty times.

    Args:
        write: Where to write. A raw write, **not** a line printer: this
            overwrites one line, and a printer that appends a newline
            turns every update into its own line, which is the opposite
            of the point.
        stream: Whether the destination is a terminal that can be redrawn.
    """

    #: How often a non-terminal gets a line, as a percentage step. Every
    #: update would be accurate and unreadable.
    _MILESTONE = 10

    def __init__(self, write: Callable[[str], None], stream: bool) -> None:
        self._write = write
        self._stream = stream
        self._status = ""
        self._percent = -1
        self._width = 0
        self._committed = True

    def __call__(self, update: PullProgress) -> None:
        """Write one update.

        Args:
            update: What the server last said.
        """
        line = self._format(update)
        if self._stream:
            self._committed = False
            if update.status == "success":
                # No padding on the last line: a newline follows it, so
                # there is nothing left to overwrite.
                self._write("\r" + line)
                self.finish()
                return
            # Overwrite in place, padded so a shorter line does not leave
            # the tail of the longer one it replaced.
            self._width = max(self._width, len(line))
            self._write("\r" + line.ljust(self._width))
            return

        changed = update.status != self._status
        milestone = (
            update.percent is not None
            and update.percent // self._MILESTONE
            != (self._percent // self._MILESTONE if self._percent >= 0 else -1)
        )
        if changed or milestone or update.status == "success":
            self._write(line)
        self._status = update.status
        if update.percent is not None:
            self._percent = update.percent

    def finish(self) -> None:
        """End the progress line, so later output starts on a clean one.

        A terminal leaves the cursor sitting on the progress line after
        the last update, and whatever the caller prints next would be
        appended to it. Only writes anything if a line is still open, so
        the success case does not get a blank line under it.
        """
        if self._stream and not self._committed:
            self._write("\n")
        self._committed = True
        self._width = 0

    def _format(self, update: PullProgress) -> str:
        """Return the line to show for one update."""
        percent = f"{update.percent:>3}%" if update.percent is not None else "  ?%"
        return f"  {percent}  {update.status}"


def ensure_model(
    name: str,
    out: Callable[[str], None] = print,
    on_progress: Callable[[PullProgress], None] | None = None,
) -> SetupReport:
    """Check for `name` and download it if it is missing.

    Checks before downloading, because a 5GB pull that was already
    satisfied is the kind of thing a user notices immediately and never
    trusts again.

    Args:
        name: The model to make sure is present.
        out: Where progress lines go.
        on_progress: Called with each download update. Defaults to a
            reporter that suits `out`.

    Returns:
        A report saying what was found and what was done.
    """
    report = SetupReport(model=name)
    models = OllamaModels()

    try:
        version = models.version()
    except RuntimeError as exc:
        report.problems.append(str(exc))
        return report

    out(f"Ollama {version} is running at {models.root}.")
    report.ollama_path = models.root

    try:
        already = models.has_model(name)
    except RuntimeError as exc:
        report.problems.append(str(exc))
        return report

    if already:
        report.was_installed = True
        out(f"  {name} is already installed. Nothing to download.")
        return report

    out(f"  {name} is not installed. Downloading it now; this takes a while.")
    report.downloaded = True
    reporter = on_progress or DownloadProgress(sys.stdout.write, _is_terminal())
    try:
        models.pull(name, on_progress=reporter)
    except RuntimeError as exc:
        _finish(reporter)
        report.problems.append(str(exc))
        return report
    _finish(reporter)

    out(f"  {name} is installed.")
    return report


def _finish(reporter: object) -> None:
    """Close a progress reporter, if it is one that can be closed.

    A caller can substitute any callable for the reporter, so this cannot
    assume the interface; swallowing the absence is cheaper than a
    Protocol for one method.
    """
    closer = getattr(reporter, "finish", None)
    if callable(closer):
        closer()


def write_settings(
    model: str, config_dir: Path | None = None, provider: str = "openai"
) -> Path:
    """Write the settings file that makes this choice the default.

    A real environment variable still wins over the file, so a one-off
    `AGENT_MODEL=... marth ...` is unaffected by anything written here.

    Args:
        model: The model to record.
        config_dir: Where to write. Defaults to the user's config dir.
        provider: The provider to record.
    """
    target_dir = config_dir or config.CONFIG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / ".env"
    body = (
        "# Written by `marth --setup`. Edit freely, or delete this file to go\n"
        "# back to the defaults. A real environment variable overrides it.\n"
        f"AGENT_PROVIDER={provider}\n"
        f"AGENT_MODEL={model}\n"
    )
    target.write_text(body)
    return target


def run_setup(
    model: str | None = None,
    out: Callable[[str], None] = print,
    config_dir: Path | None = None,
) -> SetupReport:
    """Make the agent usable: find Ollama, fetch the model, save the choice.

    The order matters. Each step can only be attempted once the one before
    it is known to have worked, and each failure is a different fix for
    the user, so they are reported one at a time rather than as a list.

    Args:
        model: The model to install. Defaults to the recommended local one.
        out: Where progress lines go.
        config_dir: Where to write settings. Defaults to the user's.
    """
    wanted = (model or config.RECOMMENDED_LOCAL_MODEL).strip()
    report = SetupReport(model=wanted)

    ollama = find_ollama()
    if ollama is None:
        report.problems.append(f"Ollama is not installed. {OLLAMA_INSTALL_HINT}")
        out(f"  Ollama is not installed. {OLLAMA_INSTALL_HINT}")
        return report
    report.ollama_path = ollama
    out(f"Found Ollama at {ollama}.")

    if not server_is_running(ollama):
        started = _start_server(ollama, out)
        if not started:
            report.problems.append(
                "Ollama is installed but no server is running. Start it with "
                "`ollama serve` and re-run `marth setup`."
            )
            return report

    fetched = ensure_model(
        wanted,
        out=out,
        on_progress=DownloadProgress(sys.stdout.write, _is_terminal()),
    )
    report.was_installed = fetched.was_installed
    report.downloaded = fetched.downloaded
    if fetched.problems:
        report.problems.extend(fetched.problems)
        return report

    report.settings_written = write_settings(wanted, config_dir=config_dir)
    out(f"Saved your settings to {report.settings_written}.")
    return report


def _start_server(ollama: str, out: Callable[[str], None]) -> bool:
    """Try to start a background Ollama server, and say whether it took.

    Ollama normally runs as a login item, but a machine that has the
    binary and no server is common enough - a fresh install, a container,
    a stopped service - that setup should try before giving up. Spawned
    with its own session so it outlives the installer rather than dying
    with it.

    Args:
        ollama: The binary to run.
        out: Where progress lines go.
    """
    out("No server is running. Starting one.")
    try:
        subprocess.Popen(  # noqa: S603 - the path came from find_ollama
            [ollama, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        out(f"  Could not start it: {exc}")
        return False

    # The server binds its port before it answers anything, so polling
    # beats a fixed sleep: a fast machine is not made to wait for a slow
    # one, and the loop is bounded either way.
    for _ in range(config.SETUP_STARTUP_ATTEMPTS):
        if server_is_running(ollama):
            return True
        time.sleep(config.SETUP_STARTUP_POLL_SECONDS)
    return False
