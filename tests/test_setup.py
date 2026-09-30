"""Tests for `marth --setup`.

A fake Ollama server stands in for the real one. That is deliberate: the
flows worth testing here are "already installed", "not installed",
"server not running" and "downloaded but the name was wrong", and running
them against a real server would mean gigabyte downloads and a test
suite that only passes on a machine with Ollama already set up.

The one test that cannot be faked is the progress line, because whether it
works depends on what `sys.stdout` is. That one runs the real CLI in a
real pty, because a `print` where a raw write belongs passes every other
test in this file and then puts 400 lines on a user's screen.
"""

import json
import os
import pty
import re
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent import config, setup
from agent.llm import OllamaModels, _PullTally

REPO_ROOT = Path(__file__).resolve().parent.parent

# The layers a pull reports. Two, so that a percentage which is really
# only counting the current layer cannot pass by accident: with one layer
# the aggregate and the per-layer figure are the same number.
_LAYERS = (
    ("sha256:aaaa", 1000),
    ("sha256:bbbb", 3000),
)


class FakeOllama:
    """A stand-in for the bits of Ollama that setup talks to.

    Args:
        models: The names to report as already installed.
        reachable: Whether to answer at all, so the "no server" path can
            be exercised without stopping the real one.
    """

    def __init__(self, models: list[str], reachable: bool = True) -> None:
        self.models = list(models)
        self.reachable = reachable
        self.pulls: list[str] = []
        self.server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        """The URL to hand to OllamaModels, with the /v1 it strips."""
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def start(self) -> "FakeOllama":
        """Begin serving on a background thread."""
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                """Stay quiet; pytest captures stdout."""

            def do_GET(self) -> None:  # noqa: N802 - name fixed by base class
                if not fake.reachable:
                    self.send_error(503)
                    return
                if self.path == "/api/version":
                    self._json({"version": "0.0.0-fake"})
                elif self.path == "/api/tags":
                    self._json({"models": [{"name": n} for n in fake.models]})
                else:
                    self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802 - name fixed by base class
                if not fake.reachable:
                    self.send_error(503)
                    return
                if self.path != "/api/pull":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                wanted = json.loads(self.rfile.read(length)).get("name", "")
                fake.pulls.append(wanted)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for digest, size in _LAYERS:
                    for done in range(0, size + 1, size // 4):
                        self._line(
                            {
                                "status": f"pulling {digest[7:13]}",
                                "digest": digest,
                                "total": size,
                                "completed": done,
                            }
                        )
                self._line({"status": "success"})

            def _json(self, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _line(self, payload: dict) -> None:
                self.wfile.write(json.dumps(payload).encode() + b"\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the port."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch):
    """Return a factory for a running fake server, stopped afterwards.

    Also points the agent at it. Without that, a test that forgets to
    passes quietly: the code under test reads `config.OPENAI_BASE_URL`,
    finds the developer's real Ollama on localhost, and answers the
    question from whatever happens to be downloaded on this machine. That
    is not a failure, it is worse - the test passes for the wrong reason
    and only on the author's machine.
    """
    made: list[FakeOllama] = []

    def build(models: list[str], reachable: bool = True) -> FakeOllama:
        fake = FakeOllama(models, reachable).start()
        made.append(fake)
        monkeypatch.setattr(config, "OPENAI_BASE_URL", fake.base_url)
        return fake

    yield build
    for fake in made:
        fake.stop()


@pytest.fixture
def on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a fake `ollama` on PATH so the test does not need the real one.

    Without this, `marth --setup` on a machine with no Ollama installed
    stops at the first check and the progress line is never reached, so
    the one test that matters most would quietly test nothing.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "ollama"
    shim.write_text("#!/bin/sh\nexit 0\n")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return bindir


# --- deciding whether a model is already there -----------------------------


def test_an_exact_match_is_found(fake_ollama) -> None:
    models = OllamaModels(fake_ollama(["granite4.1:8b"]).base_url)
    assert models.has_model("granite4.1:8b") is True


def test_a_different_tag_is_not_the_same_model(fake_ollama) -> None:
    """A regression, and a nasty one. `qwen2.5:0.5b` and `qwen2.5:1.5b`
    share a name and are entirely different models. Matching on the name
    alone told setup the model was installed, skipped the download, and
    left every later run to fail with a 404."""
    models = OllamaModels(fake_ollama(["qwen2.5:0.5b"]).base_url)
    assert models.has_model("qwen2.5:1.5b") is False


def test_a_bare_name_does_not_match_a_tagged_one(fake_ollama) -> None:
    """Asking for `granite4.1` says nothing about which tag is the
    registry default. Reporting a 3b model as satisfying that would leave
    every run either using the wrong model or failing."""
    models = OllamaModels(fake_ollama(["granite4.1:3b"]).base_url)
    assert models.has_model("granite4.1") is False


def test_the_v1_suffix_is_stripped_from_the_url(fake_ollama) -> None:
    """The agent is configured with an OpenAI-shaped base URL, and the
    model API is not under /v1. Passing the configured value through
    unchanged would 404 on every check."""
    models = OllamaModels(fake_ollama([]).base_url)
    assert models.root.endswith("/api/version") is False
    assert models.version() == "0.0.0-fake"


# --- the check-then-download flow ------------------------------------------


def test_nothing_is_downloaded_when_the_model_is_there(fake_ollama) -> None:
    fake = fake_ollama(["granite4.1:8b"])
    out: list[str] = []
    report = setup.ensure_model("granite4.1:8b", out=out.append)

    assert report.ok is True
    assert report.was_installed is True
    assert report.downloaded is False
    assert fake.pulls == []
    assert any("already installed" in line for line in out)


def test_a_missing_model_is_downloaded(fake_ollama) -> None:
    fake = fake_ollama([])
    out: list[str] = []
    report = setup.ensure_model("granite4.1:8b", out=out.append)

    assert report.ok is True
    assert report.downloaded is True
    assert fake.pulls == ["granite4.1:8b"]


def test_a_missing_server_is_reported_rather_than_raised(fake_ollama) -> None:
    """A dead server and a missing model look identical from outside, and
    the fixes are different, so this has to say which one it found."""
    fake = fake_ollama([], reachable=False)
    report = setup.ensure_model("granite4.1:8b", out=lambda _line: None)

    assert report.ok is False
    assert "No Ollama server" in report.problems[0]


def test_a_failed_download_is_reported_not_raised(fake_ollama) -> None:
    """Setup returning a report with a problem is what the CLI turns into
    an exit code. An exception here would be a traceback instead."""
    fake = fake_ollama([])
    fake.stop()
    report = setup.ensure_model("granite4.1:8b", out=lambda _line: None)
    assert report.ok is False
    assert report.problems


# --- the settings file ------------------------------------------------------


def test_settings_name_the_provider_and_the_model(tmp_path: Path) -> None:
    target = setup.write_settings("granite4.1:8b", config_dir=tmp_path)
    body = target.read_text()
    assert "AGENT_PROVIDER=openai" in body
    assert "AGENT_MODEL=granite4.1:8b" in body


def test_the_settings_file_is_actually_loadable(tmp_path: Path) -> None:
    """Written by hand and parsed by the thing that will read it in anger.
    A file that only satisfies a substring check is not a config file."""
    from dotenv import dotenv_values

    target = setup.write_settings("granite4.1:8b", config_dir=tmp_path)
    values = dotenv_values(target)
    assert values["AGENT_PROVIDER"] == "openai"
    assert values["AGENT_MODEL"] == "granite4.1:8b"


def test_the_written_settings_point_at_the_right_file(tmp_path: Path) -> None:
    """`config.env_file_for` is what will look for this, so the two have to
    agree on the name or the install is silently ignored."""
    target = setup.write_settings("granite4.1:8b", config_dir=tmp_path)
    chosen = config.env_file_for(False, Path("/some/site-packages"), tmp_path)
    assert chosen == target


# --- the whole flow ---------------------------------------------------------


def test_setup_finishes_and_reports_success(
    fake_ollama, on_path, tmp_path: Path
) -> None:
    fake = fake_ollama([])
    out: list[str] = []
    report = setup.run_setup(
        "granite4.1:8b", out=out.append, config_dir=tmp_path
    )

    assert report.ok is True, report.problems
    assert report.downloaded is True
    assert report.settings_written == tmp_path / ".env"
    assert fake.pulls == ["granite4.1:8b"]


def test_setup_says_so_when_ollama_is_not_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(setup, "find_ollama", lambda: None)
    out: list[str] = []
    report = setup.run_setup("granite4.1:8b", out=out.append, config_dir=tmp_path)

    assert report.ok is False
    assert "not installed" in report.problems[0]
    assert "ollama.com" in report.problems[0]
    # Nothing was written, so a failed setup does not leave a config that
    # points at a model which was never downloaded.
    assert report.settings_written is None
    assert not (tmp_path / ".env").exists()


# --- the progress line ------------------------------------------------------


def test_a_terminal_gets_one_line_not_four_hundred() -> None:
    """The reason this test file exists. A reporter that goes through
    `print` appends a newline per update, so 400 updates become 400 lines
    and the user watches a scrollback instead of a progress bar."""
    from agent.llm import PullProgress

    written: list[str] = []
    reporter = setup.DownloadProgress(written.append, stream=True)
    for percent in range(0, 101, 1):
        reporter(PullProgress(status="pulling abc123", percent=percent))
    reporter.finish()

    raw = "".join(written)
    assert raw.count("\n") == 1, "expected a single rewritten line"
    assert raw.count("\r") == 101
    assert raw.endswith("\n")


def test_a_log_gets_a_line_per_milestone_not_per_update() -> None:
    from agent.llm import PullProgress

    written: list[str] = []
    reporter = setup.DownloadProgress(written.append, stream=False)
    for percent in range(0, 100):
        reporter(PullProgress(status="pulling abc123", percent=percent))
    reporter(PullProgress(status="success", percent=100))

    lines = written
    assert len(lines) < 20, f"a log should not hold {len(lines)} progress lines"
    assert lines[-1].endswith("success")
    assert any("%" in line for line in lines)


def test_the_percentage_covers_the_whole_model_not_one_layer() -> None:
    """Ollama reports progress per layer. Taken literally that is 100% for
    the first of two layers, on a download that is a quarter done, and the
    bar then races ahead of reality. The tally adds the layers up."""
    from agent.llm import _PullTally

    tally = _PullTally()
    # Both layers announce their size, as Ollama does, before either is
    # finished.
    tally.update(
        {"status": "pulling aaaaaa", "digest": "sha256:aaaa", "total": 1000,
         "completed": 1000}
    )
    tally.update(
        {"status": "pulling bbbbbb", "digest": "sha256:bbbb", "total": 3000,
         "completed": 0}
    )
    assert tally.percent() == 25

    tally.update(
        {"status": "pulling bbbbbb", "digest": "sha256:bbbb", "total": 3000,
         "completed": 1500}
    )
    assert tally.percent() == 62


def test_a_resent_chunk_does_not_inflate_the_percentage() -> None:
    """Ollama repeats the same `completed` value many times per layer. A
    naive sum would count each repeat and sail past the end."""
    from agent.llm import _PullTally

    tally = _PullTally()
    for _ in range(50):
        tally.update(
            {"status": "pulling a", "digest": "sha256:aaaa", "total": 1000,
             "completed": 500}
        )
    assert tally.percent() == 50


def test_the_bar_never_reaches_100_before_the_server_says_so() -> None:
    """Layers already on disk are never mentioned, so the denominator
    under-counts and a naive figure would hit 100% and then keep going."""
    from agent.llm import _PullTally

    tally = _PullTally()
    for _ in range(20):
        tally.update(
            {"status": "pulling a", "digest": "sha256:aaaa", "total": 100,
             "completed": 100}
        )
    assert tally.percent() == 99


def test_a_server_error_becomes_an_exception_not_a_silent_finish() -> None:
    """Ollama reports a failed pull in the stream, on a 200 response. Left
    unread, setup would report a model installed that is not there."""
    models = OllamaModels("http://127.0.0.1:9/v1")
    line = json.dumps({"error": "no space left on device"}).encode() + b"\n"
    with pytest.raises(RuntimeError, match="no space left"):
        models._read_progress(line, _PullTally())


def test_a_blank_line_in_the_stream_is_skipped() -> None:
    models = OllamaModels("http://127.0.0.1:9/v1")
    assert models._read_progress(b"\n", _PullTally()) is None


# --- the real CLI, in a real terminal ---------------------------------------


def run_in_pty(argv: list[str], env: dict[str, str]) -> str:
    """Run the CLI attached to a pty and return what a terminal would show.

    Args:
        argv: The command to run.
        env: Extra environment for the child.

    Returns:
        The output with carriage-return overwrites applied, which is what
        a person actually ends up looking at.
    """
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - runs in the child, never returns
        os.environ.update(env)
        os.execv(sys.executable, [sys.executable, *argv])

    # A pty defaults to 0x0, which makes anything that measures the
    # terminal decide it has no room and behave as if piped.
    import fcntl
    import termios

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 100, 0, 0))

    chunks: list[bytes] = []
    try:
        while True:
            piece = os.read(fd, 65536)
            if not piece:
                break
            chunks.append(piece)
    except OSError:
        pass
    os.close(fd)
    os.waitpid(pid, 0)

    raw = b"".join(chunks).decode("utf-8", "replace")
    # The pty turns every \n into \r\n. Undo that first, or the \r that
    # ends a line gets mistaken for a progress overwrite and every line
    # renders blank.
    text = raw.replace("\r\n", "\n")
    return "\n".join(chunk.split("\r")[-1] for chunk in text.split("\n"))


@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)
def test_the_cli_draws_one_progress_line_in_a_real_terminal(
    fake_ollama, on_path, tmp_path: Path
) -> None:
    """End to end, and the only test that would have caught the reporter
    being wired to `print` instead of `sys.stdout.write`.

    The fork warning is expected and safe here: the fake server runs in a
    thread so it can outlive this call, and the child immediately execs,
    which throws the inherited locks away rather than trying to release
    them. Filtered on this one test rather than suite-wide, so a genuine
    threading warning elsewhere is still visible.
    """
    fake = fake_ollama([])
    shown = run_in_pty(
        ["-m", "agent.main", "--setup", "--model", "granite4.1:8b"],
        {
            "PATH": os.environ["PATH"],
            "OPENAI_BASE_URL": fake.base_url,
            "AGENT_CONFIG_DIR": str(tmp_path / "cfg"),
        },
    )

    assert "is not installed" in shown
    assert "is installed." in shown
    progress = [line for line in shown.splitlines() if "%" in line]
    assert len(progress) == 1, f"expected one progress line, got {progress}"
    assert "success" in progress[0]
    assert "\r" not in shown


def test_the_cli_reports_a_missing_server_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, on_path, tmp_path: Path
) -> None:
    """Pointed at a port nothing is listening on."""
    shown = run_in_pty(
        ["-m", "agent.main", "--setup"],
        {
            "PATH": os.environ["PATH"],
            "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
            "AGENT_CONFIG_DIR": str(tmp_path / "cfg"),
            "AGENT_SETUP_ATTEMPTS": "1",
            "AGENT_SETUP_POLL": "0.05",
        },
    )
    assert "Setup did not finish" in shown
    assert "Traceback" not in shown
    assert shown.strip().splitlines()[-1].strip() != ""


def test_every_command_the_agent_tells_users_to_type_actually_works() -> None:
    """A regression, and one a user pays for directly.

    The hint said `marth setup` when the flag is `--setup`, so the one
    instruction given to somebody whose machine was not ready yet was a
    command that errors out. The same mistake was in the header of the
    settings file setup writes, which is a file people read and edit.

    Scanned across `setup.py` rather than one string, because these are
    exactly the places a name is typed by hand rather than derived, and
    they are invisible to any test that only checks behaviour.
    """
    from agent.main import build_parser

    known = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    source = Path(setup.__file__).read_text()
    mentioned = set(re.findall(r"marth (--?[a-z][a-z-]*)", source))
    assert mentioned, "setup.py should tell the user how to re-run it"

    unknown = mentioned - known - {"-h", "--help"}
    assert not unknown, f"setup.py names commands that do not exist: {unknown}"
