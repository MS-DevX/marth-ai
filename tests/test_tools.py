"""Tests for the tool functions.

Phase 2 covers list_files and read_file. Each test builds its own tiny
project in a temp directory so the tests never depend on the real
workspace, and so a stray file cannot make them flaky.
"""

from pathlib import Path

import pytest

from agent import config, safety, tools


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a temp directory that looks like a small project.

    The sandbox root is redirected to it so the tools operate on this
    fixture instead of the real workspace. Tools deliberately take only a
    `path` argument (the model supplies it), so the root has to come from
    config rather than from a parameter.
    """
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "main.py").write_text("print('hi')\n")
    (tmp_path / "README.md").write_text("# Title\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret=1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "left-pad.js").write_text("//\n")
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


# --- list_files ----------------------------------------------------------


def test_list_files_shows_files_and_dirs(project: Path) -> None:
    result = tools.list_files(".")
    assert "README.md" in result
    assert "agent/" in result  # directories get a trailing slash


def test_list_files_skips_ignored_dirs(project: Path) -> None:
    result = tools.list_files(".")
    assert ".git" not in result
    assert "node_modules" not in result
    assert "config" not in result  # the file inside .git, not just the dir


def test_list_files_in_subdirectory(project: Path) -> None:
    assert tools.list_files("agent") == "main.py"


def test_list_files_reports_empty_dir(project: Path) -> None:
    (project / "empty").mkdir()
    assert "No files" in tools.list_files("empty")


def test_list_files_rejects_path_outside_workspace(project: Path) -> None:
    with pytest.raises(safety.SandboxError):
        tools.list_files("..")


# --- read_file -----------------------------------------------------------


def test_read_file_returns_contents(project: Path) -> None:
    assert tools.read_file("README.md") == "# Title\n"


def test_read_file_missing_path_is_an_error_not_a_crash(project: Path) -> None:
    assert "Not a file" in tools.read_file("nope.txt")


def test_read_file_on_a_directory_is_an_error(project: Path) -> None:
    assert "Not a file" in tools.read_file("agent")


def test_read_file_rejects_path_outside_workspace(project: Path) -> None:
    with pytest.raises(safety.SandboxError):
        tools.read_file("../secrets.txt")


def test_read_file_refuses_binary_content(project: Path) -> None:
    (project / "blob.bin").write_bytes(b"\x00\x01\x02\xff")
    assert "binary" in tools.read_file("blob.bin")


def test_read_file_refuses_oversized_file(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (project / "big.txt").write_text("x" * 5000)
    monkeypatch.setattr(config, "MAX_FILE_BYTES", 1000)
    assert "too large" in tools.read_file("big.txt")


# --- format_size ---------------------------------------------------------


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [(512, "512 B"), (2048, "2.0 KB"), (5 * 1024 * 1024, "5.0 MB")],
)
def test_format_size(num_bytes: int, expected: str) -> None:
    assert tools.format_size(num_bytes) == expected


# --- schemas -------------------------------------------------------------


def test_every_schema_names_a_registered_tool() -> None:
    """A schema the model can see but no tool to run would hang the loop."""
    for schema in tools.TOOL_SCHEMAS:
        assert schema["name"] in tools.TOOL_REGISTRY
        assert schema["description"].strip()
        assert schema["parameters"]["type"] == "object"


def test_read_file_schema_requires_a_path() -> None:
    schema = next(s for s in tools.TOOL_SCHEMAS if s["name"] == "read_file")
    assert schema["parameters"]["required"] == ["path"]
