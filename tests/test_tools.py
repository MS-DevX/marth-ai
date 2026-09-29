"""Tests for the tool functions.

Phase 2 covers list_files and read_file. Each test builds its own tiny
project in a temp directory so the tests never depend on the real
workspace, and so a stray file cannot make them flaky.
"""

from pathlib import Path

import pytest

from agent import config, safety, tools


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


# --- line ranges ----------------------------------------------------------
# A truncated read leaves the model unable to count lines or see the end of
# a file. Reading a range is the only way back, so it has to work.


def test_whole_file_read_is_unchanged(project: Path) -> None:
    (project / "a.txt").write_text("one\ntwo\nthree")
    assert tools.read_file("a.txt") == "one\ntwo\nthree"


def test_start_line_skips_the_beginning(project: Path) -> None:
    (project / "a.txt").write_text("one\ntwo\nthree\nfour")
    assert tools.read_file("a.txt", start_line=2) == "[lines 2-4 of 4]\ntwo\nthree\nfour"


def test_a_bounded_range_returns_only_that_range(project: Path) -> None:
    (project / "a.txt").write_text("\n".join(str(n) for n in range(1, 21)))
    out = tools.read_file("a.txt", start_line=5, end_line=8)
    assert out == "[lines 5-8 of 20]\n5\n6\n7\n8"


def test_range_says_where_it_came_from(project: Path) -> None:
    """Without the position the model cannot tell what it is looking at."""
    (project / "a.txt").write_text("\n".join(str(n) for n in range(1, 101)))
    assert "[lines 10-20 of 100]" in tools.read_file("a.txt", 10, 20)


def test_range_works_on_a_file_too_large_to_read_whole(project: Path) -> None:
    """The point of the range: get a whole section under the output cap."""
    body = "\n".join(f"line {n}" for n in range(1, 4001))
    (project / "big.txt").write_text(body)
    chunk = tools.read_file("big.txt", start_line=1, end_line=20)
    assert len(chunk) < config.MAX_OUTPUT_CHARS
    assert "line 1\n" in chunk and "line 20" in chunk
    assert "line 21" not in chunk


def test_range_past_the_end_says_how_long_the_file_is(project: Path) -> None:
    """Tells the model what it could have asked for instead."""
    (project / "a.txt").write_text("one\ntwo")
    out = tools.read_file("a.txt", start_line=99)
    assert "2 lines" in out


def test_reversed_range_is_reported(project: Path) -> None:
    (project / "a.txt").write_text("one\ntwo\nthree")
    assert "before start_line" in tools.read_file("a.txt", 5, 2)


def test_single_line_range(project: Path) -> None:
    (project / "a.txt").write_text("one\ntwo\nthree")
    assert tools.read_file("a.txt", 2, 2) == "[lines 2-2 of 3]\ntwo"


def test_secret_file_is_still_refused_with_a_range(project: Path) -> None:
    (project / ".env").write_text("KEY=secret")
    with pytest.raises(safety.SecretFileError):
        tools.read_file(".env", 1, 2)
