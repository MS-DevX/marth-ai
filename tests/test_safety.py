"""Tests for the path sandbox.

`resolve_path` is the security boundary for every tool, so these tests are
the ones worth having early. The confirmation prompts and command blocklist
arrive with the write tools in Phase 4.
"""

from pathlib import Path

import pytest

from agent import config, safety


# --- paths inside the workspace -------------------------------------------


def test_relative_path_resolves_against_the_root(tmp_path: Path) -> None:
    assert safety.resolve_path("a/b.txt", tmp_path) == tmp_path / "a" / "b.txt"


def test_root_itself_is_allowed(tmp_path: Path) -> None:
    assert safety.resolve_path(".", tmp_path) == tmp_path.resolve()


def test_dot_dot_that_stays_inside_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    assert safety.resolve_path("sub/../a.txt", tmp_path) == tmp_path / "a.txt"


# --- paths outside the workspace -----------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../escape.txt",
        "../../escape.txt",
        "sub/../../escape.txt",
        "/etc/passwd",
        "/root/.ssh/id_rsa",
    ],
)
def test_paths_outside_the_root_are_rejected(path: str, tmp_path: Path) -> None:
    with pytest.raises(safety.SandboxError):
        safety.resolve_path(path, tmp_path)


def test_symlink_pointing_outside_is_rejected(tmp_path: Path) -> None:
    """A symlink is resolved before the check, so it cannot be used to escape."""
    (tmp_path / "innocent").symlink_to("/etc")
    with pytest.raises(safety.SandboxError):
        safety.resolve_path("innocent/passwd", tmp_path)


def test_symlinked_parent_directory_is_rejected(tmp_path: Path) -> None:
    """Escaping via a symlinked *parent*, not just the final component."""
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(outside)
    with pytest.raises(safety.SandboxError):
        safety.resolve_path("link/secret.txt", tmp_path)


# --- default root ---------------------------------------------------------


def test_falls_back_to_the_configured_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    assert safety.resolve_path("x.txt") == tmp_path / "x.txt"
    with pytest.raises(safety.SandboxError):
        safety.resolve_path("/etc/passwd")
