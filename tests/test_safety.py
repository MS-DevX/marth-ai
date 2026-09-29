"""Tests for the path sandbox, secret blocking, and output truncation.

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


# --- secret files ---------------------------------------------------------
# `.env` sits inside the workspace, so the boundary above does not stop the
# agent reading the API key out of it. These tests pin that hole shut.


def test_env_file_is_blocked(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("GEMINI_API_KEY=secret")
    with pytest.raises(safety.SecretFileError):
        safety.resolve_path(".env", tmp_path)


def test_env_file_is_blocked_by_absolute_path(tmp_path: Path) -> None:
    """Reaching the file by a different route must not help."""
    (tmp_path / ".env").write_text("GEMINI_API_KEY=secret")
    with pytest.raises(safety.SecretFileError):
        safety.resolve_path(tmp_path / ".env", tmp_path)


def test_env_file_is_blocked_from_a_subdirectory(tmp_path: Path) -> None:
    """The check runs on the resolved name, so `../` cannot rename it."""
    (tmp_path / ".env").write_text("GEMINI_API_KEY=secret")
    (tmp_path / "agent").mkdir()
    with pytest.raises(safety.SecretFileError):
        safety.resolve_path("agent/../.env", tmp_path)


def test_secret_block_is_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / ".ENV").write_text("GEMINI_API_KEY=secret")
    with pytest.raises(safety.SecretFileError):
        safety.resolve_path(".ENV", tmp_path)


@pytest.mark.parametrize("name", ["server.key", "cert.pem", "store.p12", "x.pfx"])
def test_key_and_certificate_files_are_blocked(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("material")
    with pytest.raises(safety.SecretFileError):
        safety.resolve_path(name, tmp_path)


@pytest.mark.parametrize(
    "name", ["id_rsa", "id_ed25519", ".netrc", ".npmrc", "credentials", "secrets"]
)
def test_known_credential_files_are_blocked(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("material")
    with pytest.raises(safety.SecretFileError):
        safety.resolve_path(name, tmp_path)


@pytest.mark.parametrize("name", [".env.example", ".env.sample", ".env.template"])
def test_env_templates_are_still_readable(tmp_path: Path, name: str) -> None:
    """The committed template documents the variables and holds no secrets."""
    (tmp_path / name).write_text("GEMINI_API_KEY=your_key_here")
    assert safety.resolve_path(name, tmp_path) == (tmp_path / name).resolve()


def test_a_file_merely_containing_env_is_not_blocked(tmp_path: Path) -> None:
    """Only the real names are blocked, not anything matching a substring."""
    (tmp_path / "environment.md").write_text("about env")
    assert safety.resolve_path("environment.md", tmp_path)


def test_secret_error_does_not_leak_the_path_outside(tmp_path: Path) -> None:
    """The message names the file but not its contents."""
    (tmp_path / ".env").write_text("GEMINI_API_KEY=super-secret-value")
    with pytest.raises(safety.SecretFileError) as caught:
        safety.resolve_path(".env", tmp_path)
    assert "super-secret-value" not in str(caught.value)


# --- truncation -----------------------------------------------------------


def test_short_text_is_returned_unchanged() -> None:
    assert safety.truncate("hello", limit=100) == "hello"


def test_text_at_the_limit_is_untouched() -> None:
    text = "x" * 100
    assert safety.truncate(text, limit=100) == text


def test_long_text_is_cut_to_the_limit() -> None:
    """The kept part must not exceed the cap, or the cap does nothing."""
    out = safety.truncate("y" * 5000, limit=100)
    assert out.count("y") == 100


def test_truncation_says_how_much_was_dropped() -> None:
    """Otherwise the model cannot tell a short file from a cut one."""
    out = safety.truncate("y" * 5000, limit=100)
    assert "4900" in out
    assert "5000" in out


def test_truncation_defaults_to_the_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_OUTPUT_CHARS", 50)
    out = safety.truncate("z" * 200)
    assert out.count("z") == 50


def test_truncation_suggests_a_way_forward() -> None:
    """Telling the model how to get less is what stops it retrying the same
    oversized read for another twenty steps."""
    assert "grep" in safety.truncate("z" * 5000, limit=100)
