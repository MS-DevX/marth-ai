"""Tests for how the agent decides where it lives and what it reads.

The module-level values in config are read once at import, which is right
for the running agent and useless for testing. The two decisions that
matter are therefore plain functions taking their inputs as arguments,
and that is what these tests call: a checkout and an install are told
apart the same way whether they are being exercised or merely run.
"""

import importlib
import os
from pathlib import Path

from dotenv import load_dotenv

from agent import config


# --- telling a checkout from an install ------------------------------------


def test_a_directory_with_pyproject_is_a_checkout(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    assert config.is_checkout(tmp_path) is True


def test_site_packages_is_not_a_checkout(tmp_path: Path) -> None:
    """The whole point. An install ships the package, not the file that
    built it, so the marker is absent and the agent must not treat a venv
    as a project directory."""
    (tmp_path / "agent").mkdir()
    assert config.is_checkout(tmp_path) is False


def test_this_repository_is_a_checkout() -> None:
    """Guards the assumption the rest of this file rests on. If the
    manifest is renamed or moved, the next tests stop meaning anything,
    and this is what notices."""
    assert config.is_checkout(config.PROJECT_ROOT) is True
    assert config._IS_CHECKOUT is True


# --- the workspace root ----------------------------------------------------


def test_a_checkout_works_on_its_own_directory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    assert config.default_workspace(True, project, elsewhere) == project


def test_an_install_works_on_the_users_directory(tmp_path: Path) -> None:
    """A regression this prevents: defaulting to project_root here means
    site-packages, so the sandbox would hold the agent's own dependencies
    and none of the code the user asked about."""
    project = tmp_path / "site-packages"
    cwd = tmp_path / "my-project"
    assert config.default_workspace(False, project, cwd) == cwd


def test_the_agent_in_this_checkout_still_sandboxes_the_project() -> None:
    """The change above must not move the boundary for a checkout. Most of
    the suite relies on this."""
    assert config.WORKSPACE_ROOT == config.PROJECT_ROOT


def test_agent_workspace_still_overrides_the_default(
    monkeypatch, tmp_path: Path
) -> None:
    """The escape hatch, which is the only way to point the installed
    agent somewhere other than the current directory."""
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.WORKSPACE_ROOT == tmp_path.resolve()
    finally:
        monkeypatch.delenv("AGENT_WORKSPACE")
        importlib.reload(config)


# --- the settings file -----------------------------------------------------


def test_a_checkout_reads_the_project_env_file(tmp_path: Path) -> None:
    got = config.env_file_for(True, tmp_path, tmp_path / "config")
    assert got == tmp_path / ".env"


def test_an_install_reads_the_per_user_file(tmp_path: Path) -> None:
    """Where `marth setup` writes, so the provider and model it set up are
    the ones the agent then uses."""
    got = config.env_file_for(False, tmp_path / "site-packages", tmp_path / "cfg")
    assert got == tmp_path / "cfg" / ".env"


def test_a_checkout_ignores_the_user_config(tmp_path: Path) -> None:
    """Exactly one file is ever read, and a developer's own ~/.config must
    not change what running from a checkout does."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("AGENT_MAX_STEPS=1\n")
    got = config.env_file_for(True, tmp_path, config_dir)
    assert got != config_dir / ".env"


def test_agents_config_dir_overrides_the_home_location(
    monkeypatch, tmp_path: Path
) -> None:
    """So a test cannot write to a real home directory."""
    monkeypatch.setenv("AGENT_CONFIG_DIR", str(tmp_path / "elsewhere"))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.CONFIG_DIR == tmp_path / "elsewhere"
    finally:
        monkeypatch.delenv("AGENT_CONFIG_DIR")
        importlib.reload(config)


def test_a_real_environment_variable_beats_the_settings_file(
    monkeypatch, tmp_path: Path
) -> None:
    """The documented override, and the reason load_dotenv is called
    without override=True. If the file won there would be no way to try a
    different model for one command without editing a file."""
    env = tmp_path / ".env"
    env.write_text("AGENT_MAX_STEPS=3\n")
    monkeypatch.setenv("AGENT_MAX_STEPS", "9")
    load_dotenv(env)
    assert os.environ["AGENT_MAX_STEPS"] == "9"


def test_a_settings_file_fills_in_a_value_the_environment_does_not_set(
    monkeypatch, tmp_path: Path
) -> None:
    """The other half of the same rule, and the reason setup writes a file
    at all: a model set in the file is still honoured."""
    env = tmp_path / ".env"
    env.write_text("AGENT_MODEL=some-model\n")
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    load_dotenv(env)
    assert os.environ["AGENT_MODEL"] == "some-model"
